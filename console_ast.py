import sys
import os
import csv
import time
import collections
import wget
import numpy as np
import torch
import torchaudio
import pyaudio

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000
BUFFER_SECONDS = 10.5 
CHUNK_SIZE = 1024

print(f"--- STARTING AUDIO SPECTROGRAM TRANSFORMER ---")
print(f"Running on: {DEVICE}")

# --- IMPORT AST MODEL ---
# We use the import path exactly as you had it in your snippet
#sys.path.append('./src')
# try:
#     from src.models.ast_models import ASTModel
# except ImportError:
#     try:
#         # Fallback in case folder structure is different
#         from .src.models.ast_models import ASTModel
#     except ImportError:
#         print("CRITICAL ERROR: Could not find 'ASTModel'.") 
#         print("Make sure you are running this from the 'ast' folder and 'src/models/ast_models.py' exists.")
#         sys.exit(1)
from src.models.ast_models import ASTModel

# --- MODEL WRAPPER ---
class RealTimeAST:
    def __init__(self):
        self.download_models()
        print("Loading AST Model... (This takes 10-20 seconds)")
        
        # Load labels
        self.labels = self.load_labels('./egs/audioset/data/class_labels_indices.csv')
        
        # Initialize Model
        self.model = ASTModel(label_dim=527, input_tdim=1024, imagenet_pretrain=False, audioset_pretrain=False)
        
        checkpoint_path = './pretrained_models/audio_mdl.pth'
        # Load to CPU first to avoid memory issues, then move to DEVICE
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        
        # Handle DataParallel wrap if it exists
        if 'module.' in list(checkpoint.keys())[0]:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:] 
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model = self.model.to(DEVICE)
        self.model.eval()
        
        self.mean = -4.2677393
        self.std = 4.5689974
        print("Model Loaded Successfully.")

    def download_models(self):
        if not os.path.exists('./pretrained_models'):
            os.mkdir('./pretrained_models')
        if not os.path.exists('./pretrained_models/audio_mdl.pth'):
            print("Downloading Pretrained Weights...")
            wget.download('https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1', 
                          out='./pretrained_models/audio_mdl.pth')

    def load_labels(self, csv_path):
        if not os.path.exists(csv_path):
            print(f"ERROR: Cannot find label file at {csv_path}")
            sys.exit(1)
        with open(csv_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            lines = list(reader)
        labels = []
        for i1 in range(1, len(lines)):
            labels.append(lines[i1][2]) 
        return labels

    def preprocess(self, audio_data):
        # Convert to tensor
        waveform = torch.from_numpy(audio_data).unsqueeze(0).float()
        
        # Create Fbank
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, htk_compat=True, sample_frequency=SAMPLE_RATE, use_energy=False,
            window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
        
        # AST expects exactly 1024 frames
        target_length = 1024
        n_frames = fbank.shape[0]
        p = target_length - n_frames

        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :] 

        norm_fbank = (fbank - self.mean) / (self.std * 2)
        return norm_fbank

    def predict(self, norm_fbank):
        input_tensor = norm_fbank.unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            # Disable autocast on CPU to avoid warnings/errors
            if DEVICE.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    output = self.model(input_tensor)
            else:
                output = self.model(input_tensor)
                
            output = torch.sigmoid(output)
            result = output.data.cpu().numpy()[0]
        
        sorted_indexes = np.argsort(result)[::-1]
        top_results = []
        for k in range(5):
            label = self.labels[sorted_indexes[k]]
            score = result[sorted_indexes[k]]
            top_results.append((label, score))
            
        return top_results

# --- AUDIO THREAD ---
class AudioStream:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.running = False
        self.buffer_len = int(SAMPLE_RATE * BUFFER_SECONDS)
        self.audio_buffer = collections.deque(maxlen=self.buffer_len)

    def start(self):
        self.running = True
        self.stream = self.p.open(format=pyaudio.paFloat32,
                                  channels=1,
                                  rate=SAMPLE_RATE,
                                  input=True,
                                  frames_per_buffer=CHUNK_SIZE,
                                  stream_callback=self.callback)
        self.stream.start_stream()

    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

    def callback(self, in_data, frame_count, time_info, status):
        if self.running:
            data = np.frombuffer(in_data, dtype=np.float32)
            self.audio_buffer.extend(data)
        return (in_data, pyaudio.paContinue)

    def get_buffer(self):
        return np.array(self.audio_buffer)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    try:
        # Initialize
        engine = RealTimeAST()
        stream = AudioStream()
        
        print("\n" + "="*40)
        print("  SYSTEM READY - LISTENING TO MICROPHONE")
        print("="*40)
        print("Press Ctrl+C to stop.\n")
        
        stream.start()
        
        while True:
            # 1. Get Audio
            raw_audio = stream.get_buffer()
            
            # 2. Check if we have enough data (wait for buffer to fill slightly)
            if len(raw_audio) > SAMPLE_RATE:
                
                # 3. Predict
                norm_spec = engine.preprocess(raw_audio)
                preds = engine.predict(norm_spec)
                
                # 4. Clear Screen (Windows specific command)
                os.system('cls')
                
                # 5. Print Results
                print("\n" + "="*40)
                print("   REAL-TIME AUDIO CLASSIFICATION")
                print("="*40)
                print(f"Status: Listening... (Buffer: {len(raw_audio)/SAMPLE_RATE:.1f}s)")
                print("-" * 40)
                
                for i, (label, score) in enumerate(preds):
                    # Create a visual bar for the score
                    bar_len = int(score * 30)
                    bar = "█" * bar_len + "░" * (30 - bar_len)
                    
                    # Highlight top prediction
                    prefix = ">> " if i == 0 else "   "
                    print(f"{prefix}{label:<25} [{bar}] {score*100:.1f}%")
                    
                print("-" * 40)
                
            # Update every 0.5 seconds
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nStopping...")
        stream.stop()
        sys.exit(0)
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        input("Press Enter to exit...")