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
import cv2  # OpenCV for the cool UI

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000
BUFFER_SECONDS = 10.5 
CHUNK_SIZE = 1024

# UI Configuration
WINDOW_NAME = "AST Audio Classifier - Real Time"
WIDTH = 800
HEIGHT_SPEC = 300
HEIGHT_PRED = 300

print(f"--- STARTING DASHBOARD ---")
print(f"Running on: {DEVICE}")

# --- IMPORT AST MODEL ---
sys.path.append('./src')
try:
    from models.ast_models import ASTModel
except ImportError:
    try:
        from models import ASTModel
    except ImportError:
        print("Error: Could not find ASTModel. Make sure you are in the 'ast' folder.")
        sys.exit(1)

# --- MODEL WRAPPER ---
class RealTimeAST:
    def __init__(self):
        self.download_models()
        print("Loading AST Model... (Please wait)")
        
        self.labels = self.load_labels('./egs/audioset/data/class_labels_indices.csv')
        
        # Initialize Model
        self.model = ASTModel(label_dim=527, input_tdim=1024, imagenet_pretrain=False, audioset_pretrain=False)
        
        checkpoint_path = './pretrained_models/audio_mdl.pth'
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        
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

    def download_models(self):
        if not os.path.exists('./pretrained_models'):
            os.mkdir('./pretrained_models')
        if not os.path.exists('./pretrained_models/audio_mdl.pth'):
            print("Downloading weights...")
            wget.download('https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1', 
                          out='./pretrained_models/audio_mdl.pth')

    def load_labels(self, csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            lines = list(reader)
        labels = []
        for i1 in range(1, len(lines)):
            labels.append(lines[i1][2]) 
        return labels

    def preprocess(self, audio_data):
        waveform = torch.from_numpy(audio_data).unsqueeze(0).float()
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, htk_compat=True, sample_frequency=SAMPLE_RATE, use_energy=False,
            window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
        
        target_length = 1024
        n_frames = fbank.shape[0]
        p = target_length - n_frames

        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :] 

        norm_fbank = (fbank - self.mean) / (self.std * 2)
        return fbank, norm_fbank

    def predict(self, norm_fbank):
        input_tensor = norm_fbank.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
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

# --- AUDIO STREAM ---
class AudioStream:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.running = False
        self.buffer_len = int(SAMPLE_RATE * BUFFER_SECONDS)
        self.audio_buffer = collections.deque(maxlen=self.buffer_len)

    def start(self):
        self.running = True
        self.stream = self.p.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK_SIZE, stream_callback=self.callback)
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

# --- VISUALIZATION HELPERS ---
def create_spectrogram_image(spec_tensor):
    # spec_tensor is [1024, 128]
    # Transpose to [128, 1024] (Freq on Y, Time on X)
    spec_img = spec_tensor.t().numpy()
    
    # Normalize to 0-255
    min_val = spec_img.min()
    max_val = spec_img.max()
    if max_val - min_val > 0:
        spec_img = 255 * (spec_img - min_val) / (max_val - min_val)
    else:
        spec_img = spec_img * 0
        
    spec_img = spec_img.astype(np.uint8)
    
    # Resize to fit UI
    spec_img = cv2.resize(spec_img, (WIDTH, HEIGHT_SPEC))
    
    # Apply Cool Colormap (Inferno/Fire look)
    spec_color = cv2.applyColorMap(spec_img, cv2.COLORMAP_INFERNO)
    return spec_color

def create_dashboard(spec_img, preds):
    # Create black canvas for predictions
    pred_img = np.zeros((HEIGHT_PRED, WIDTH, 3), dtype=np.uint8)
    
    # Header
    cv2.putText(pred_img, "TOP PREDICTIONS", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Bars
    y = 90
    for i, (label, score) in enumerate(preds):
        # Bar color (Green for top, white for others)
        color = (0, 255, 0) if i == 0 else (200, 200, 200)
        
        # Draw Bar Background
        cv2.rectangle(pred_img, (250, y-20), (WIDTH-50, y), (50, 50, 50), -1)
        
        # Draw Score Bar
        bar_width = int((WIDTH - 300) * score)
        cv2.rectangle(pred_img, (250, y-20), (250 + bar_width, y), color, -1)
        
        # Text Label
        text = f"{i+1}. {label}"
        cv2.putText(pred_img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Score Text
        score_txt = f"{score*100:.1f}%"
        cv2.putText(pred_img, score_txt, (WIDTH-120, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        y += 50
        
    # Stack images vertically
    dashboard = np.vstack([spec_img, pred_img])
    return dashboard

# --- MAIN LOOP ---
if __name__ == "__main__":
    try:
        engine = RealTimeAST()
        stream = AudioStream()
        
        print("Starting stream...")
        stream.start()
        
        while True:
            # 1. Get Data
            raw_audio = stream.get_buffer()
            
            if len(raw_audio) > SAMPLE_RATE:
                # 2. Process
                spec, norm_spec = engine.preprocess(raw_audio)
                preds = engine.predict(norm_spec)
                
                # 3. Visualize
                spec_img = create_spectrogram_image(spec)
                dashboard = create_dashboard(spec_img, preds)
                
                # 4. Show
                cv2.imshow(WINDOW_NAME, dashboard)
            
            # Quit on 'q' key
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
                
        stream.stop()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error: {e}")
        # Fallback to pure console if something goes wrong even here
        input("Press Enter to exit...")