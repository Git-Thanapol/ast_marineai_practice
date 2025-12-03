import sys
import os
import csv
import time
import collections
import wget
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import pyaudio
import cv2
import timm

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000
BUFFER_SECONDS = 10.5 
CHUNK_SIZE = 1024

# UI Layout
WIDTH = 1400  
HEIGHT_SPEC = 250
HEIGHT_PRED = 400
WINDOW_NAME = "Architecture Battle: AST vs ResNet vs EfficientNet"

print(f"--- STARTING BATTLE ---")
print(f"Running on: {DEVICE}")

# --- IMPORT AST ---
# Ensure local src folder is found
sys.path.append('./src')
try:
    from models.ast_models import ASTModel
except ImportError:
    try:
        from src.models.ast_models import ASTModel
    except ImportError:
        print("Error: Could not import ASTModel. Ensure 'ast_models.py' is in './src/models/'")
        sys.exit(1)

# ==========================================
# MODEL MANAGER
# ==========================================
class BattleManager:
    def __init__(self):
        self.download_ast_weights()
        self.labels = self.load_labels('./egs/audioset/data/class_labels_indices.csv')
        
        # 1. AST (The Specialist)
        print("\nLoading Model 1: AST (Audio Spectrogram Transformer)...")
        print("   -> Loading AudioSet Pretrained Weights (High Accuracy)")
        self.ast = ASTModel(label_dim=527, input_tdim=1024, imagenet_pretrain=False, audioset_pretrain=False)
        self.load_ast_weights(self.ast, './pretrained_models/audio_mdl.pth')
        self.ast.to(DEVICE).eval()
        
        # 2. ResNet-50 (The Heavy Vision Standard)
        print("\nLoading Model 2: ResNet-50...")
        print("   -> Adapting ImageNet weights (1-channel input). needs finetuning for high accuracy.")
        # in_chans=1 allows us to pass the spectrogram directly as a greyscale image
        self.resnet = timm.create_model('resnet50', pretrained=True, num_classes=527, in_chans=1)
        self.resnet.to(DEVICE).eval()
        
        # 3. EfficientNet (The Modern Lightweight)
        print("\nLoading Model 3: EfficientNet-B0...")
        print("   -> Adapting ImageNet weights (1-channel input). needs finetuning for high accuracy.")
        self.effnet = timm.create_model('efficientnet_b0', pretrained=True, num_classes=527, in_chans=1)
        self.effnet.to(DEVICE).eval()
        
        # Normalization stats for AST (we will use same for others for consistency)
        self.mean = -4.2677393
        self.std = 4.5689974
        print("\nAll Models Loaded Successfully.")

    def download_ast_weights(self):
        if not os.path.exists('./pretrained_models'): os.mkdir('./pretrained_models')
        if not os.path.exists('./pretrained_models/audio_mdl.pth'):
            print("Downloading AST Weights...")
            wget.download('https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1', out='./pretrained_models/audio_mdl.pth')

    def load_ast_weights(self, model, path):
        # Fix for weights_only issue in newer PyTorch
        try:
            checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=DEVICE)

        if 'module.' in list(checkpoint.keys())[0]:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                new_state_dict[k.replace('module.', '')] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint)

    def load_labels(self, csv_path):
        if not os.path.exists(csv_path):
            # Fallback labels if CSV missing
            return [f"Class {i}" for i in range(527)]
        with open(csv_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            lines = list(reader)
        return [lines[i][2] for i in range(1, len(lines))]

    def preprocess(self, raw_audio):
        # Convert to tensor
        waveform = torch.from_numpy(raw_audio).unsqueeze(0).float()
        
        # Create Fbank (Same as AST requirements)
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, htk_compat=True, sample_frequency=SAMPLE_RATE, use_energy=False,
            window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
        
        # Pad or Crop to 1024 frames
        target_length = 1024
        n_frames = fbank.shape[0]
        p = target_length - n_frames
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]
            
        # Normalize
        norm_fbank = (fbank - self.mean) / (self.std * 2)
        return norm_fbank

    def predict(self, raw_audio):
        # Preprocess once for all models
        spectrogram = self.preprocess(raw_audio)
        
        # Prepare inputs
        # AST expects: (Batch, Time, Freq) -> (1, 1024, 128)
        input_ast = spectrogram.unsqueeze(0).to(DEVICE)
        
        # ResNet/EffNet expect: (Batch, Channels, Height, Width) -> (1, 1, 1024, 128)
        input_vision = input_ast.unsqueeze(1)

        results = {}
        
        with torch.no_grad():
            # 1. AST Inference
            t0 = time.time()
            if DEVICE.type == 'cuda':
                with torch.amp.autocast('cuda'): out_ast = self.ast(input_ast)
            else:
                out_ast = self.ast(input_ast)
            out_ast = torch.sigmoid(out_ast)
            results['ast'] = (self._get_top_5(out_ast), (time.time() - t0) * 1000)

            # 2. ResNet-50 Inference
            t0 = time.time()
            if DEVICE.type == 'cuda':
                with torch.amp.autocast('cuda'): out_res = self.resnet(input_vision)
            else:
                out_res = self.resnet(input_vision)
            # Timm models return logits, need sigmoid for multi-label audio classification
            out_res = torch.sigmoid(out_res) 
            results['resnet'] = (self._get_top_5(out_res), (time.time() - t0) * 1000)

            # 3. EfficientNet Inference
            t0 = time.time()
            if DEVICE.type == 'cuda':
                with torch.amp.autocast('cuda'): out_eff = self.effnet(input_vision)
            else:
                out_eff = self.effnet(input_vision)
            out_eff = torch.sigmoid(out_eff)
            results['effnet'] = (self._get_top_5(out_eff), (time.time() - t0) * 1000)

        results['spec'] = spectrogram.cpu()
        return results

    def _get_top_5(self, output_tensor):
        res = output_tensor.data.cpu().numpy()[0]
        sorted_idx = np.argsort(res)[::-1]
        return [(self.labels[sorted_idx[k]], res[sorted_idx[k]]) for k in range(5)]

# --- AUDIO STREAM ---
class AudioStream:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.running = False
        self.buffer_len = int(SAMPLE_RATE * BUFFER_SECONDS)
        self.buffer = collections.deque(maxlen=self.buffer_len)

    def start(self):
        self.running = True
        self.stream = self.p.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK_SIZE, stream_callback=self.callback)
        self.stream.start_stream()

    def stop(self):
        self.running = False
        if self.stream: self.stream.stop_stream(); self.stream.close()

    def callback(self, in_data, frame_count, time_info, status):
        if self.running: self.buffer.extend(np.frombuffer(in_data, dtype=np.float32))
        return (in_data, pyaudio.paContinue)

    def get_data(self):
        return np.array(self.buffer)

# --- VISUALIZATION ---
def draw_ui(spec_tensor, results):
    # 1. Spectrogram
    spec_img = spec_tensor.numpy() # (1024, 128)
    spec_img = spec_img.T # (128, 1024)
    spec_img = cv2.normalize(spec_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    spec_img = cv2.resize(spec_img, (WIDTH, HEIGHT_SPEC))
    spec_img = cv2.applyColorMap(spec_img, cv2.COLORMAP_INFERNO)
    
    # 2. Panels
    panel = np.zeros((HEIGHT_PRED, WIDTH, 3), dtype=np.uint8)
    w_p = WIDTH // 3
    
    # Dividers
    cv2.line(panel, (w_p, 20), (w_p, HEIGHT_PRED-20), (50, 50, 50), 2)
    cv2.line(panel, (w_p*2, 20), (w_p*2, HEIGHT_PRED-20), (50, 50, 50), 2)
    
    def draw_col(idx, name, color, data, latency):
        x_offset = idx * w_p
        # Header
        cv2.putText(panel, name, (x_offset+20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(panel, f"Inference: {latency:.1f}ms", (x_offset+20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        
        # Bars
        y = 120
        for i, (lbl, scr) in enumerate(data):
            bar_c = color if i==0 else (100, 100, 100)
            max_bar_width = w_p - 150
            bar_w = int(max_bar_width * scr)
            
            # Draw Bar
            cv2.rectangle(panel, (x_offset+20, y-20), (x_offset+20+bar_w, y), bar_c, -1)
            # Draw Label
            cv2.putText(panel, f"{lbl[:25]}", (x_offset+20, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            # Draw Score
            cv2.putText(panel, f"{scr*100:.0f}%", (x_offset+20+bar_w+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            y += 55

    # Draw Columns
    # EfficientNet (Lightweight)
    draw_col(0, "EfficientNet-B0", (0, 255, 255), results['effnet'][0], results['effnet'][1])
    # ResNet (Standard)
    draw_col(1, "ResNet-50", (255, 100, 255), results['resnet'][0], results['resnet'][1])
    # AST (Specialized)
    draw_col(2, "AST (Transformer)", (0, 150, 255), results['ast'][0], results['ast'][1])       

    return np.vstack([spec_img, panel])

# --- MAIN ---
if __name__ == "__main__":
    try:
        manager = BattleManager()
        stream = AudioStream()
        
        print("\n" + "="*50)
        print("  BATTLE READY - LISTENING TO MICROPHONE")
        print("="*50)
        print("Note: ResNet & EfficientNet are running with ImageNet weights adapted for audio.")
        print("      They demonstrate architecture speed/output, but accuracy will favor AST.")
        print("Press 'q' to quit.")
        
        stream.start()
        
        while True:
            raw = stream.get_data()
            if len(raw) > SAMPLE_RATE:
                res = manager.predict(raw)
                frame = draw_ui(res['spec'], res)
                cv2.imshow(WINDOW_NAME, frame)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
        stream.stop()
        cv2.destroyAllWindows()
        
    except KeyboardInterrupt:
        print("\nStopping...")
        stream.stop()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")