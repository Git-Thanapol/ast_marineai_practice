import sys
import os
import csv
import time
import collections
import wget
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import pyaudio
import cv2

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CAPTURE_RATE = 32000 # CNN14 likes 32k, AST likes 16k. We capture 32k and downsample.
BUFFER_SECONDS = 10.5 
CHUNK_SIZE = 1024

# UI Layout
WIDTH = 1200
HEIGHT_SPEC = 250
HEIGHT_PRED = 400
WINDOW_NAME = "Model Comparison: CNN14 vs AST"

print(f"--- STARTING DUAL MODEL COMPARISON ---")
print(f"Running on: {DEVICE}")

# --- IMPORT AST ---
sys.path.append('./src')
try:
    from models.ast_models import ASTModel
except ImportError:
    from models import ASTModel

# ==========================================
# 1. DEFINE CNN14 ARCHITECTURE (Compact)
# ==========================================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class Cnn14(nn.Module):
    def __init__(self, classes_num=527):
        super(Cnn14, self).__init__()
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
    
    def forward(self, input, mixup_lambda=None):
        # input shape: (batch, 1, time, freq)
        x = input.transpose(1, 3) # PANNs expects (batch, 1, freq, time)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        x = self.conv_block1(x)
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = self.conv_block2(x)
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = self.conv_block3(x)
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = self.conv_block4(x)
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = self.conv_block5(x)
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = self.conv_block6(x)
        x = F.avg_pool2d(x, kernel_size=(1, 1))

        x = torch.mean(x, dim=3) # Pool time
        (x1, _) = torch.max(x, dim=2) # Max pool freq
        x2 = torch.mean(x, dim=2)     # Avg pool freq
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        return clipwise_output

# ==========================================
# 2. MANAGER CLASS
# ==========================================
class DualModelManager:
    def __init__(self):
        self.check_downloads()
        self.labels = self.load_labels('./egs/audioset/data/class_labels_indices.csv')
        
        # --- LOAD AST ---
        print("Loading AST Model...")
        self.ast = ASTModel(label_dim=527, input_tdim=1024, imagenet_pretrain=False, audioset_pretrain=False)
        ast_ckpt = torch.load('./pretrained_models/audio_mdl.pth', map_location=DEVICE)
        self.ast.load_state_dict(self._fix_state_dict(ast_ckpt))
        self.ast.to(DEVICE).eval()
        
        # --- LOAD CNN14 ---
        print("Loading CNN14 Model...")
        self.cnn = Cnn14(classes_num=527)
        cnn_ckpt = torch.load('./pretrained_models/Cnn14_mAP=0.431.pth', map_location=DEVICE)
        self.cnn.load_state_dict(self._fix_state_dict(cnn_ckpt), strict=False)
        self.cnn.to(DEVICE).eval()
        
        # Resampler for AST (32k -> 16k)
        self.resampler = torchaudio.transforms.Resample(CAPTURE_RATE, 16000).to(DEVICE)
        
        print("Models Loaded.")

    def check_downloads(self):
        if not os.path.exists('./pretrained_models'): os.mkdir('./pretrained_models')
        # AST Weights
        if not os.path.exists('./pretrained_models/audio_mdl.pth'):
            print("Downloading AST weights...")
            wget.download('https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1', out='./pretrained_models/audio_mdl.pth')
        # CNN14 Weights
        if not os.path.exists('./pretrained_models/Cnn14_mAP=0.431.pth'):
            print("Downloading CNN14 weights...")
            wget.download('https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1', out='./pretrained_models/Cnn14_mAP=0.431.pth')

    def load_labels(self, csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            lines = list(reader)
        return [lines[i][2] for i in range(1, len(lines))]

    def _fix_state_dict(self, ckpt):
        # Removes 'module.' prefix if present
        if 'model' in ckpt: ckpt = ckpt['model']
        if 'module.' in list(ckpt.keys())[0]:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in ckpt.items():
                new_state_dict[k.replace('module.', '')] = v
            return new_state_dict
        return ckpt

    def predict(self, raw_audio_32k):
        # raw_audio_32k is numpy array
        waveform_32k = torch.from_numpy(raw_audio_32k).unsqueeze(0).float().to(DEVICE)
        
        # --- PREPARE DATA FOR CNN14 (32k, 64 mels) ---
        fbank_cnn = torchaudio.compliance.kaldi.fbank(
            waveform_32k, htk_compat=True, sample_frequency=32000, use_energy=False,
            window_type='hanning', num_mel_bins=64, dither=0.0, frame_shift=10)
        
        # Pad/Cut CNN input to ~10s (1000 frames)
        if fbank_cnn.shape[0] < 1000:
            fbank_cnn = torch.nn.ZeroPad2d((0, 0, 0, 1000 - fbank_cnn.shape[0]))(fbank_cnn)
        else:
            fbank_cnn = fbank_cnn[:1000, :]
            
        # PANNs expects (batch, 1, time, freq)
        input_cnn = fbank_cnn.unsqueeze(0).unsqueeze(0)
        
        # --- PREPARE DATA FOR AST (16k, 128 mels) ---
        waveform_16k = self.resampler(waveform_32k)
        fbank_ast = torchaudio.compliance.kaldi.fbank(
            waveform_16k, htk_compat=True, sample_frequency=16000, use_energy=False,
            window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
        
        # AST Normalization
        target_len_ast = 1024
        if fbank_ast.shape[0] < target_len_ast:
            fbank_ast = torch.nn.ZeroPad2d((0, 0, 0, target_len_ast - fbank_ast.shape[0]))(fbank_ast)
        else:
            fbank_ast = fbank_ast[:target_len_ast, :]
            
        norm_ast = (fbank_ast - (-4.2677393)) / (4.5689974 * 2)
        input_ast = norm_ast.unsqueeze(0)

        # --- INFERENCE ---
        with torch.no_grad():
            # Run CNN14
            t0 = time.time()
            if DEVICE.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    out_cnn = self.cnn(input_cnn)
            else:
                out_cnn = self.cnn(input_cnn)
            cnn_time = (time.time() - t0) * 1000
            
            # Run AST
            t0 = time.time()
            if DEVICE.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    out_ast = torch.sigmoid(self.ast(input_ast))
            else:
                out_ast = torch.sigmoid(self.ast(input_ast))
            ast_time = (time.time() - t0) * 1000

        return {
            'cnn_preds': self._get_top_5(out_cnn),
            'cnn_time': cnn_time,
            'ast_preds': self._get_top_5(out_ast),
            'ast_time': ast_time,
            'spec': norm_ast.cpu() # Return AST spec for viz
        }

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
        self.buffer = collections.deque(maxlen=int(CAPTURE_RATE * BUFFER_SECONDS))

    def start(self):
        self.running = True
        self.stream = self.p.open(format=pyaudio.paFloat32, channels=1, rate=CAPTURE_RATE, input=True, frames_per_buffer=CHUNK_SIZE, stream_callback=self.callback)
        self.stream.start_stream()

    def stop(self):
        self.running = False
        if self.stream: self.stream.stop_stream(); self.stream.close()

    def callback(self, in_data, frame_count, time_info, status):
        if self.running: self.buffer.extend(np.frombuffer(in_data, dtype=np.float32))
        return (in_data, pyaudio.paContinue)
    
    def get_data(self): return np.array(self.buffer)

# --- VISUALIZATION ---
def draw_ui(spec_tensor, cnn_data, ast_data, cnn_ms, ast_ms):
    # 1. Spectrogram
    spec_img = spec_tensor.t().numpy()
    spec_img = cv2.normalize(spec_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    spec_img = cv2.resize(spec_img, (WIDTH, HEIGHT_SPEC))
    spec_img = cv2.applyColorMap(spec_img, cv2.COLORMAP_INFERNO)
    
    # 2. Prediction Panel (Split L/R)
    panel = np.zeros((HEIGHT_PRED, WIDTH, 3), dtype=np.uint8)
    mid_x = WIDTH // 2
    
    # Draw Divider
    cv2.line(panel, (mid_x, 20), (mid_x, HEIGHT_PRED-20), (100, 100, 100), 2)
    
    # --- LEFT: CNN14 ---
    cv2.putText(panel, "CNN14 (Baseline)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0), 2)
    cv2.putText(panel, f"Latency: {cnn_ms:.1f}ms", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    
    y = 120
    for i, (lbl, scr) in enumerate(cnn_data):
        color = (255, 200, 0) if i==0 else (100, 100, 100) # Cyan/Blueish for CNN
        bar_w = int((mid_x - 150) * scr)
        cv2.rectangle(panel, (20, y-20), (20+bar_w, y), color, -1)
        cv2.putText(panel, f"{lbl}", (20, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(panel, f"{scr*100:.1f}%", (20+bar_w+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 55

    # --- RIGHT: AST ---
    cv2.putText(panel, "AST (Transformer)", (mid_x + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
    cv2.putText(panel, f"Latency: {ast_ms:.1f}ms", (mid_x + 20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    
    y = 120
    for i, (lbl, scr) in enumerate(ast_data):
        color = (0, 100, 255) if i==0 else (100, 100, 100) # Orange for AST
        bar_w = int((mid_x - 150) * scr)
        cv2.rectangle(panel, (mid_x+20, y-20), (mid_x+20+bar_w, y), color, -1)
        cv2.putText(panel, f"{lbl}", (mid_x+20, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(panel, f"{scr*100:.1f}%", (mid_x+20+bar_w+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 55

    return np.vstack([spec_img, panel])

# --- MAIN ---
if __name__ == "__main__":
    try:
        manager = DualModelManager()
        stream = AudioStream()
        stream.start()
        
        print("Stream started. Press 'q' to quit.")
        
        while True:
            raw = stream.get_data()
            if len(raw) > CAPTURE_RATE:
                # Predict
                res = manager.predict(raw)
                
                # Draw
                frame = draw_ui(res['spec'], res['cnn_preds'], res['ast_preds'], res['cnn_time'], res['ast_time'])
                cv2.imshow(WINDOW_NAME, frame)
            
            if cv2.waitKey(50) & 0xFF == ord('q'): break
            
        stream.stop()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error: {e}")
        input("Press Enter to exit...")