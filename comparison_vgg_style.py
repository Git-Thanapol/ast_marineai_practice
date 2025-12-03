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
CAPTURE_RATE = 32000
BUFFER_SECONDS = 10.5 
CHUNK_SIZE = 1024

# UI Layout
WIDTH = 1400  
HEIGHT_SPEC = 250
HEIGHT_PRED = 400
WINDOW_NAME = "AI Battle: CNN14 vs CNN10 vs AST"

print(f"--- STARTING TRI-MODEL BATTLE ---")
print(f"Running on: {DEVICE}")

# --- IMPORT AST ---
sys.path.append('./src')
try:
    from models.ast_models import ASTModel
except ImportError:
    from models import ASTModel

# ==========================================
# SHARED BLOCKS
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

# ==========================================
# 1. DEFINE CNN14 (The Heavy Standard)
# ==========================================
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
    
    def forward(self, input):
        x = input.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        x = self.conv_block1(x); x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = self.conv_block2(x); x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = self.conv_block3(x); x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = self.conv_block4(x); x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = self.conv_block5(x); x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = self.conv_block6(x); x = F.avg_pool2d(x, kernel_size=(1, 1))
        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.relu(self.fc1(x))
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        return clipwise_output

# ==========================================
# 2. DEFINE CNN10 (VGGish Equivalent)
# ==========================================
# CNN10 is essentially VGG-10. It has 4 blocks of 2 layers (8 layers) + FCs = 10 layers.
# This makes it very similar to VGGish (11 layers) in complexity.
class Cnn10(nn.Module):
    def __init__(self, classes_num=527):
        super(Cnn10, self).__init__()
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)
    
    def forward(self, input):
        x = input.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        x = self.conv_block1(x); x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = self.conv_block2(x); x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = self.conv_block3(x); x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = self.conv_block4(x); x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.relu(self.fc1(x))
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        return clipwise_output

# ==========================================
# 3. MANAGER CLASS
# ==========================================
class TriModelManager:
    def __init__(self):
        self.check_downloads()
        self.labels = self.load_labels('./egs/audioset/data/class_labels_indices.csv')
        
        # --- LOAD AST ---
        print("Loading AST (Transformer)...")
        self.ast = ASTModel(label_dim=527, input_tdim=1024, imagenet_pretrain=False, audioset_pretrain=False)
        self.load_model(self.ast, './pretrained_models/audio_mdl.pth')
        self.ast.to(DEVICE).eval()
        
        # --- LOAD CNN14 ---
        print("Loading CNN14 (Heavy)...")
        self.cnn = Cnn14(classes_num=527)
        self.load_model(self.cnn, './pretrained_models/Cnn14_mAP=0.431.pth', strict=False)
        self.cnn.to(DEVICE).eval()
        
        # --- LOAD CNN10 (VGGish Equivalent) ---
        print("Loading CNN10 (VGG-Style)...")
        self.vgg_like = Cnn10(classes_num=527)
        self.load_model(self.vgg_like, './pretrained_models/Cnn10_mAP=0.380.pth', strict=False)
        self.vgg_like.to(DEVICE).eval()
        
        self.resampler = torchaudio.transforms.Resample(CAPTURE_RATE, 16000).to(DEVICE)
        print("All Models Loaded Successfully.")

    def check_downloads(self):
        if not os.path.exists('./pretrained_models'): os.mkdir('./pretrained_models')
        
        if not os.path.exists('./pretrained_models/audio_mdl.pth'):
            print("Downloading AST...")
            wget.download('https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1', out='./pretrained_models/audio_mdl.pth')
            
        if not os.path.exists('./pretrained_models/Cnn14_mAP=0.431.pth'):
            print("Downloading CNN14...")
            wget.download('https://zenodo.org/records/3987831/files/Cnn14_mAP%3D0.431.pth?download=1', out='./pretrained_models/Cnn14_mAP=0.431.pth')
            
        # Download CNN10
        if not os.path.exists('./pretrained_models/Cnn10_mAP=0.380.pth'):
            print("Downloading CNN10 (VGG-Style)...")
            wget.download('https://zenodo.org/records/3987831/files/Cnn10_mAP%3D0.380.pth?download=1', out='./pretrained_models/Cnn10_mAP=0.380.pth')

    def load_model(self, model, path, strict=True):
        try:
            # Fix for PyTorch 2.6+ 'weights_only' error
            ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=DEVICE)
            
        if 'model' in ckpt: ckpt = ckpt['model']
        if 'module.' in list(ckpt.keys())[0]:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in ckpt.items():
                new_state_dict[k.replace('module.', '')] = v
            ckpt = new_state_dict
        model.load_state_dict(ckpt, strict=strict)

    def load_labels(self, csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            lines = list(reader)
        return [lines[i][2] for i in range(1, len(lines))]

    def predict(self, raw_audio_32k):
        waveform_32k = torch.from_numpy(raw_audio_32k).unsqueeze(0).float().to(DEVICE)
        
        # --- 32k Input for PANNs ---
        fbank_32k = torchaudio.compliance.kaldi.fbank(
            waveform_32k, htk_compat=True, sample_frequency=32000, use_energy=False,
            window_type='hanning', num_mel_bins=64, dither=0.0, frame_shift=10)
        
        if fbank_32k.shape[0] < 1000:
            fbank_32k = torch.nn.ZeroPad2d((0, 0, 0, 1000 - fbank_32k.shape[0]))(fbank_32k)
        else:
            fbank_32k = fbank_32k[:1000, :]
        input_32k = fbank_32k.unsqueeze(0).unsqueeze(0)
        
        # --- 16k Input for AST ---
        waveform_16k = self.resampler(waveform_32k)
        fbank_ast = torchaudio.compliance.kaldi.fbank(
            waveform_16k, htk_compat=True, sample_frequency=16000, use_energy=False,
            window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
        
        if fbank_ast.shape[0] < 1024:
            fbank_ast = torch.nn.ZeroPad2d((0, 0, 0, 1024 - fbank_ast.shape[0]))(fbank_ast)
        else:
            fbank_ast = fbank_ast[:1024, :]
        norm_ast = (fbank_ast - (-4.2677393)) / (4.5689974 * 2)
        input_ast = norm_ast.unsqueeze(0)

        # --- INFERENCE ---
        with torch.no_grad():
            # 1. CNN14
            t0 = time.time()
            if DEVICE.type == 'cuda':
                with torch.amp.autocast('cuda'): out_cnn = self.cnn(input_32k)
            else: out_cnn = self.cnn(input_32k)
            t_cnn = (time.time() - t0) * 1000

            # 2. CNN10 (VGG-Style)
            t0 = time.time()
            if DEVICE.type == 'cuda':
                with torch.amp.autocast('cuda'): out_vgg = self.vgg_like(input_32k)
            else: out_vgg = self.vgg_like(input_32k)
            t_vgg = (time.time() - t0) * 1000
            
            # 3. AST
            t0 = time.time()
            if DEVICE.type == 'cuda':
                with torch.amp.autocast('cuda'): out_ast = torch.sigmoid(self.ast(input_ast))
            else: out_ast = torch.sigmoid(self.ast(input_ast))
            t_ast = (time.time() - t0) * 1000

        return {
            'cnn': (self._get_top_5(out_cnn), t_cnn),
            'vgg': (self._get_top_5(out_vgg), t_vgg),
            'ast': (self._get_top_5(out_ast), t_ast),
            'spec': norm_ast.cpu()
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
def draw_ui(spec_tensor, results):
    # 1. Spectrogram
    spec_img = spec_tensor.t().numpy()
    spec_img = cv2.normalize(spec_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    spec_img = cv2.resize(spec_img, (WIDTH, HEIGHT_SPEC))
    spec_img = cv2.applyColorMap(spec_img, cv2.COLORMAP_INFERNO)
    
    # 2. Panels
    panel = np.zeros((HEIGHT_PRED, WIDTH, 3), dtype=np.uint8)
    w_p = WIDTH // 3
    
    cv2.line(panel, (w_p, 20), (w_p, HEIGHT_PRED-20), (100, 100, 100), 2)
    cv2.line(panel, (w_p*2, 20), (w_p*2, HEIGHT_PRED-20), (100, 100, 100), 2)
    
    def draw_col(idx, name, color, data, latency):
        x_offset = idx * w_p
        cv2.putText(panel, name, (x_offset+20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(panel, f"Latency: {latency:.1f}ms", (x_offset+20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        y = 120
        for i, (lbl, scr) in enumerate(data):
            bar_c = color if i==0 else (100, 100, 100)
            bar_w = int((w_p - 150) * scr)
            cv2.rectangle(panel, (x_offset+20, y-20), (x_offset+20+bar_w, y), bar_c, -1)
            cv2.putText(panel, f"{lbl}", (x_offset+20, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(panel, f"{scr*100:.0f}%", (x_offset+20+bar_w+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y += 55

    draw_col(0, "CNN14 (Heavy)", (255, 200, 0), results['cnn'][0], results['cnn'][1])       
    draw_col(1, "CNN10 (VGG-Style)", (50, 255, 50), results['vgg'][0], results['vgg'][1])   
    draw_col(2, "AST (Transformer)", (0, 100, 255), results['ast'][0], results['ast'][1])         

    return np.vstack([spec_img, panel])

# --- MAIN ---
if __name__ == "__main__":
    try:
        manager = TriModelManager()
        stream = AudioStream()
        stream.start()
        print("Stream started. Press 'q' to quit.")
        
        while True:
            raw = stream.get_data()
            if len(raw) > CAPTURE_RATE:
                res = manager.predict(raw)
                frame = draw_ui(res['spec'], res)
                cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(50) & 0xFF == ord('q'): break
            
        stream.stop()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error: {e}")
        input("Press Enter to exit...")