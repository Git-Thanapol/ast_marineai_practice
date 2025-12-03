import sys
import os
import csv
import time
import threading
import collections
import wget
import numpy as np
import torch
import torchaudio
import pyaudio
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from torch.cuda.amp import autocast

# --- IMPORT AST MODEL ---
# We assume this script is running inside the 'ast' directory
sys.path.append('./src')
from models.ast_models import ASTModel

# --- CONFIGURATION ---
SAMPLE_RATE = 16000
# AST requires ~10.24 seconds. 1024 frames * 10ms.
# We buffer slightly more to ensure we always have enough data.
BUFFER_SECONDS = 10.5 
CHUNK_SIZE = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL WRAPPER ---
class RealTimeAST:
    def __init__(self):
        self.download_models()
        print(f"Loading AST Model on {DEVICE}...")
        
        # Load labels
        self.labels = self.load_labels('./egs/audioset/data/class_labels_indices.csv')
        
        # Initialize Model
        # input_tdim=1024 corresponds to ~10s of audio
        self.model = ASTModel(label_dim=527, input_tdim=1024, imagenet_pretrain=False, audioset_pretrain=False)
        
        checkpoint_path = './pretrained_models/audio_mdl.pth'
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        
        # Handle DataParallel wrap if it exists in checkpoint
        if 'module.' in list(checkpoint.keys())[0]:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model = self.model.to(DEVICE)
        self.model.eval()
        
        # Normalization stats for AudioSet
        self.mean = -4.2677393
        self.std = 4.5689974

    def download_models(self):
        if not os.path.exists('./pretrained_models'):
            os.mkdir('./pretrained_models')
        if not os.path.exists('./pretrained_models/audio_mdl.pth'):
            print("Downloading Pretrained Weights...")
            wget.download('https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1', 
                          out='./pretrained_models/audio_mdl.pth')

    def load_labels(self, csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            lines = list(reader)
        labels = []
        for i1 in range(1, len(lines)):
            labels.append(lines[i1][2]) # Column 2 is the display name
        return labels

    def preprocess(self, audio_data):
        # audio_data is a numpy array (N,)
        # Convert to tensor
        waveform = torch.from_numpy(audio_data).unsqueeze(0).float()
        
        # Create Fbank
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, htk_compat=True, sample_frequency=SAMPLE_RATE, use_energy=False,
            window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
        
        # AST expects exactly 1024 frames. 
        target_length = 1024
        n_frames = fbank.shape[0]
        p = target_length - n_frames

        # Cut or Pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :] # Take the first 1024 frames (which is usually the end of buffer)

        # Normalize
        norm_fbank = (fbank - self.mean) / (self.std * 2)
        return fbank, norm_fbank

    def predict(self, norm_fbank):
        input_tensor = norm_fbank.unsqueeze(0).to(DEVICE) # Add batch dimension
        with torch.no_grad():
            # Use autocast if on GPU for speed, otherwise standard float32
            if DEVICE.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    output = self.model(input_tensor)
            else:
                output = self.model(input_tensor)
                
            output = torch.sigmoid(output)
            result = output.data.cpu().numpy()[0]
        
        # Get Top 5
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
        # Deque acts as a ring buffer. Max length = seconds * rate
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
            # Convert byte data to numpy array
            data = np.frombuffer(in_data, dtype=np.float32)
            self.audio_buffer.extend(data)
        return (in_data, pyaudio.paContinue)

    def get_buffer(self):
        return np.array(self.audio_buffer)

# --- GUI APPLICATION ---
class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Real-Time AST Audio Classifier")
        self.geometry("800x600")

        # Logic objects
        self.audio_stream = AudioStream()
        self.engine = RealTimeAST()
        self.is_listening = False

        # GUI Components
        self.create_widgets()
        
        # Update Loop Timer
        self.update_interval = 500 # ms (Update UI every 0.5 seconds)
        self.after(100, self.update_loop)

    def create_widgets(self):
        # 1. Top Section: Controls
        control_frame = tk.Frame(self)
        control_frame.pack(side=tk.TOP, pady=10)
        
        self.btn_listen = tk.Button(control_frame, text="Start Listening", 
                                    command=self.toggle_listening, 
                                    bg="green", fg="white", font=("Arial", 14))
        self.btn_listen.pack()

        # 2. Middle Section: Spectrogram
        self.fig, self.ax = plt.subplots(figsize=(8, 3))
        self.ax.set_title("Audio Spectrogram")
        self.ax.set_axis_off()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # 3. Bottom Section: Predictions
        pred_frame = tk.Frame(self)
        pred_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=20, padx=20)
        
        tk.Label(pred_frame, text="Top Predictions:", font=("Arial", 16, "bold")).pack(anchor="w")
        
        self.pred_labels = []
        for i in range(5):
            lbl = tk.Label(pred_frame, text=f"{i+1}. ...", font=("Courier", 14), anchor="w")
            lbl.pack(fill=tk.X)
            self.pred_labels.append(lbl)

    def toggle_listening(self):
        if not self.is_listening:
            self.is_listening = True
            self.btn_listen.config(text="Stop Listening", bg="red")
            self.audio_stream.start()
        else:
            self.is_listening = False
            self.btn_listen.config(text="Start Listening", bg="green")
            self.audio_stream.stop()

    def update_loop(self):
        if self.is_listening:
            # 1. Get Audio
            raw_audio = self.audio_stream.get_buffer()
            
            # Ensure we have enough data (at least 1 second) to display something
            if len(raw_audio) > SAMPLE_RATE:
                
                # 2. Preprocess (Get Spectrogram)
                spec, norm_spec = self.engine.preprocess(raw_audio)
                
                # 3. Predict
                preds = self.engine.predict(norm_spec)
                
                # 4. Update UI - Spectrogram
                self.ax.clear()
                # Transpose spec to make it look like a standard spectrogram (Time x Freq)
                # spec shape is [1024, 128], we want [128, 1024] for plotting
                self.ax.imshow(spec.t().numpy(), aspect='auto', origin='lower', cmap='inferno')
                self.ax.set_axis_off()
                self.canvas.draw()
                
                # 5. Update UI - Labels
                for i, (label, score) in enumerate(preds):
                    color = "black"
                    if i == 0: color = "blue" # Highlight top prediction
                    self.pred_labels[i].config(text=f"{label}: {score*100:.2f}%", fg=color)

        # Schedule next update
        self.after(self.update_interval, self.update_loop)

    def on_closing(self):
        self.audio_stream.stop()
        self.destroy()

if __name__ == "__main__":
    app = Application()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()