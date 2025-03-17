#!/usr/bin/env python
import sys
import subprocess
import importlib.util
import os
import warnings
import threading
import time
import queue

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def install(package):
    """Install a package via pip."""
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

def check_and_install(packages):
    """Check for each package and install it if it is not found."""
    for package in packages:
        if importlib.util.find_spec(package) is None:
            print(f"Package '{package}' not found. Installing ...")
            install(package)

# Add required packages including UI libraries
required_packages = ["torch", "sounddevice", "numpy", "transformers", "accelerate", "matplotlib", "customtkinter"]
check_and_install(required_packages)

# Suppress the Hugging Face symlinks warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import customtkinter as ctk
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from threading import Thread
import tkinter as tk
from tkinter import scrolledtext

class AudioAnalyzer:
    def __init__(self, update_interval=0.1):
        self.update_interval = update_interval
        self.audio_queue = queue.Queue()
        self.intensity = 0
        self.running = False
    
    def analyze_audio_stream(self):
        def callback(indata, frames, time, status):
            if status:
                print(status)
            # Calculate audio intensity (RMS amplitude)
            audio_data = indata.copy()
            rms = np.sqrt(np.mean(audio_data**2)) * 100
            self.intensity = min(1.0, rms)  # Normalize to 0-1 range
            self.audio_queue.put(audio_data[:, 0])  # Store only one channel
            
        with sd.InputStream(callback=callback, channels=1, samplerate=16000):
            while self.running:
                time.sleep(self.update_interval)
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.analyze_audio_stream)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1)
    
    def get_intensity(self):
        return self.intensity

def load_asr_pipeline(model_id, cache_dir):
    """
    Loads the automatic speech recognition (ASR) pipeline with the given model.
    Uses the provided cache_dir to store/retrieve the model files.
    """
    # Use a device index for the pipeline (0 for GPU, -1 for CPU)
    device = 0 if torch.cuda.is_available() else -1
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Loading model from {model_id}...")
    
    # This call will load the model (downloading it if necessary)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )

    # Move the model to the appropriate device
    if torch.cuda.is_available():
        model.to("cuda")
    else:
        model.to("cpu")

    # Load the processor holding the tokenizer and feature extractor
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)

    # Create the pipeline for automatic speech recognition
    # FIX: Move max_new_tokens into generate_kwargs
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
        generate_kwargs={"max_new_tokens": 128}
    )
    return asr_pipe

class SpeechTranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Speech Transcription App")
        self.root.geometry("800x600")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        ctk.set_appearance_mode("dark")  # Options: "dark", "light"
        ctk.set_default_color_theme("blue")
        
        self.recording = False
        self.audio_analyzer = AudioAnalyzer()
        self.recording_duration = 5
        self.transcription_active = False
        self.transcription_thread = None
        
        self.setup_ui()
        self.load_model()
        
    def setup_ui(self):
        # Create frames
        self.top_frame = ctk.CTkFrame(self.root)
        self.top_frame.pack(fill="x", padx=10, pady=10)
        
        self.middle_frame = ctk.CTkFrame(self.root)
        self.middle_frame.pack(fill="x", padx=10, pady=5)
        
        self.bottom_frame = ctk.CTkFrame(self.root)
        self.bottom_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Top Frame - Controls
        self.record_button = ctk.CTkButton(
            self.top_frame, 
            text="Start Recording", 
            command=self.toggle_recording,
            width=120
        )
        self.record_button.pack(side="left", padx=10, pady=10)
        
        self.duration_label = ctk.CTkLabel(self.top_frame, text="Recording Duration (s):")
        self.duration_label.pack(side="left", padx=10, pady=10)
        
        self.duration_slider = ctk.CTkSlider(
            self.top_frame, 
            from_=1, 
            to=10, 
            number_of_steps=9,
            command=self.update_duration
        )
        self.duration_slider.set(5)
        self.duration_slider.pack(side="left", padx=10, pady=10, fill="x", expand=True)
        
        self.duration_value_label = ctk.CTkLabel(self.top_frame, text="5s")
        self.duration_value_label.pack(side="left", padx=10, pady=10)
        
        self.clear_button = ctk.CTkButton(
            self.top_frame, 
            text="Clear Transcript", 
            command=self.clear_transcript,
            width=120
        )
        self.clear_button.pack(side="left", padx=10, pady=10)
        
        # Middle Frame - Microphone Visualizer
        self.fig, self.ax = plt.subplots(figsize=(8, 1.5), facecolor='#2b2b2b')
        self.ax.set_facecolor('#2b2b2b')
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_title('Microphone Intensity', color='white')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        self.intensity_bar = self.ax.barh(0.5, 0, height=0.7, color='#1f6aa5')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.middle_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Status label
        self.status_label = ctk.CTkLabel(self.middle_frame, text="Ready")
        self.status_label.pack(pady=5)
        
        # Bottom Frame - Transcription Display
        self.transcript_label = ctk.CTkLabel(self.bottom_frame, text="Transcription:")
        self.transcript_label.pack(anchor="w", padx=10, pady=5)
        
        self.transcript_area = scrolledtext.ScrolledText(
            self.bottom_frame,
            wrap=tk.WORD,
            font=("TkDefaultFont", 12)
        )
        self.transcript_area.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Set up animation for the intensity bar
        self.animation = FuncAnimation(self.fig, self.update_intensity_bar, interval=100, blit=False)
    
    def load_model(self):
        def load_model_thread():
            self.status_label.configure(text="Loading model... This may take a while")
            
            model_id = "distil-whisper/distil-large-v3"
            cache_dir = "./model_cache"
            cache_marker = os.path.join(cache_dir, "distil_large_v3_downloaded.marker")
            
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            
            self.asr_pipe = load_asr_pipeline(model_id, cache_dir)
            
            if not os.path.exists(cache_marker):
                with open(cache_marker, "w") as f:
                    f.write("downloaded")
            
            self.root.after(0, lambda: self.status_label.configure(text="Ready"))
        
        Thread(target=load_model_thread, daemon=True).start()
    
    def update_intensity_bar(self, frame):
        if hasattr(self, 'audio_analyzer'):
            intensity = self.audio_analyzer.get_intensity()
            self.intensity_bar[0].set_width(intensity)
            
            # Change color based on intensity
            if intensity < 0.3:
                color = '#3498db'  # Blue
            elif intensity < 0.6:
                color = '#f39c12'  # Orange
            else:
                color = '#e74c3c'  # Red
            
            self.intensity_bar[0].set_color(color)
        return self.intensity_bar
    
    def update_duration(self, value):
        self.recording_duration = int(value)
        self.duration_value_label.configure(text=f"{self.recording_duration}s")
    
    def toggle_recording(self):
        if not self.transcription_active:
            self.start_transcription()
        else:
            self.stop_transcription()
    
    def start_transcription(self):
        self.transcription_active = True
        self.record_button.configure(text="Stop Recording", fg_color="#e74c3c")
        self.status_label.configure(text="Transcribing...")
        self.audio_analyzer.start()
        
        def transcription_thread():
            while self.transcription_active:
                # Record audio
                self.status_label.configure(text="Recording...")
                audio = sd.rec(
                    int(self.recording_duration * 16000), 
                    samplerate=16000, 
                    channels=1, 
                    dtype="float32"
                )
                sd.wait()
                
                if not self.transcription_active:
                    break
                
                # Process audio
                self.status_label.configure(text="Processing...")
                audio = np.squeeze(audio)
                audio_input = {"array": audio, "sampling_rate": 16000}
                
                try:
                    result = self.asr_pipe(audio_input)
                    transcription = result["text"].strip()
                    
                    # Update UI from main thread
                    self.root.after(0, lambda t=transcription: self.update_transcript(t))
                except Exception as e:
                    print(f"Error transcribing: {e}")
                    self.root.after(0, lambda: self.status_label.configure(
                        text=f"Error: {str(e)[:50]}..."
                    ))
        
        self.transcription_thread = Thread(target=transcription_thread, daemon=True)
        self.transcription_thread.start()
    
    def stop_transcription(self):
        self.transcription_active = False
        self.record_button.configure(text="Start Recording", fg_color="#1f6aa5")
        self.status_label.configure(text="Stopped")
        self.audio_analyzer.stop()
    
    def update_transcript(self, text):
        if text.strip():  # Only add non-empty transcriptions
            current_time = time.strftime("%H:%M:%S")
            self.transcript_area.insert(tk.END, f"[{current_time}] {text}\n\n")
            self.transcript_area.see(tk.END)  # Scroll to the bottom
    
    def clear_transcript(self):
        self.transcript_area.delete(1.0, tk.END)
    
    def on_close(self):
        if self.transcription_active:
            self.stop_transcription()
        if self.animation:
            self.animation.event_source.stop()
        self.root.destroy()

def main():
    root = ctk.CTk()
    app = SpeechTranscriptionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()