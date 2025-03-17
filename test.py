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
warnings.filterwarnings("ignore", category=UserWarning, message=".*frames=None.*")

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
required_packages = [
    "torch", 
    "sounddevice", 
    "numpy", 
    "transformers", 
    "accelerate", 
    "matplotlib", 
    "customtkinter"
]
check_and_install(required_packages)

# Import modules
import torch

# When a package might not be available at static analysis time, we try to import
# it inside a try/except block, installing it on the fly if needed.
try:
    import sounddevice as sd  # type: ignore
except ImportError:
    install("sounddevice")
    import sounddevice as sd  # type: ignore

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    import customtkinter as ctk  # type: ignore
except ImportError:
    install("customtkinter")
    import customtkinter as ctk  # type: ignore

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from threading import Thread
import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox

# Define available models with their details
AVAILABLE_MODELS = {
    "openai/whisper-tiny": {
        "name": "Whisper Tiny",
        "description": "Fast but less accurate",
        "size_mb": 150,
        "recommended": False
    },
    "openai/whisper-base": {
        "name": "Whisper Base",
        "description": "Good balance for basic use",
        "size_mb": 290,
        "recommended": False
    },
    "openai/whisper-small": {
        "name": "Whisper Small",
        "description": "Better accuracy, moderate size",
        "size_mb": 990,
        "recommended": False
    },
    "distil-whisper/distil-small": {
        "name": "Distil Whisper Small",
        "description": "Faster than original small, similar accuracy",
        "size_mb": 400,
        "recommended": False
    },
    "distil-whisper/distil-medium.en": {
        "name": "Distil Whisper Medium (English)",
        "description": "English optimized, good performance",
        "size_mb": 770,
        "recommended": False
    },
    "openai/whisper-medium": {
        "name": "Whisper Medium",
        "description": "High accuracy, larger size",
        "size_mb": 3000,
        "recommended": False
    },
    "distil-whisper/distil-large-v3": {
        "name": "Distil Whisper Large v3",
        "description": "Best balance of accuracy and speed",
        "size_mb": 1500,
        "recommended": True
    },
}

class AudioAnalyzer:
    def __init__(self, update_interval=0.05):
        self.update_interval = update_interval
        self.audio_queue = queue.Queue()
        self.intensity = 0
        self.running = False
        self.buffer = np.zeros(100)  # Buffer to smooth intensity values
        self.buffer_index = 0
        self.samplerate = 16000
        
    def analyze_audio_stream(self):
        def callback(indata, frames, time_info, status):
            if status:
                print(status)
            # Calculate audio intensity (RMS amplitude)
            audio_data = indata.copy()
            # Apply proper normalization and scaling for better sensitivity
            rms = np.sqrt(np.mean(audio_data ** 2)) * 150
            
            # Smooth the intensity using the buffer
            self.buffer[self.buffer_index] = rms
            self.buffer_index = (self.buffer_index + 1) % len(self.buffer)
            self.intensity = min(1.0, np.mean(self.buffer))
            
            self.audio_queue.put(audio_data[:, 0])  # Store only one channel
            
        try:
            with sd.InputStream(callback=callback, channels=1, samplerate=self.samplerate):
                while self.running:
                    time.sleep(self.update_interval)
        except Exception as e:
            print(f"Error in audio stream: {e}")
    
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
    
    def get_audio_chunk(self, duration_seconds):
        """Record audio for the specified duration"""
        print(f"Recording audio for {duration_seconds} seconds...")
        audio = sd.rec(
            int(duration_seconds * self.samplerate), 
            samplerate=self.samplerate, 
            channels=1, 
            dtype="float32"
        )
        sd.wait()
        return np.squeeze(audio)

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
    # Move generate_kwargs to the proper place to avoid the warning
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
        max_new_tokens=128
    )
    return asr_pipe

class ModelSelectionFrame(ctk.CTkFrame):
    def __init__(self, master, models_dict, on_model_selected, **kwargs):
        super().__init__(master, **kwargs)
        self.models_dict = models_dict
        self.on_model_selected = on_model_selected
        self.selected_model_id = None
        
        # Create model selection UI
        self.setup_ui()
    
    def setup_ui(self):
        # Title
        self.title_label = ctk.CTkLabel(
            self, 
            text="Select Transcription Model", 
            font=("Helvetica", 16, "bold")
        )
        self.title_label.pack(pady=(15, 10))
        
        # Description
        self.desc_label = ctk.CTkLabel(
            self, 
            text="Choose a model based on your needs. Larger models offer better accuracy but require more resources.",
            wraplength=400
        )
        self.desc_label.pack(pady=(0, 15), padx=20)
        
        # Create a scrollable frame for models
        self.models_container = ctk.CTkScrollableFrame(self, width=400, height=300)
        self.models_container.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Add models to the container
        self.model_frames = {}
        
        # Find recommended model
        recommended_model_id = next(
            (model_id for model_id, info in self.models_dict.items() if info.get("recommended", False)), 
            None
        )
        
        for model_id, info in self.models_dict.items():
            # Create a frame for each model
            model_frame = ctk.CTkFrame(self.models_container, corner_radius=10)
            model_frame.pack(fill="x", padx=5, pady=5, ipady=5)
            
            # Recommended badge if applicable
            if info.get("recommended", False):
                badge_frame = ctk.CTkFrame(model_frame, fg_color="#388e3c", corner_radius=5)
                badge_frame.place(x=10, y=5)
                badge_label = ctk.CTkLabel(
                    badge_frame, 
                    text=" RECOMMENDED ", 
                    text_color="white", 
                    font=("Helvetica", 10, "bold"),
                    padx=5, pady=0
                )
                badge_label.pack()
            
            # Model name
            name_label = ctk.CTkLabel(
                model_frame,
                text=info["name"],
                font=("Helvetica", 14, "bold"),
                anchor="w"
            )
            name_label.pack(fill="x", padx=(20 if not info.get("recommended") else 120, 20), pady=(10, 0))
            
            # Model details
            details = f"{info['description']} • {info['size_mb']} MB"
            details_label = ctk.CTkLabel(
                model_frame,
                text=details,
                anchor="w",
                text_color="#a0a0a0"
            )
            details_label.pack(fill="x", padx=20, pady=(0, 5))
            
            # Select button
            select_btn = ctk.CTkButton(
                model_frame,
                text="Select",
                width=80,
                command=lambda mid=model_id: self.select_model(mid),
                fg_color="#2b5797" if model_id == recommended_model_id else "transparent"
            )
            select_btn.pack(anchor="e", padx=20, pady=(5, 10))
            
            self.model_frames[model_id] = {
                "frame": model_frame,
                "button": select_btn
            }
            
        # If there's a recommended model and we have a parent window, select it by default
        # Do this after both the frame and parent window are fully set up
        if recommended_model_id and hasattr(self.master, 'winfo_exists'):
            self.master.after(100, lambda: self.select_model(recommended_model_id))
    
    def select_model(self, model_id):
        """Handle model selection"""
        # Reset all buttons to default style
        for mid, components in self.model_frames.items():
            if mid == model_id:
                components["button"].configure(fg_color="#1f6aa5", text="Selected ✓")
                components["frame"].configure(fg_color="#233d55")
            else:
                is_recommended = self.models_dict[mid].get("recommended", False)
                components["button"].configure(
                    fg_color="#2b5797" if is_recommended else "transparent",
                    text="Select"
                )
                components["frame"].configure(fg_color="transparent")
        
        self.selected_model_id = model_id
        if self.on_model_selected:
            self.on_model_selected(model_id)
    
    def get_selected_model(self):
        return self.selected_model_id

class SpeechTranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Speech Transcription Tool")
        self.root.geometry("900x700")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Apply modern theme
        ctk.set_appearance_mode("dark")  # Options: "dark", "light", "system"
        ctk.set_default_color_theme("blue")
        
        self.recording = False
        self.audio_analyzer = AudioAnalyzer()
        self.recording_duration = 5
        self.transcription_active = False
        self.transcription_thread = None
        self.asr_pipe = None
        self.current_model_id = None
        self.current_model_loading = False
        self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
        
        # Create main container
        self.main_container = ctk.CTkTabview(self.root)
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create tabs
        self.main_container.add("Transcription")
        self.main_container.add("Model Selection")
        self.main_container.add("Settings")
        
        # Set up the tabs
        self.setup_transcription_tab()
        self.setup_model_selection_tab()
        self.setup_settings_tab()
        
        # Show model selection first if no model is loaded
        if not self.asr_pipe:
            self.main_container.set("Model Selection")
    
    def setup_transcription_tab(self):
        tab = self.main_container.tab("Transcription")
        
        # Create frames
        self.top_frame = ctk.CTkFrame(tab)
        self.top_frame.pack(fill="x", padx=10, pady=10)
        
        self.middle_frame = ctk.CTkFrame(tab)
        self.middle_frame.pack(fill="x", padx=10, pady=5)
        
        self.bottom_frame = ctk.CTkFrame(tab)
        self.bottom_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Top Frame - Controls
        self.record_button = ctk.CTkButton(
            self.top_frame, 
            text="Start Recording", 
            command=self.toggle_recording,
            width=150,
            height=40,
            corner_radius=8,
            font=("Helvetica", 14, "bold"),
            fg_color="#1f6aa5",
            hover_color="#154e7a"
        )
        self.record_button.pack(side="left", padx=10, pady=10)
        
        self.duration_label = ctk.CTkLabel(
            self.top_frame, 
            text="Recording Duration (s):",
            font=("Helvetica", 12)
        )
        self.duration_label.pack(side="left", padx=10, pady=10)
        
        self.duration_slider = ctk.CTkSlider(
            self.top_frame, 
            from_=1, 
            to=15, 
            number_of_steps=14,
            command=self.update_duration,
            width=200,
            progress_color="#1f6aa5"
        )
        self.duration_slider.set(5)
        self.duration_slider.pack(side="left", padx=10, pady=10, fill="x", expand=True)
        
        self.duration_value_label = ctk.CTkLabel(
            self.top_frame, 
            text="5s",
            font=("Helvetica", 12, "bold"),
            width=30,
            anchor="w"
        )
        self.duration_value_label.pack(side="left", padx=(0, 10), pady=10)
        
        self.clear_button = ctk.CTkButton(
            self.top_frame, 
            text="Clear Transcript", 
            command=self.clear_transcript,
            width=120,
            height=40,
            corner_radius=8,
            font=("Helvetica", 14),
            fg_color="#555555",
            hover_color="#444444"
        )
        self.clear_button.pack(side="left", padx=10, pady=10)
        
        # Middle Frame - Microphone Visualizer
        self.setup_visualizer()
        
        # Status label with fancy styling
        self.status_frame = ctk.CTkFrame(
            self.middle_frame, 
            fg_color="#1a1a1a",
            corner_radius=8,
            height=30
        )
        self.status_frame.pack(fill="x", padx=10, pady=5)
        
        self.status_label = ctk.CTkLabel(
            self.status_frame, 
            text="Ready",
            font=("Helvetica", 12),
            text_color="#8ab4f8"
        )
        self.status_label.pack(pady=5)
        
        # Model indicator
        self.model_indicator = ctk.CTkLabel(
            self.middle_frame,
            text="No model loaded",
            font=("Helvetica", 10),
            text_color="#a0a0a0"
        )
        self.model_indicator.pack(pady=5)
        
        # Bottom Frame - Transcription Display with improved styling
        self.transcript_header_frame = ctk.CTkFrame(self.bottom_frame, fg_color="transparent")
        self.transcript_header_frame.pack(fill="x", padx=5, pady=(5, 0))
        
        self.transcript_label = ctk.CTkLabel(
            self.transcript_header_frame, 
            text="Transcription:",
            font=("Helvetica", 14, "bold")
        )
        self.transcript_label.pack(side="left", padx=5)
        
        self.export_button = ctk.CTkButton(
            self.transcript_header_frame,
            text="Export",
            command=self.export_transcript,
            width=80,
            height=25,
            corner_radius=5
        )
        self.export_button.pack(side="right", padx=5)
        
        # Custom styled transcript area
        self.transcript_area = scrolledtext.ScrolledText(
            self.bottom_frame,
            wrap=tk.WORD,
            font=("Consolas", 12),
            bg="#1e1e1e",
            fg="#ffffff",
            insertbackground="#ffffff",
            selectbackground="#264f78",
            relief="flat",
            borderwidth=0,
            padx=10,
            pady=10
        )
        self.transcript_area.pack(fill="both", expand=True, padx=5, pady=5)
    
    def setup_visualizer(self):
        """Set up the audio visualizer with fixed animation frames"""
        self.fig, self.ax = plt.subplots(figsize=(8, 1.5), facecolor='#2b2b2b')
        self.ax.set_facecolor('#2b2b2b')
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_title('Microphone Intensity', color='white', fontsize=10)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Create initial bar
        self.intensity_bar = self.ax.barh(0.5, 0, height=0.7, color='#1f6aa5')
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.middle_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Set up animation with explicit save_count to avoid warning
        self.animation = FuncAnimation(
            self.fig, 
            self.update_intensity_bar, 
            interval=100, 
            blit=False,
            cache_frame_data=False,
            save_count=100  # Explicit save_count to avoid warning
        )
    
    def setup_model_selection_tab(self):
        tab = self.main_container.tab("Model Selection")
        
        # Create model selection frame
        self.model_selection_frame = ModelSelectionFrame(
            tab, 
            AVAILABLE_MODELS, 
            self.on_model_selected
        )
        self.model_selection_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Bottom action frame
        self.model_action_frame = ctk.CTkFrame(tab)
        self.model_action_frame.pack(fill="x", padx=10, pady=10)
        
        # Load selected model button
        self.load_model_button = ctk.CTkButton(
            self.model_action_frame,
            text="Load Selected Model",
            command=self.load_selected_model,
            font=("Helvetica", 14, "bold"),
            height=40,
            fg_color="#1f6aa5",
            hover_color="#154e7a"
        )
        self.load_model_button.pack(pady=10)
    
    def setup_settings_tab(self):
        tab = self.main_container.tab("Settings")
        
        # Create settings frame
        settings_frame = ctk.CTkFrame(tab)
        settings_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Title
        settings_title = ctk.CTkLabel(
            settings_frame,
            text="Application Settings",
            font=("Helvetica", 18, "bold")
        )
        settings_title.pack(pady=(15, 20))
        
        # Cache settings
        cache_frame = ctk.CTkFrame(settings_frame)
        cache_frame.pack(fill="x", padx=20, pady=10)
        
        cache_title = ctk.CTkLabel(
            cache_frame,
            text="Model Cache",
            font=("Helvetica", 14, "bold"),
            anchor="w"
        )
        cache_title.pack(fill="x", padx=10, pady=(10, 5))
        
        cache_path = ctk.CTkLabel(
            cache_frame,
            text=f"Cache Directory: {self.cache_dir}",
            font=("Helvetica", 12),
            anchor="w"
        )
        cache_path.pack(fill="x", padx=10, pady=5)
        
        cache_size_frame = ctk.CTkFrame(cache_frame, fg_color="transparent")
        cache_size_frame.pack(fill="x", padx=10, pady=5)
        
        cache_size_label = ctk.CTkLabel(
            cache_size_frame,
            text="Current Cache Size:",
            font=("Helvetica", 12),
            anchor="w"
        )
        cache_size_label.pack(side="left")
        
        self.cache_size_value = ctk.CTkLabel(
            cache_size_frame,
            text="Calculating...",
            font=("Helvetica", 12, "bold"),
            text_color="#8ab4f8"
        )
        self.cache_size_value.pack(side="left", padx=(5, 0))
        
        # Calculate cache size in background
        threading.Thread(target=self.update_cache_size, daemon=True).start()
        
        cache_buttons_frame = ctk.CTkFrame(cache_frame, fg_color="transparent")
        cache_buttons_frame.pack(fill="x", padx=10, pady=(5, 10))
        
        clear_cache_button = ctk.CTkButton(
            cache_buttons_frame,
            text="Clear Cache",
            command=self.clear_cache,
            fg_color="#d32f2f",
            hover_color="#b71c1c",
            width=120
        )
        clear_cache_button.pack(side="left", padx=(0, 10))
        
        change_cache_dir_button = ctk.CTkButton(
            cache_buttons_frame,
            text="Change Cache Directory",
            command=self.change_cache_dir,
            fg_color="#555555",
            hover_color="#444444",
            width=180
        )
        change_cache_dir_button.pack(side="left")
        
        # Audio settings
        audio_frame = ctk.CTkFrame(settings_frame)
        audio_frame.pack(fill="x", padx=20, pady=10)
        
        audio_title = ctk.CTkLabel(
            audio_frame,
            text="Audio Settings",
            font=("Helvetica", 14, "bold"),
            anchor="w"
        )
        audio_title.pack(fill="x", padx=10, pady=(10, 5))
        
        # Sensitivity slider
        sensitivity_frame = ctk.CTkFrame(audio_frame, fg_color="transparent")
        sensitivity_frame.pack(fill="x", padx=10, pady=5)
        
        sensitivity_label = ctk.CTkLabel(
            sensitivity_frame,
            text="Microphone Sensitivity:",
            font=("Helvetica", 12),
            anchor="w"
        )
        sensitivity_label.pack(side="left")
        
        self.sensitivity_value = ctk.CTkLabel(
            sensitivity_frame,
            text="100%",
            font=("Helvetica", 12, "bold"),
            width=50,
            anchor="w"
        )
        self.sensitivity_value.pack(side="right")
        
        self.sensitivity_slider = ctk.CTkSlider(
            audio_frame,
            from_=50,
            to=200,
            number_of_steps=15,
            command=self.update_sensitivity
        )
        self.sensitivity_slider.set(100)
        self.sensitivity_slider.pack(fill="x", padx=10, pady=(0, 10))
        
        # About section
        about_frame = ctk.CTkFrame(settings_frame)
        about_frame.pack(fill="x", padx=20, pady=(10, 20))
        
        about_title = ctk.CTkLabel(
            about_frame,
            text="About",
            font=("Helvetica", 14, "bold"),
            anchor="w"
        )
        about_title.pack(fill="x", padx=10, pady=(10, 5))
        
        about_text = ctk.CTkLabel(
            about_frame,
            text="Enhanced Speech Transcription Tool v1.0\n"
                 "This application uses AI models to transcribe speech in real-time.",
            font=("Helvetica", 12),
            anchor="w",
            justify="left",
            wraplength=500
        )
        about_text.pack(fill="x", padx=10, pady=(0, 10))
    
    def update_cache_size(self):
        """Calculate and update the cache size display"""
        if os.path.exists(self.cache_dir):
            total_size = 0
            for dirpath, _, filenames in os.walk(self.cache_dir):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if not os.path.islink(fp):
                        total_size += os.path.getsize(fp)
            
            # Convert to appropriate unit
            if total_size < 1024**2:
                size_str = f"{total_size / 1024:.1f} KB"
            elif total_size < 1024**3:
                size_str = f"{total_size / (1024**2):.1f} MB"
            else:
                size_str = f"{total_size / (1024**3):.2f} GB"
            
            # Update in main thread
            self.root.after(0, lambda: self.cache_size_value.configure(text=size_str))
        else:
            self.root.after(0, lambda: self.cache_size_value.configure(text="0 KB"))
    
    def clear_cache(self):
        """Clear the model cache directory"""
        if not os.path.exists(self.cache_dir):
            messagebox.showinfo("Info", "Cache directory does not exist.")
            return
        
        result = messagebox.askyesno(
            "Confirm Clear Cache", 
            "Are you sure you want to clear the model cache? "
            "You'll need to download models again."
        )
        
        if result:
            try:
                # Delete all files in cache directory
                for item in os.listdir(self.cache_dir):
                    item_path = os.path.join(self.cache_dir, item)
                    if os.path.isfile(item_path):
                        os.unlink(item_path)
                    else:
                        for subitem in os.listdir(item_path):
                            subitem_path = os.path.join(item_path, subitem)
                            if os.path.isfile(subitem_path):
                                os.unlink(subitem_path)
                        os.rmdir(item_path)
                
                messagebox.showinfo("Success", "Cache cleared successfully.")
                # Update cache size display
                threading.Thread(target=self.update_cache_size, daemon=True).start()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to clear cache: {str(e)}")
    
    def change_cache_dir(self):
        """Change the model cache directory"""
        new_dir = filedialog.askdirectory(
            title="Select New Cache Directory", 
            initialdir=os.path.dirname(self.cache_dir)
        )
        
        if new_dir:
            self.cache_dir = new_dir
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
            
            # Update cache path display
            for widget in self.main_container.tab("Settings").winfo_children():
                for subwidget in widget.winfo_children():
                    if isinstance(subwidget, ctk.CTkLabel) and "Cache Directory:" in subwidget._text:
                        subwidget.configure(text=f"Cache Directory: {self.cache_dir}")
                        break
            
            # Update cache size display
            threading.Thread(target=self.update_cache_size, daemon=True).start()
            
            # Notify user
            messagebox.showinfo(
                "Cache Directory Changed", 
                f"Cache directory changed to:\n{self.cache_dir}\n\n"
                "You'll need to reload any models you want to use."
            )
    
    def update_sensitivity(self, value):
        """Update microphone sensitivity"""
        sensitivity = int(value)
        self.sensitivity_value.configure(text=f"{sensitivity}%")
        
        # Apply sensitivity to audio analyzer
        if hasattr(self, 'audio_analyzer'):
            # This will scale the visualization sensitivity
            self.audio_analyzer.buffer = np.zeros(100)  # Reset buffer
    
    def on_model_selected(self, model_id):
        """Callback when a model is selected in the model selection tab"""
        self.selected_model_id = model_id
        model_info = AVAILABLE_MODELS.get(model_id, {})
        model_name = model_info.get("name", model_id.split('/')[-1])
        
        # Enable the load button
        self.load_model_button.configure(
            text=f"Load {model_name}",
            state="normal"
        )
    
    def load_selected_model(self):
        """Load the selected model"""
        if not hasattr(self, 'selected_model_id') or not self.selected_model_id:
            messagebox.showinfo("Info", "Please select a model first.")
            return
        
        model_id = self.selected_model_id
        model_info = AVAILABLE_MODELS.get(model_id, {})
        model_name = model_info.get("name", model_id.split('/')[-1])
        
        # Check if it's already the current model
        if self.current_model_id == model_id and self.asr_pipe is not None:
            messagebox.showinfo("Info", f"{model_name} is already loaded.")
            # Switch to transcription tab
            self.main_container.set("Transcription")
            return
        
        # Disable the load button during loading
        self.load_model_button.configure(
            text=f"Loading {model_name}...",
            state="disabled"
        )
        
        # Update status
        self.status_label.configure(text=f"Loading {model_name}... This may take a while")
        self.current_model_loading = True
        
        def load_model_thread():
            try:
                # Create cache directory if it doesn't exist
                if not os.path.exists(self.cache_dir):
                    os.makedirs(self.cache_dir)
                
                # Load the model
                self.asr_pipe = load_asr_pipeline(model_id, self.cache_dir)
                self.current_model_id = model_id
                
                # Update UI in main thread
                self.root.after(0, lambda: self.on_model_loaded(model_id))
            except Exception as e:
                # Handle error in main thread
                self.root.after(0, lambda: self.on_model_load_error(str(e)))
        
        # Start loading in background
        threading.Thread(target=load_model_thread, daemon=True).start()
    
    def on_model_loaded(self, model_id):
        """Called when model is successfully loaded"""
        model_info = AVAILABLE_MODELS.get(model_id, {})
        model_name = model_info.get("name", model_id.split('/')[-1])
        
        # Update UI
        self.load_model_button.configure(
            text=f"Load {model_name}",
            state="normal"
        )
        self.status_label.configure(text="Ready")
        self.model_indicator.configure(text=f"Using model: {model_name}")
        self.current_model_loading = False
        
        # Switch to transcription tab
        self.main_container.set("Transcription")
        
        # Show success message
        messagebox.showinfo("Success", f"{model_name} loaded successfully!")
    
    def on_model_load_error(self, error_msg):
        """Called when model loading fails"""
        self.load_model_button.configure(
            text="Load Selected Model",
            state="normal"
        )
        self.status_label.configure(text="Error loading model")
        self.current_model_loading = False
        
        # Show error message
        messagebox.showerror("Error", f"Failed to load model: {error_msg}")
    
    def update_intensity_bar(self, frame):
        """Update the microphone intensity visualization"""
        if hasattr(self, 'audio_analyzer'):
            intensity = self.audio_analyzer.get_intensity()
            
            # Apply sensitivity setting
            sensitivity = self.sensitivity_slider.get() / 100 if hasattr(self, 'sensitivity_slider') else 1.0
            adjusted_intensity = min(1.0, intensity * sensitivity)
            
            # Update the bar width
            self.intensity_bar[0].set_width(adjusted_intensity)
            
            # Change color based on intensity
            if adjusted_intensity < 0.3:
                color = '#3498db'  # Blue
            elif adjusted_intensity < 0.6:
                color = '#f39c12'  # Orange
            else:
                color = '#e74c3c'  # Red
            
            self.intensity_bar[0].set_color(color)
            
            # Add a vertical line at the end of the bar for better visibility
            if hasattr(self, 'intensity_line') and self.intensity_line:
                self.intensity_line.remove()
            
            if adjusted_intensity > 0.05:  # Only show line for non-zero intensities
                self.intensity_line = self.ax.axvline(
                    x=adjusted_intensity, 
                    color='white', 
                    linestyle='-', 
                    alpha=0.5,
                    linewidth=1
                )
            else:
                self.intensity_line = None
        
        return self.intensity_bar
    
    def update_duration(self, value):
        """Update recording duration setting"""
        self.recording_duration = int(value)
        self.duration_value_label.configure(text=f"{self.recording_duration}s")
    
    def toggle_recording(self):
        """Toggle between recording and not recording"""
        if not self.asr_pipe:
            messagebox.showinfo("Info", "Please load a model first.")
            self.main_container.set("Model Selection")
            return
            
        if not self.transcription_active:
            self.start_transcription()
        else:
            self.stop_transcription()
    
    def start_transcription(self):
        """Start the transcription process"""
        self.transcription_active = True
        self.record_button.configure(
            text="Stop Recording", 
            fg_color="#e74c3c",
            hover_color="#c0392b"
        )
        self.status_label.configure(text="Transcribing...")
        self.audio_analyzer.start()
        
        def transcription_thread():
            while self.transcription_active:
                # Record audio
                self.root.after(0, lambda: self.status_label.configure(text="Recording..."))
                
                # Use the AudioAnalyzer to record audio
                audio = self.audio_analyzer.get_audio_chunk(self.recording_duration)
                
                if not self.transcription_active:
                    break
                
                # Process audio
                self.root.after(0, lambda: self.status_label.configure(text="Processing..."))
                audio_input = {"array": audio, "sampling_rate": 16000}
                
                try:
                    result = self.asr_pipe(audio_input)
                    transcription = result["text"].strip()
                    
                    # Only update if there's meaningful content
                    if transcription and not transcription.lower() in ["", "thank you.", "thank you", "thanks"]:
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
        """Stop the transcription process"""
        self.transcription_active = False
        self.record_button.configure(
            text="Start Recording", 
            fg_color="#1f6aa5",
            hover_color="#154e7a"
        )
        self.status_label.configure(text="Stopped")
        self.audio_analyzer.stop()
    
    def update_transcript(self, text):
        """Update the transcript display with new text"""
        if text.strip():  # Only add non-empty transcriptions
            current_time = time.strftime("%H:%M:%S")
            
            # Format the timestamp with a different color using tags
            self.transcript_area.tag_configure("timestamp", foreground="#8ab4f8")
            
            # Insert timestamp with tag
            self.transcript_area.insert(tk.END, f"[{current_time}] ", "timestamp")
            
            # Insert the transcription
            self.transcript_area.insert(tk.END, f"{text}\n\n")
            
            # Scroll to the bottom
            self.transcript_area.see(tk.END)
    
    def clear_transcript(self):
        """Clear the transcript display"""
        self.transcript_area.delete(1.0, tk.END)
    
    def export_transcript(self):
        """Export the transcript to a text file"""
        if not self.transcript_area.get(1.0, tk.END).strip():
            messagebox.showinfo("Info", "No transcript to export.")
            return
        
        # Get file path from user
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Export Transcript"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.transcript_area.get(1.0, tk.END))
                messagebox.showinfo("Success", f"Transcript exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export transcript: {str(e)}")
    
    def on_close(self):
        """Handle application closing"""
        if self.transcription_active:
            self.stop_transcription()
        if hasattr(self, 'animation') and self.animation:
            self.animation.event_source.stop()
        self.root.destroy()

def main():
    root = ctk.CTk()
    app = SpeechTranscriptionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()