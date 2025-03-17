# Speech Transcription App 🎤

A modern, user-friendly desktop application for real-time speech transcription using Distil Whisper's large v3 model. This application allows you to record audio from your microphone and instantly convert it into text with remarkable accuracy.

![Speech Transcription App Screenshot](app_screenshot.png)

## ✨ Features

- **Real-time Speech-to-Text**: Convert spoken language into text in real-time
- **Audio Visualization**: Visual microphone intensity meter with color indicators
- **Adjustable Recording Duration**: Set recording segments from 1-10 seconds
- **Persistent Transcription History**: Timestamped transcript entries
- **Modern Dark UI**: Clean, intuitive interface built with CustomTkinter
- **GPU Acceleration**: Uses CUDA when available for faster processing

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- A working microphone

### Installation

No manual installation of dependencies required! The application will automatically install all necessary packages on first run.

```bash
# Clone the repository
git clone https://github.com/yourusername/speech-transcription-app.git

# Navigate to the project directory
cd speech-transcription-app

# Run the application
python app.py
```

The first run may take a few minutes as the application downloads the speech recognition model (~500MB).

## 🖥️ Usage

1. Launch the application
2. Adjust the recording duration using the slider (default: 5 seconds)
3. Click "Start Recording" to begin continuous transcription
4. Speak into your microphone
5. View the transcription results in the text area
6. Click "Stop Recording" to end the session
7. Use "Clear Transcript" to reset the transcription history

## 🔧 Technical Details

The application uses:

- Distil Whisper: A lightweight, high-performance speech recognition model
- PyTorch: For machine learning operations
- SoundDevice: For audio recording
- CustomTkinter: For the modern UI components
- Matplotlib: For audio visualization

Models are cached locally to avoid repeated downloads.

## 🔄 Recent Improvements

- Improved package management and dependency handling
- Better error handling for import failures
- Code organization and typing improvements
- Parameter name consistency in callback functions
- Cleaner separation of UI and backend components

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 Acknowledgements

- Distil Whisper for the speech recognition model
- CustomTkinter for the modern UI components
