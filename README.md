# Live Whisper Transcription

A real-time web-based speech-to-text transcription application using the power of WebAudio API and audio processing.

## Features

- Real-time audio capture from microphone
- Live audio processing using AudioWorklet
- Chunked audio processing (3-second intervals)
- Browser-based interface
- High-quality audio sampling at 16kHz

## Prerequisites

- Modern web browser with AudioWorklet API support (Chrome, Firefox, Edge)
- Local development server (due to AudioWorklet security requirements)

## Project Structure

```
live_whisper/
├── audio-processor.js    # AudioWorklet processor for real-time audio handling
├── index.html           # Main HTML interface
├── script.js            # Main application logic
└── style.css           # Application styling
```

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/insomniachooman/Tetx2Speech.git
   cd Tetx2Speech
   ```

2. Due to AudioWorklet security requirements, you need to serve the files through a local web server. You can use any of these methods:

   Using Python:
   ```bash
   # Python 3
   python -m http.server 8000
   ```

   Using Node.js (with http-server):
   ```bash
   npm install -g http-server
   http-server
   ```

3. Open your browser and navigate to:
   - If using Python: `http://localhost:8000`
   - If using http-server: `http://localhost:8080`

## How It Works

1. The application captures audio input from your microphone using the WebAudio API
2. Audio is processed in real-time using an AudioWorklet
3. The audio is chunked into 3-second segments for efficient processing
4. Each chunk is processed at 16kHz sample rate for optimal speech recognition

## Technical Details

- Sample Rate: 16kHz
- Chunk Duration: 3 seconds
- Audio Format: Float32Array
- Processing Method: AudioWorklet for efficient real-time processing

## Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Web Audio API
- AudioWorklet API