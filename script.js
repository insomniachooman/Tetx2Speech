// script.js
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1';

// --- Configuration ---
//const MODEL_NAME = 'Xenova/whisper-tiny.en'; // Faster, less accurate
const MODEL_NAME = 'Xenova/whisper-base.en'; // Slower, more accurate
//const MODEL_NAME = 'Xenova/whisper-small.en'; // Even slower, better accuracy
const TARGET_SAMPLE_RATE = 16000; // Whisper expects 16kHz
const CHUNK_LENGTH_S = 5; // Use a longer chunk for the pipeline (can overlap with stride)
const STRIDE_LENGTH_S = 2; // Stride for overlapping chunks (reduces missed words)


// --- DOM Elements ---
const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const statusDiv = document.getElementById('status');
const transcriptDiv = document.getElementById('transcript');
const loadingInfo = document.getElementById('loading-info');

// --- State Variables ---
let transcriber = null;
let audioContext = null;
let mediaStream = null;
let audioWorkletNode = null;
let isTranscribing = false;
let isModelLoading = false;
let accumulatedTranscript = ""; // Store the full transcript

// --- Initialization ---

async function initialize() {
    statusDiv.textContent = 'Checking WebGPU support...';
    if (!navigator.gpu) {
        statusDiv.textContent = 'Error: WebGPU is not supported on this browser.';
        console.error('WebGPU not supported.');
        loadingInfo.style.display = 'none';
        startButton.disabled = true;
        return;
    }
    statusDiv.textContent = 'WebGPU supported. Ready to load model.';
    startButton.disabled = false; // Enable start button only after check
}

// --- Transcription Pipeline Setup ---

async function loadTranscriber() {
    if (isModelLoading || transcriber) return; // Prevent multiple loads

    isModelLoading = true;
    startButton.disabled = true;
    stopButton.disabled = true;
    statusDiv.textContent = 'Loading model...';
    loadingInfo.style.display = 'block';
    console.log('Loading model:', MODEL_NAME);

    try {
        // Configure Transformers.js
        env.allowLocalModels = false; // Disable local models for web usage
        env.backends.onnx.wasm.numThreads = 1; // Use single thread for stability in browser

        // Load the pipeline
        transcriber = await pipeline('automatic-speech-recognition', MODEL_NAME, {
            quantized: true, // Use quantized model for potentially better performance
            device: 'webgpu', // Specify WebGPU backend
            dtype: 'float32', // Data type (fp16 might be faster if supported well)
            // progress_callback: (progress) => {
            //     console.log('Model loading progress:', progress);
            //     statusDiv.textContent = `Loading model: ${progress.status} (${Math.round(progress.progress)}%)`;
            // }
        });

        console.log('Transcription pipeline loaded successfully.');
        statusDiv.textContent = 'Model loaded. Ready to start.';
        startButton.disabled = false; // Re-enable start after loading

    } catch (error) {
        console.error('Error loading transcription pipeline:', error);
        statusDiv.textContent = `Error loading model: ${error.message}`;
        transcriber = null; // Ensure transcriber is null on failure
    } finally {
        isModelLoading = false;
        loadingInfo.style.display = 'none';
        // Don't enable stop button here, only when actually listening
    }
}

// --- Audio Processing ---

async function startTranscription() {
    if (isTranscribing || isModelLoading) return;
    if (!transcriber) {
        statusDiv.textContent = 'Model not loaded yet. Please wait or try reloading.';
        // Optionally trigger loading again:
        // await loadTranscriber();
        // if (!transcriber) return; // Exit if loading failed again
        return;
    }

    isTranscribing = true;
    startButton.disabled = true;
    stopButton.disabled = false;
    statusDiv.textContent = 'Initializing audio...';
    accumulatedTranscript = ""; // Clear previous transcript
    transcriptDiv.textContent = ""; // Clear display

    try {
        // 1. Get User Media (Microphone)
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('getUserMedia is not supported in this browser.');
        }
        mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: TARGET_SAMPLE_RATE, // Request desired sample rate
                channelCount: 1                  // Request mono audio
            }
        });

        // 2. Create Audio Context and Source
        // Use the sample rate provided by the media stream, but log if it differs
        const trackSettings = mediaStream.getAudioTracks()[0].getSettings();
        const actualSampleRate = trackSettings.sampleRate;
        console.log(`Requested sample rate: ${TARGET_SAMPLE_RATE}, Actual sample rate: ${actualSampleRate}`);
        if (actualSampleRate !== TARGET_SAMPLE_RATE) {
            console.warn(`Warning: Audio source sample rate (${actualSampleRate}Hz) differs from target (${TARGET_SAMPLE_RATE}Hz). Transcription quality may be affected. The AudioWorklet currently doesn't resample.`);
            // Ideally, resampling should happen in the AudioWorklet for robustness.
        }

        audioContext = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });

        // Resume context if needed (browsers often require user interaction)
        if (audioContext.state === 'suspended') {
            await audioContext.resume();
        }

        const source = audioContext.createMediaStreamSource(mediaStream);

        // 3. Load and Connect Audio Worklet
        try {
            await audioContext.audioWorklet.addModule('audio-processor.js');
        } catch (e) {
            throw new Error(`Failed to load audio worklet module: ${e.message}`);
        }

        audioWorkletNode = new AudioWorkletNode(audioContext, 'audio-processor');
        console.log("AudioWorkletNode created");

        // 4. Setup Message Handling (from Worklet to Main Thread)
        audioWorkletNode.port.onmessage = (event) => {
            // Received an audio chunk from the worklet
            const audioChunk = event.data; // Float32Array
            // console.log(`Received chunk: ${audioChunk.length} samples`);
            processAudioChunk(audioChunk);
        };

        audioWorkletNode.port.onmessageerror = (error) => {
            console.error("Error receiving message from worklet:", error);
        };

        // 5. Connect the Graph: Source -> Worklet
        source.connect(audioWorkletNode);
        // The worklet node doesn't need to connect further if it only sends data back
        // audioWorkletNode.connect(audioContext.destination); // Only if you want to hear the input

        statusDiv.textContent = 'Listening...';
        console.log("Audio graph connected, listening started.");

    } catch (error) {
        console.error('Error starting transcription:', error);
        statusDiv.textContent = `Error: ${error.message}`;
        stopTranscription(); // Clean up resources if setup failed
    }
}

let isProcessing = false; // Flag to prevent overlapping pipeline calls

async function processAudioChunk(audioChunk) {
    if (!transcriber || isProcessing) {
        // console.log("Skipping chunk processing: Transcriber not ready or already processing.");
        return;
    }

    isProcessing = true;
    // statusDiv.textContent = 'Transcribing...'; // Optional: Indicate processing
    // console.time("Transcription"); // Start timing

    try {
        // Perform transcription
        const output = await transcriber(audioChunk, {
            // Parameters for continuous transcription:
            chunk_length_s: CHUNK_LENGTH_S, // Process longer chunks internally
            stride_length_s: STRIDE_LENGTH_S, // Overlap chunks for smoother results
            // task: 'transcribe', // Default is transcribe
            // language: 'en', // Specify language if known
            return_timestamps: false, // Timestamps not needed for simple live display
        });

        // console.timeEnd("Transcription"); // End timing
        // console.log("Transcription output:", output);

        if (output && output.text) {
            const newText = output.text.trim();
            // Basic handling: Append new text. More sophisticated logic could
            // try to merge based on timestamps or confidence if available.
            if (newText && newText !== accumulatedTranscript.slice(-newText.length)) {
                accumulatedTranscript += " " + newText; // Add space between chunks
                transcriptDiv.textContent = accumulatedTranscript.trim();
                // Scroll to bottom
                transcriptDiv.scrollTop = transcriptDiv.scrollHeight;
            }
        }
    } catch (error) {
        console.error('Error during transcription:', error);
        statusDiv.textContent = `Transcription Error: ${error.message}`;
    } finally {
        isProcessing = false;
        // Only change status back to Listening if still transcribing
        if (isTranscribing) {
            statusDiv.textContent = 'Listening...';
        }
    }
}


function stopTranscription() {
    if (!isTranscribing && !audioContext) return; // Nothing to stop

    console.log("Stopping transcription...");
    isTranscribing = false;
    startButton.disabled = isModelLoading; // Disable if loading, otherwise enable
    stopButton.disabled = true;
    statusDiv.textContent = 'Idle';

    // 1. Close AudioContext (this also stops the worklet and source)
    if (audioContext) {
        audioContext.close().catch(e => console.error("Error closing AudioContext:", e));
        audioContext = null;
    }

    // 2. Stop MediaStream Tracks
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }

    // 3. Clean up references
    audioWorkletNode = null; // No need to explicitly disconnect if context is closed

    // Reset processing flag
    isProcessing = false;

    console.log("Transcription stopped.");
}

// --- Event Listeners ---
startButton.addEventListener('click', async () => {
    if (!transcriber && !isModelLoading) {
        await loadTranscriber(); // Load model first if not already loaded/loading
    }
    // Check again if loading was successful before starting
    if (transcriber) {
        startTranscription();
    }
});
stopButton.addEventListener('click', stopTranscription);

// --- Initial Setup Call ---
document.addEventListener('DOMContentLoaded', initialize);

// Optional: Load model immediately on page load (can increase initial load time)
// document.addEventListener('DOMContentLoaded', async () => {
//     await initialize();
//     if (!startButton.disabled) { // Only load if WebGPU is supported
//         loadTranscriber();
//     }
// });