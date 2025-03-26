// audio-processor.js

// Audio processing parameters
const TARGET_SAMPLE_RATE = 16000;
const CHUNK_DURATION_SECONDS = 3; // Process audio in 3-second chunks
const SAMPLES_PER_CHUNK = TARGET_SAMPLE_RATE * CHUNK_DURATION_SECONDS;

class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.audioBuffer = []; // Store audio samples
        this.totalSamples = 0;
        this.port.onmessage = (event) => {
            // Handle messages from the main thread if needed (e.g., configuration)
            // console.log("Worklet received message:", event.data);
        };
        console.log("AudioWorkletProcessor initialized");
    }

    process(inputs, outputs, parameters) {
        // Input format: inputs[inputIndex][channelIndex][sampleIndex]
        // We expect mono input, so inputs[0][0]
        const inputChannel = inputs[0]?.[0];

        if (!inputChannel || inputChannel.length === 0) {
            // No input data, or node disconnected. Keep processor alive.
            return true;
        }

        // Append new samples to our buffer
        // inputChannel is a Float32Array
        this.audioBuffer.push(...inputChannel);
        this.totalSamples += inputChannel.length;

        // Check if we have enough samples for a chunk
        while (this.audioBuffer.length >= SAMPLES_PER_CHUNK) {
            // Extract a chunk
            const chunk = this.audioBuffer.slice(0, SAMPLES_PER_CHUNK);

            // Remove the processed chunk from the beginning of the buffer
            this.audioBuffer.splice(0, SAMPLES_PER_CHUNK);

            // Send the chunk (as Float32Array) to the main thread
            // It's crucial to send a copy or transfer ownership if possible,
            // but Float32Array is usually copied by postMessage.
            this.port.postMessage(new Float32Array(chunk));
        }

        // Return true to keep the processor alive
        return true;
    }
}

registerProcessor('audio-processor', AudioProcessor);