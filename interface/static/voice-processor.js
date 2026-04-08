/**
 * Sovereign Voice AudioWorklet Processor
 * Captures microphone audio at 16kHz mono and sends PCM chunks to main thread.
 * Runs in a separate audio thread for zero-latency capture.
 */
class VoiceCaptureProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this._buffer = [];
        this._bufferSize = 0;
        // Send chunks every ~100ms (1600 samples at 16kHz)
        this._chunkSize = 1600;
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (!input || !input[0]) return true;

        const channelData = input[0]; // mono
        let sumSquares = 0;
        let peak = 0;
        let zeroCrossings = 0;
        let clippingSamples = 0;
        
        // Convert float32 [-1, 1] to int16 PCM
        for (let i = 0; i < channelData.length; i++) {
            const s = Math.max(-1, Math.min(1, channelData[i]));
            const abs = Math.abs(s);
            sumSquares += s * s;
            if (abs > peak) peak = abs;
            if (abs >= 0.985) clippingSamples += 1;
            if (i > 0 && ((channelData[i - 1] >= 0 && s < 0) || (channelData[i - 1] < 0 && s >= 0))) {
                zeroCrossings += 1;
            }
            const int16 = s < 0 ? s * 0x8000 : s * 0x7FFF;
            this._buffer.push(int16);
        }
        this._bufferSize += channelData.length;

        const rms = Math.sqrt(sumSquares / Math.max(1, channelData.length));
        this.port.postMessage({
            type: 'features',
            rms,
            peak,
            zcr: zeroCrossings / Math.max(1, channelData.length - 1),
            clippingRatio: clippingSamples / Math.max(1, channelData.length),
            sampleCount: channelData.length,
        });

        // When we have enough samples, send to main thread
        if (this._bufferSize >= this._chunkSize) {
            const pcmArray = new Int16Array(this._buffer);
            this.port.postMessage({
                type: 'pcm',
                data: pcmArray.buffer
            }, [pcmArray.buffer]);
            
            this._buffer = [];
            this._bufferSize = 0;
        }

        return true; // Keep processor alive
    }
}

registerProcessor('voice-capture-processor', VoiceCaptureProcessor);
