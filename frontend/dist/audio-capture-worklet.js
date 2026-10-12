/**
 * AudioWorkletProcessor for high-performance, click-free microphone capture.
 * Captures mono audio at native browser rate, downsamples to 16,000 Hz,
 * computes live input RMS, and outputs fixed 320-sample (20ms) PCM16 frames.
 */
class AudioCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super()
    this.targetSampleRate = 16000
    this.buffer = []
    this.frameSize = 320 // 20ms at 16kHz
    this.rmsCounter = 0
    this.rmsAccum = 0

    this.port.onmessage = (event) => {
      if (event.data && event.data.command === 'reset') {
        this.buffer = []
      }
    }
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0]
    if (!input || input.length === 0) return true

    const channelData = input[0]
    if (!channelData || channelData.length === 0) return true

    // Compute live input RMS for mic volume meter
    let sumSq = 0
    for (let i = 0; i < channelData.length; i++) {
      const s = channelData[i]
      sumSq += s * s
    }
    this.rmsAccum += sumSq
    this.rmsCounter += channelData.length

    if (this.rmsCounter >= 800) { // ~50ms updates
      const rms = Math.sqrt(this.rmsAccum / this.rmsCounter)
      this.port.postMessage({ type: 'mic_rms', rms })
      this.rmsAccum = 0
      this.rmsCounter = 0
    }

    // Resample to 16,000 Hz if native rate differs
    const nativeRate = sampleRate // Global in AudioWorkletGlobalScope
    if (nativeRate === this.targetSampleRate) {
      for (let i = 0; i < channelData.length; i++) {
        this.buffer.push(channelData[i])
      }
    } else {
      const ratio = nativeRate / this.targetSampleRate
      const outputLength = Math.floor(channelData.length / ratio)
      for (let i = 0; i < outputLength; i++) {
        const srcIdx = i * ratio
        const idx0 = Math.floor(srcIdx)
        const idx1 = Math.min(idx0 + 1, channelData.length - 1)
        const frac = srcIdx - idx0
        const sample = channelData[idx0] * (1 - frac) + channelData[idx1] * frac
        this.buffer.push(sample)
      }
    }

    // Drain fixed-size 320-sample frames (640 bytes PCM16)
    while (this.buffer.length >= this.frameSize) {
      const chunk = this.buffer.splice(0, this.frameSize)
      const pcm16 = new Int16Array(this.frameSize)
      for (let i = 0; i < this.frameSize; i++) {
        const s = Math.max(-1.0, Math.min(1.0, chunk[i]))
        pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff
      }
      this.port.postMessage(
        { type: 'pcm16', data: pcm16.buffer },
        [pcm16.buffer]
      )
    }

    return true
  }
}

registerProcessor('audio-capture-processor', AudioCaptureProcessor)
