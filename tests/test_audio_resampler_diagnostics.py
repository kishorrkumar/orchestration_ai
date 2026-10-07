"""
Diagnostic unit test verifying bidirectional audio resampling between
16 kHz PCM16 (client browser) and 24 kHz Float32 (PersonaPlex Mimi 80ms / 1920-sample frames).
"""

import math
import numpy as np
import pytest

from orchestration.audio.resample import AudioResampler, StreamingResampleBuffer
from orchestration.protocol.audio import (
    CLIENT_SAMPLE_RATE,
    MODEL_FRAME_SIZE,
    MODEL_SAMPLE_RATE,
    compute_rms,
    float32_to_int16,
    int16_to_float32,
)


def test_sine_440hz_bidirectional_roundtrip():
    """
    Round-trips a 440 Hz pure tone:
    16 kHz PCM16 -> 24 kHz Float32 (1920 frames) -> 16 kHz PCM16.
    Verifies frequency peak, amplitude preservation, and frame consistency.
    """
    duration_sec = 1.0
    freq_hz = 440.0
    amp = 0.5  # -6 dBFS

    t_16k = np.linspace(0, duration_sec, int(CLIENT_SAMPLE_RATE * duration_sec), endpoint=False, dtype=np.float32)
    original_f32 = amp * np.sin(2 * np.pi * freq_hz * t_16k)
    original_pcm16 = float32_to_int16(original_f32)
    original_rms = compute_rms(original_f32)

    # 1. Resample 16 kHz PCM16 -> 24 kHz Float32 with 1920-sample frame buffering
    in_buffer = StreamingResampleBuffer(
        in_rate=CLIENT_SAMPLE_RATE,
        out_rate=MODEL_SAMPLE_RATE,
        out_frame_samples=MODEL_FRAME_SIZE,  # 1920 samples = 80ms at 24kHz
        quality="QQ",
    )

    # Stream in chunks of 320 samples (20ms at 16kHz) as incoming WebSocket chunks
    chunk_size = 320
    forward_frames_24k = []
    for i in range(0, len(original_pcm16), chunk_size):
        chunk_pcm16 = original_pcm16[i : i + chunk_size]
        chunk_f32 = int16_to_float32(chunk_pcm16)
        frames = in_buffer.push_chunk(chunk_f32)
        forward_frames_24k.extend(frames)

    # Flush remaining
    flushed = in_buffer.flush()
    forward_frames_24k.extend(flushed)

    assert len(forward_frames_24k) > 0, "Should generate 24 kHz frames"
    for frame in forward_frames_24k:
        assert len(frame) == MODEL_FRAME_SIZE, f"Every frame to worker must be exactly {MODEL_FRAME_SIZE} samples (80ms)"
        assert frame.dtype == np.float32, "Worker audio must be float32"

    concatenated_24k = np.concatenate(forward_frames_24k)
    assert not np.isnan(concatenated_24k).any(), "Resampled 24k audio contains NaNs"

    # 2. Resample 24 kHz Float32 -> 16 kHz PCM16 for browser playback
    out_resampler = AudioResampler(
        in_rate=MODEL_SAMPLE_RATE,
        out_rate=CLIENT_SAMPLE_RATE,
        quality="QQ",
    )

    return_chunks_16k = []
    for frame_24k in forward_frames_24k:
        resampled_16k = out_resampler.resample_chunk(frame_24k, last=False)
        return_chunks_16k.append(resampled_16k)

    final_chunk = out_resampler.resample_chunk(np.empty(0, dtype=np.float32), last=True)
    if len(final_chunk) > 0:
        return_chunks_16k.append(final_chunk)

    concatenated_return_f32 = np.concatenate(return_chunks_16k)
    return_pcm16 = float32_to_int16(concatenated_return_f32)

    # 3. Frequency & Amplitude Analysis
    # Exclude initial 80ms and final 80ms filter ring/transient window
    trim_start = int(0.08 * CLIENT_SAMPLE_RATE)
    trim_end = int(0.92 * CLIENT_SAMPLE_RATE)
    analyzed_signal = concatenated_return_f32[trim_start:trim_end]

    # FFT peak frequency test
    fft_vals = np.abs(np.fft.rfft(analyzed_signal))
    fft_freqs = np.fft.rfftfreq(len(analyzed_signal), 1.0 / CLIENT_SAMPLE_RATE)
    peak_freq = fft_freqs[np.argmax(fft_vals)]
    assert abs(peak_freq - freq_hz) < 3.0, f"Expected 440 Hz, measured peak at {peak_freq:.2f} Hz"

    # RMS amplitude preservation test (within 10% tolerance)
    return_rms = compute_rms(analyzed_signal)
    rms_ratio = return_rms / original_rms
    assert 0.90 <= rms_ratio <= 1.10, f"RMS ratio {rms_ratio:.3f} outside expected [0.90, 1.10] tolerance"
