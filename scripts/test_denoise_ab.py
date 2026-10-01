"""
A/B Test and Benchmark for Server-Side CPU Noise Suppression.
Measures:
1. Processing latency per 80ms frame (Target: < 20 ms).
2. Noise floor attenuation (dB) on speech + acoustic noise.
3. Speech band preservation (ensures speech formant retention without clipping).
"""

import time
import numpy as np

from orchestration.audio.cleaner import CallerAudioCleaner
from orchestration.protocol.audio import FRAME_SIZE, SAMPLE_RATE, compute_rms


def run_noise_ab_benchmark():
    # 1. Synthesize a clean 1.5s speech signal (harmonics at 150 Hz, 300 Hz, 800 Hz, 2400 Hz)
    duration_sec = 2.0
    t = np.linspace(0, duration_sec, int(SAMPLE_RATE * duration_sec), endpoint=False, dtype=np.float32)
    clean_speech = (
        0.30 * np.sin(2 * np.pi * 180 * t) +
        0.20 * np.sin(2 * np.pi * 360 * t) +
        0.15 * np.sin(2 * np.pi * 1200 * t) +
        0.10 * np.sin(2 * np.pi * 2400 * t)
    )

    # Add intermittent pauses (silence between words)
    speech_mask = (np.sin(2 * np.pi * 1.5 * t) > 0.1).astype(np.float32)
    clean_speech *= speech_mask

    # 2. Add realistic room acoustic noise (50Hz mains hum + pink/white background hiss)
    np.random.seed(42)
    hum_50hz = 0.05 * np.sin(2 * np.pi * 50 * t)
    hiss = 0.03 * np.random.normal(0, 1, len(t)).astype(np.float32)
    noisy_audio = clean_speech + hum_50hz + hiss

    # 3. Test A: Cleaner with BYPASS = True (Raw stream)
    cleaner_bypass = CallerAudioCleaner(sample_rate=SAMPLE_RATE, frame_size=FRAME_SIZE, bypass=True)
    t0 = time.perf_counter()
    bypass_frames = []
    for i in range(0, len(noisy_audio), FRAME_SIZE):
        chunk = noisy_audio[i:i + FRAME_SIZE]
        if len(chunk) == FRAME_SIZE:
            bypass_frames.extend(cleaner_bypass.process_chunk(chunk))
    bypass_latency_ms = (time.perf_counter() - t0) * 1000.0

    # 4. Test B: Cleaner with Denoise ACTIVE (High-pass + Spectral Subtraction)
    cleaner_active = CallerAudioCleaner(sample_rate=SAMPLE_RATE, frame_size=FRAME_SIZE, bypass=False, suppression_strength=0.65)
    t0 = time.perf_counter()
    clean_frames = []
    for i in range(0, len(noisy_audio), FRAME_SIZE):
        chunk = noisy_audio[i:i + FRAME_SIZE]
        if len(chunk) == FRAME_SIZE:
            clean_frames.extend(cleaner_active.process_chunk(chunk))
    active_latency_ms = (time.perf_counter() - t0) * 1000.0

    per_frame_latency_ms = active_latency_ms / max(1, len(clean_frames))

    # 5. Measure Noise Floor in Pauses
    pause_mask = (speech_mask == 0)
    raw_noise_rms = compute_rms(noisy_audio[pause_mask])
    clean_audio = np.concatenate(clean_frames) if clean_frames else np.zeros_like(noisy_audio)
    clean_noise_rms = compute_rms(clean_audio[pause_mask[:len(clean_audio)]])

    noise_reduction_db = 20 * np.log10(max(1e-5, raw_noise_rms) / max(1e-5, clean_noise_rms))

    # 6. Speech Energy Retention in Active Speech
    active_mask = (speech_mask == 1)
    clean_speech_rms = compute_rms(clean_speech[active_mask])
    processed_speech_rms = compute_rms(clean_audio[active_mask[:len(clean_audio)]])
    speech_retention_pct = (processed_speech_rms / max(1e-5, clean_speech_rms)) * 100.0

    print("=" * 60)
    print("SERVER-SIDE CPU AUDIO CLEANER A/B BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total Audio Processed:       {duration_sec:.2f} s ({len(clean_frames)} frames of 80ms)")
    print(f"Bypass Processing Latency:   {bypass_latency_ms:.3f} ms total")
    print(f"Active Denoise Latency:      {active_latency_ms:.3f} ms total")
    print(f"Per-Frame Latency:           {per_frame_latency_ms:.3f} ms / frame (Target: < 20 ms) -> {'PASS' if per_frame_latency_ms < 20 else 'FAIL'}")
    print(f"Raw Noise Floor RMS:         {raw_noise_rms:.4f}")
    print(f"Cleaned Noise Floor RMS:     {clean_noise_rms:.4f}")
    print(f"Noise Attenuation:           {noise_reduction_db:.2f} dB reduction")
    print(f"Speech Energy Retention:     {speech_retention_pct:.1f}% preserved")
    print("=" * 60)

    assert per_frame_latency_ms < 20.0, f"Per-frame latency {per_frame_latency_ms} ms exceeds 20 ms target"
    assert noise_reduction_db > 6.0, f"Noise reduction {noise_reduction_db} dB is insufficient"
    return {
        "per_frame_latency_ms": round(per_frame_latency_ms, 2),
        "noise_reduction_db": round(noise_reduction_db, 2),
        "speech_retention_pct": round(speech_retention_pct, 1),
    }


if __name__ == "__main__":
    run_noise_ab_benchmark()
