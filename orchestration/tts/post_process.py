"""
Audio Post-Processing for Natural Indian-English Speech Synthesis.

Pipeline:
1. Gentle High-Pass Filter (80 Hz) to eliminate sub-bass rumble and DC bias.
2. Sibilance De-Esser (5.5 kHz - 7.5 kHz) to tame harsh high-frequency fricatives.
3. Target Loudness Normalization (~ -18 LUFS / RMS reference -18 dBFS).
4. Smooth Soft-Limiter (hyperbolic tangent / cubic soft knee, zero hard digital clipping).
5. Crossfade concatenation utility for seamless multi-chunk speech joins without clicks.
"""

from __future__ import annotations
import math
import numpy as np


def apply_high_pass(audio: np.ndarray, cutoff_hz: float = 80.0, fs: int = 24000) -> np.ndarray:
    """1st-order IIR high-pass filter."""
    if len(audio) < 2:
        return audio
    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    dt = 1.0 / fs
    alpha = rc / (rc + dt)
    out = np.empty_like(audio)
    out[0] = audio[0]
    for i in range(1, len(audio)):
        out[i] = alpha * (out[i - 1] + audio[i] - audio[i - 1])
    return out


def apply_de_esser(audio: np.ndarray, fs: int = 24000, threshold: float = 0.05, ratio: float = 0.5) -> np.ndarray:
    """
    Subtle spectral de-esser: attenuates excessive energy in the 5,000–8,000 Hz band
    only when high-frequency energy spikes above threshold.
    """
    if len(audio) < 512:
        return audio

    # Use FFT to isolate sibilant frequencies
    n = len(audio)
    fft_vals = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    # Mask for sibilance range 5500 - 8000 Hz
    sib_mask = (freqs >= 5500) & (freqs <= 8000)
    if np.any(sib_mask):
        sib_energy = np.mean(np.abs(fft_vals[sib_mask]))
        total_energy = np.mean(np.abs(fft_vals)) + 1e-8
        if (sib_energy / total_energy) > threshold:
            # Attenuate sibilant band smoothly
            fft_vals[sib_mask] *= ratio

    cleaned = np.fft.irfft(fft_vals, n=n).astype(np.float32)
    return cleaned


def normalize_loudness(audio: np.ndarray, target_rms: float = 0.125) -> np.ndarray:
    """
    Normalize audio to approximately -18 LUFS (-18 dBFS RMS ~ 0.125 peak-equivalent).
    Preserves dynamics and prevents volume disparity across chunks.
    """
    if len(audio) == 0:
        return audio

    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 1e-4:
        return audio

    gain = target_rms / rms
    # Limit max gain boost to +12 dB to prevent amplifying silence/hiss
    gain = min(gain, 4.0)
    return audio * gain


def apply_soft_limiter(audio: np.ndarray, ceiling: float = 0.95) -> np.ndarray:
    """
    Smooth soft limiter using tanh curve above 0.8 to prevent any digital clipping
    while preserving warmth and natural transient detail.
    """
    if len(audio) == 0:
        return audio

    # For amplitudes under 0.8, linear unity gain.
    # For amplitudes > 0.8, soft compression towards ceiling.
    threshold = 0.8
    out = audio.copy()
    high_pos = out > threshold
    high_neg = out < -threshold

    if np.any(high_pos):
        overshoot = out[high_pos] - threshold
        out[high_pos] = threshold + (ceiling - threshold) * np.tanh(overshoot / (ceiling - threshold + 1e-6))

    if np.any(high_neg):
        overshoot = -out[high_neg] - threshold
        out[high_neg] = -(threshold + (ceiling - threshold) * np.tanh(overshoot / (ceiling - threshold + 1e-6)))

    return np.clip(out, -1.0, 1.0).astype(np.float32)


def post_process_speech(
    audio: np.ndarray,
    fs: int = 24000,
    target_lufs: float = -18.0,
    enable_de_esser: bool = True,
) -> np.ndarray:
    """
    Master speech post-processing chain:
    High-Pass -> De-Esser -> Loudness Normalization -> Soft Limiter.
    Returns 24 kHz float32 audio.
    """
    if len(audio) == 0:
        return audio

    samples = audio.astype(np.float32)

    # 1. High-Pass Filter
    samples = apply_high_pass(samples, cutoff_hz=80.0, fs=fs)

    # 2. De-Esser
    if enable_de_esser:
        samples = apply_de_esser(samples, fs=fs)

    # 3. Loudness Normalization (~ -18 LUFS)
    # Target RMS ~ 10 ** (target_lufs / 20)
    target_rms = 10.0 ** (target_lufs / 20.0)
    samples = normalize_loudness(samples, target_rms=target_rms)

    # 4. Soft Limiter
    samples = apply_soft_limiter(samples, ceiling=0.95)

    return samples


def crossfade_chunks(
    chunk_a: np.ndarray,
    chunk_b: np.ndarray,
    crossfade_samples: int = 360,  # 15 ms at 24 kHz
) -> np.ndarray:
    """
    Perform an equal-power (cosine/sine) crossfade between sequential audio chunks.
    Eliminates boundary pops and spectral discontinuities.
    """
    if len(chunk_a) == 0:
        return chunk_b
    if len(chunk_b) == 0:
        return chunk_a

    n_fade = min(crossfade_samples, len(chunk_a), len(chunk_b))
    if n_fade < 2:
        return np.concatenate((chunk_a, chunk_b))

    # Equal power crossfade curves
    t = np.linspace(0, np.pi / 2, n_fade, endpoint=True)
    fade_out = np.cos(t).astype(np.float32)
    fade_in = np.sin(t).astype(np.float32)

    a_head = chunk_a[:-n_fade]
    a_tail = chunk_a[-n_fade:]
    b_head = chunk_b[:n_fade]
    b_tail = chunk_b[n_fade:]

    blended = (a_tail * fade_out) + (b_head * fade_in)
    return np.concatenate((a_head, blended, b_tail))
