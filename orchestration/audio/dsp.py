"""
Digital Signal Processing (DSP) Utilities for Voice Agent Platform.
Provides soft clipping, RMS computation, dynamic range limiter, and format conversions.
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np


def float32_to_int16(audio: np.ndarray) -> np.ndarray:
    """Convert float32 audio [-1.0, 1.0] to int16 [-32768, 32767]."""
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def int16_to_float32(audio: np.ndarray) -> np.ndarray:
    """Convert int16 audio [-32768, 32767] to float32 [-1.0, 1.0]."""
    return np.clip(audio.astype(np.float32) / 32767.0, -1.0, 1.0)


def compute_rms(audio: Union[np.ndarray, bytes]) -> float:
    """
    Compute Root Mean Square (RMS) energy of audio samples.
    Normalized to float32 scale [0.0, 1.0].
    """
    if isinstance(audio, bytes):
        samples_i16 = np.frombuffer(audio, dtype=np.int16)
        samples = int16_to_float32(samples_i16)
    elif audio.dtype == np.int16:
        samples = int16_to_float32(audio)
    else:
        samples = audio

    if len(samples) == 0:
        return 0.0

    mean_sq = np.mean(samples ** 2)
    return float(math.sqrt(max(0.0, float(mean_sq))))


def soft_clip(audio: np.ndarray, threshold: float = 0.90) -> np.ndarray:
    """
    Soft-knee tanh limiter to prevent harsh digital clipping on speech peaks.
    Leaves values below `threshold` strictly linear and applies smooth saturation above it.
    """
    if audio.dtype != np.float32:
        x = audio.astype(np.float32)
    else:
        x = audio

    mask = np.abs(x) > threshold
    if not np.any(mask):
        return x

    out = x.copy()
    above = x[mask]
    sign = np.sign(above)
    mag = np.abs(above)

    # Tanh compression above threshold
    headroom = 1.0 - threshold
    compressed = threshold + headroom * np.tanh((mag - threshold) / headroom)
    out[mask] = sign * compressed
    return out


def calculate_snr_db(signal: np.ndarray, reference: np.ndarray) -> float:
    """
    Calculates Signal-to-Noise Ratio (SNR) in decibels between a test signal and reference.
    """
    min_len = min(len(signal), len(reference))
    if min_len == 0:
        return 0.0

    s = signal[:min_len].astype(np.float64)
    r = reference[:min_len].astype(np.float64)

    noise = s - r
    sig_power = np.mean(r ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power <= 1e-12:
        return 100.0  # Essentially infinite SNR
    if sig_power <= 1e-12:
        return 0.0

    return float(10.0 * np.log10(sig_power / noise_power))
