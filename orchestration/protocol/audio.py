"""
Audio framing, conversions, and buffer utilities for PersonaPlex.

PersonaPlex specifications:
- Sample Rate: 24,000 Hz
- Frame Rate: 12.5 Hz
- Frame Size: 1,920 samples (80 ms)
- Channels: 1 (Mono)
"""

from __future__ import annotations

import math

import numpy as np

SAMPLE_RATE: int = 24000
FRAME_RATE: float = 12.5
FRAME_SIZE: int = 1920  # 24000 / 12.5
CHANNELS: int = 1

BYTES_PER_SAMPLE_INT16: int = 2
BYTES_PER_FRAME_INT16: int = FRAME_SIZE * BYTES_PER_SAMPLE_INT16  # 3840 bytes

BYTES_PER_SAMPLE_FLOAT32: int = 4
BYTES_PER_FRAME_FLOAT32: int = FRAME_SIZE * BYTES_PER_SAMPLE_FLOAT32  # 7680 bytes


def float32_to_int16(audio: np.ndarray) -> np.ndarray:
    """Convert float32 audio [-1.0, 1.0] to int16 [-32768, 32767]."""
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def int16_to_float32(audio: np.ndarray) -> np.ndarray:
    """Convert int16 audio [-32768, 32767] to float32 [-1.0, 1.0]."""
    return np.clip(audio.astype(np.float32) / 32767.0, -1.0, 1.0)


def compute_rms(samples: np.ndarray) -> float:
    """Compute Root Mean Square (RMS) energy of audio samples."""
    if len(samples) == 0:
        return 0.0
    if samples.dtype == np.int16:
        float_samples = int16_to_float32(samples)
    else:
        float_samples = samples
    mean_sq = np.mean(float_samples ** 2)
    return float(math.sqrt(max(0.0, float(mean_sq))))


def generate_silence_frame(dtype=np.float32) -> np.ndarray:
    """Generate a single silent frame (1920 samples)."""
    return np.zeros(FRAME_SIZE, dtype=dtype)


def high_pass_filter(samples: np.ndarray, cutoff_hz: float = 80.0, fs: int = SAMPLE_RATE) -> np.ndarray:
    """
    1st-order IIR High-Pass Filter to remove DC offset, desk rumble,
    and 50/60 Hz electrical hum.
    """
    if len(samples) < 2:
        return samples
    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    dt = 1.0 / fs
    alpha = rc / (rc + dt)
    out = np.empty_like(samples)
    out[0] = samples[0]
    for i in range(1, len(samples)):
        out[i] = alpha * (out[i - 1] + samples[i] - samples[i - 1])
    return out


class AdaptiveNoiseCanceller:
    """
    Adaptive Noise Cancellation and Suppression Layer for 24 kHz speech.
    Combines:
    1. Low-frequency rumble / hum attenuation (High-Pass Filter).
    2. Adaptive background noise spectrum estimation.
    3. Spectral subtraction with over-subtraction factor and spectral floor.
    4. Soft-knee dynamic noise gating during pauses.
    """

    def __init__(
        self,
        frame_size: int = FRAME_SIZE,
        sample_rate: int = SAMPLE_RATE,
        alpha: float = 2.0,       # Over-subtraction factor
        beta: float = 0.03,       # Spectral floor fraction
        gate_threshold_rms: float = 0.012, # Ambient noise floor threshold
    ):
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self.alpha = alpha
        self.beta = beta
        self.gate_threshold = gate_threshold_rms
        self.noise_spectrum: np.ndarray | None = None
        self.adaptation_rate: float = 0.08

    def clean_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process and de-noise an incoming 1920-sample audio frame."""
        if len(frame) == 0:
            return frame

        # Ensure float32
        if frame.dtype != np.float32:
            frame = frame.astype(np.float32)

        # 1. High-Pass Filter (remove low-frequency hum < 80 Hz)
        hp = high_pass_filter(frame, cutoff_hz=80.0, fs=self.sample_rate)

        # 2. Check energy
        rms = compute_rms(hp)

        # 3. Spectral Subtraction
        fft = np.fft.rfft(hp)
        mag = np.abs(fft)
        phase = np.angle(fft)

        if self.noise_spectrum is None:
            self.noise_spectrum = mag.copy()
        elif rms < self.gate_threshold:
            # During quiet moments, adapt noise profile to ambient room sound
            self.noise_spectrum = (1.0 - self.adaptation_rate) * self.noise_spectrum + self.adaptation_rate * mag

        # Subtract estimated background noise
        subtracted = mag - self.alpha * self.noise_spectrum
        cleaned_mag = np.maximum(subtracted, self.beta * mag)

        # 4. Noise gate attenuation for low-energy frames
        if rms < self.gate_threshold * 0.7:
            cleaned_mag *= 0.15  # -16 dB attenuation on quiet background hiss

        # Reconstruct time-domain signal
        cleaned_fft = cleaned_mag * np.exp(1j * phase)
        cleaned = np.fft.irfft(cleaned_fft, n=len(frame)).astype(np.float32)

        return cleaned


class AudioFrameBuffer:
    """
    Accumulator buffer for continuous streaming audio.
    Accepts arbitrary chunk sizes (bytes or numpy arrays) and yields
    exact 1,920-sample (80ms) frames for the PersonaPlex / Mimi pipeline.
    """

    def __init__(self, dtype=np.float32, max_buffer_frames: int = 100):
        self.dtype = dtype
        self.max_buffer_samples = max_buffer_frames * FRAME_SIZE
        self._buffer: np.ndarray = np.empty(0, dtype=dtype)
        self._underrun_count: int = 0
        self._total_frames_pushed: int = 0
        self._total_frames_popped: int = 0
        self._byte_remainder: bytes = b""

    def push_pcm_bytes(self, pcm_bytes: bytes, is_int16: bool = False) -> None:
        """Push raw PCM bytes into buffer."""
        if not pcm_bytes and not self._byte_remainder:
            return
        if self._byte_remainder:
            pcm_bytes = self._byte_remainder + pcm_bytes
            self._byte_remainder = b""

        elem_size = 2 if is_int16 else 4
        remainder = len(pcm_bytes) % elem_size
        if remainder != 0:
            self._byte_remainder = pcm_bytes[-remainder:]
            pcm_bytes = pcm_bytes[:-remainder]

        if not pcm_bytes:
            return

        if is_int16:
            samples_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
            samples = int16_to_float32(samples_i16) if self.dtype == np.float32 else samples_i16
        else:
            samples_f32 = np.frombuffer(pcm_bytes, dtype=np.float32)
            samples = float32_to_int16(samples_f32) if self.dtype == np.int16 else samples_f32

        self.push_samples(samples)

    def push_samples(self, samples: np.ndarray) -> None:
        """Push a numpy array of audio samples into the buffer."""
        if samples.ndim > 1:
            samples = samples.flatten()
        if samples.dtype != self.dtype:
            if self.dtype == np.float32 and samples.dtype == np.int16:
                samples = int16_to_float32(samples)
            elif self.dtype == np.int16 and samples.dtype == np.float32:
                samples = float32_to_int16(samples)
            else:
                samples = samples.astype(self.dtype)

        self._buffer = np.concatenate((self._buffer, samples))
        self._total_frames_pushed += len(samples) // FRAME_SIZE

        # Guard against buffer bloat if consumer is disconnected or lagging
        if len(self._buffer) > self.max_buffer_samples:
            dropped = len(self._buffer) - self.max_buffer_samples
            self._buffer = self._buffer[dropped:]

    def has_frame(self) -> bool:
        """Returns True if at least one complete 1920-sample frame is ready."""
        return len(self._buffer) >= FRAME_SIZE

    def pop_frame(self) -> np.ndarray | None:
        """
        Pop one exact 1,920-sample frame from the buffer.
        Returns None if buffer has fewer than FRAME_SIZE samples.
        """
        if len(self._buffer) < FRAME_SIZE:
            self._underrun_count += 1
            return None
        frame = self._buffer[:FRAME_SIZE]
        self._buffer = self._buffer[FRAME_SIZE:]
        self._total_frames_popped += 1
        return frame

    def pop_all_available_frames(self) -> list[np.ndarray]:
        """Pop all complete frames currently available in buffer."""
        frames = []
        while self.has_frame():
            frame = self.pop_frame()
            if frame is not None:
                frames.append(frame)
        return frames

    def clear(self) -> None:
        """Clear all buffered audio."""
        self._buffer = np.empty(0, dtype=self.dtype)
        self._byte_remainder = b""

    @property
    def available_frames(self) -> int:
        """Returns number of full frames currently ready in buffer."""
        return len(self._buffer) // FRAME_SIZE

    def peek_frame(self, index: int = 0) -> np.ndarray | None:
        """Peek at an available frame at 0-indexed position without popping."""
        start = index * FRAME_SIZE
        end = start + FRAME_SIZE
        if len(self._buffer) < end:
            return None
        return self._buffer[start:end]

    @property
    def buffered_samples(self) -> int:
        return len(self._buffer)

    @property
    def buffered_frames(self) -> float:
        return len(self._buffer) / FRAME_SIZE

    @property
    def buffered_duration_ms(self) -> float:
        return (len(self._buffer) / SAMPLE_RATE) * 1000.0

    @property
    def underruns(self) -> int:
        return self._underrun_count
