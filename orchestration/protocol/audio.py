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
    return (audio.astype(np.float32) / 32767.0)


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

    def push_pcm_bytes(self, pcm_bytes: bytes, is_int16: bool = False) -> None:
        """Push raw PCM bytes into buffer."""
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
