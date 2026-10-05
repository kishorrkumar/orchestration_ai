"""
Anti-aliased Streaming Audio Resamplers for Voice Agent Platform.
Supports stateful streaming between 16 kHz (WebRTC/client), 24 kHz (PersonaPlex S2S),
and 8 kHz (Telephony PSTN / G.711).
"""

from __future__ import annotations

import math
from typing import Generator, List, Optional, Union

import numpy as np

try:
    import soxr
    _HAS_SOXR = True
except ImportError:
    soxr = None
    _HAS_SOXR = False

try:
    from scipy.signal import resample_poly
    _HAS_SCIPY = True
except ImportError:
    resample_poly = None
    _HAS_SCIPY = False


class AudioResampler:
    """
    Stateful streaming resampler with anti-aliasing.
    Maintains internal filter state across chunk boundaries to eliminate clicks and phase jumps.
    """

    def __init__(
        self,
        in_rate: int = 16000,
        out_rate: int = 24000,
        quality: str = "QQ",
        dtype: type = np.float32,
    ):
        """
        Initialize resampler.

        Args:
            in_rate: Input sample rate in Hz (e.g. 16000 or 24000)
            out_rate: Output sample rate in Hz (e.g. 24000 or 16000)
            quality: 'QQ' (ultra-low latency, recommended for real-time),
                     'LQ', 'MQ', 'HQ', or 'VHQ'.
            dtype: np.float32 (default) or np.int16
        """
        self.in_rate = in_rate
        self.out_rate = out_rate
        self.quality = quality
        self.dtype = dtype
        self.is_passthrough = (in_rate == out_rate)

        if not self.is_passthrough and _HAS_SOXR:
            self._stream = soxr.ResampleStream(
                in_rate=float(in_rate),
                out_rate=float(out_rate),
                num_channels=1,
                dtype="float32",
                quality=quality,
            )
        else:
            self._stream = None

        # Fallback polyphase state if soxr is not available
        self._poly_ratio = math.gcd(in_rate, out_rate)
        self._poly_up = out_rate // self._poly_ratio
        self._poly_down = in_rate // self._poly_ratio

    def resample_chunk(
        self,
        chunk: Union[np.ndarray, bytes],
        last: bool = False,
        output_int16: bool = False,
    ) -> np.ndarray:
        """
        Resample a single audio chunk in streaming fashion.

        Args:
            chunk: Input audio chunk as numpy array (float32 [-1,1] or int16) or raw PCM bytes.
            last: True if this is the final chunk in the stream (flushes residual filter state).
            output_int16: If True, returns np.int16; otherwise np.float32.

        Returns:
            1D numpy array of resampled samples.
        """
        if len(chunk) == 0:
            return np.empty(0, dtype=np.int16 if output_int16 else np.float32)

        # Convert input to float32
        if isinstance(chunk, bytes):
            # Assume 16-bit PCM bytes
            raw_i16 = np.frombuffer(chunk, dtype=np.int16)
            in_f32 = np.clip(raw_i16.astype(np.float32) / 32767.0, -1.0, 1.0)
        elif chunk.dtype == np.int16:
            in_f32 = np.clip(chunk.astype(np.float32) / 32767.0, -1.0, 1.0)
        elif chunk.dtype != np.float32:
            in_f32 = chunk.astype(np.float32)
        else:
            in_f32 = chunk

        if self.is_passthrough:
            out_f32 = in_f32
        elif self._stream is not None:
            out_f32 = self._stream.resample_chunk(in_f32, last=last)
        elif _HAS_SCIPY and resample_poly is not None:
            out_f32 = resample_poly(in_f32, self._poly_up, self._poly_down).astype(np.float32)
        else:
            # Linear interpolation fallback
            num_out = int(round(len(in_f32) * self.out_rate / self.in_rate))
            out_f32 = np.interp(
                np.linspace(0.0, 1.0, num_out, endpoint=False),
                np.linspace(0.0, 1.0, len(in_f32), endpoint=False),
                in_f32,
            ).astype(np.float32)

        if output_int16:
            clipped = np.clip(out_f32, -1.0, 1.0)
            return (clipped * 32767.0).astype(np.int16)
        return out_f32

    def reset(self) -> None:
        """Reset internal filter state."""
        if self._stream is not None:
            self._stream.clear()


class StreamingResampleBuffer:
    """
    Accumulator buffer that resamples streaming input audio and emits
    exact, fixed-size output frames (e.g. 20 ms / 480 samples at 24 kHz, or 20 ms / 320 samples at 16 kHz).
    """

    def __init__(
        self,
        in_rate: int = 16000,
        out_rate: int = 24000,
        out_frame_samples: int = 480,
        quality: str = "QQ",
        output_int16: bool = False,
    ):
        self.resampler = AudioResampler(in_rate=in_rate, out_rate=out_rate, quality=quality)
        self.out_frame_samples = out_frame_samples
        self.output_int16 = output_int16
        self._accumulator = np.empty(0, dtype=np.float32)

    def push_chunk(self, chunk: Union[np.ndarray, bytes]) -> List[np.ndarray]:
        """
        Push incoming chunk, resample, and return list of exact `out_frame_samples` frames.
        """
        resampled = self.resampler.resample_chunk(chunk, last=False, output_int16=False)
        if len(resampled) > 0:
            if len(self._accumulator) == 0:
                self._accumulator = resampled
            else:
                self._accumulator = np.concatenate([self._accumulator, resampled])

        frames = []
        while len(self._accumulator) >= self.out_frame_samples:
            frame_f32 = self._accumulator[:self.out_frame_samples]
            self._accumulator = self._accumulator[self.out_frame_samples:]

            if self.output_int16:
                frame = (np.clip(frame_f32, -1.0, 1.0) * 32767.0).astype(np.int16)
            else:
                frame = frame_f32
            frames.append(frame)

        return frames

    def flush(self) -> List[np.ndarray]:
        """
        Flush any remaining audio by zero-padding to the final frame if needed.
        """
        final_resampled = self.resampler.resample_chunk(np.empty(0, dtype=np.float32), last=True)
        if len(final_resampled) > 0:
            self._accumulator = np.concatenate([self._accumulator, final_resampled])

        frames = []
        while len(self._accumulator) >= self.out_frame_samples:
            frame_f32 = self._accumulator[:self.out_frame_samples]
            self._accumulator = self._accumulator[self.out_frame_samples:]
            if self.output_int16:
                frame = (np.clip(frame_f32, -1.0, 1.0) * 32767.0).astype(np.int16)
            else:
                frame = frame_f32
            frames.append(frame)

        if len(self._accumulator) > 0:
            # Pad final frame with silence
            pad_len = self.out_frame_samples - len(self._accumulator)
            frame_f32 = np.pad(self._accumulator, (0, pad_len), mode="constant")
            self._accumulator = np.empty(0, dtype=np.float32)
            if self.output_int16:
                frame = (np.clip(frame_f32, -1.0, 1.0) * 32767.0).astype(np.int16)
            else:
                frame = frame_f32
            frames.append(frame)

        return frames

    def reset(self) -> None:
        """Clear accumulator and reset resampler state."""
        self._accumulator = np.empty(0, dtype=np.float32)
        self.resampler.reset()


def resample_oneshot(
    audio: np.ndarray,
    in_rate: int,
    out_rate: int,
    quality: str = "HQ",
) -> np.ndarray:
    """
    High-fidelity one-shot resampling of an entire audio buffer.
    """
    if in_rate == out_rate:
        return audio.copy()

    is_int16 = (audio.dtype == np.int16)
    if is_int16:
        audio_f32 = np.clip(audio.astype(np.float32) / 32767.0, -1.0, 1.0)
    elif audio.dtype != np.float32:
        audio_f32 = audio.astype(np.float32)
    else:
        audio_f32 = audio

    if _HAS_SOXR:
        out_f32 = soxr.resample(audio_f32, in_rate, out_rate, quality=quality)
    elif _HAS_SCIPY and resample_poly is not None:
        g = math.gcd(in_rate, out_rate)
        out_f32 = resample_poly(audio_f32, out_rate // g, in_rate // g).astype(np.float32)
    else:
        num_out = int(round(len(audio_f32) * out_rate / in_rate))
        out_f32 = np.interp(
            np.linspace(0.0, 1.0, num_out, endpoint=False),
            np.linspace(0.0, 1.0, len(audio_f32), endpoint=False),
            audio_f32,
        ).astype(np.float32)

    if is_int16:
        return (np.clip(out_f32, -1.0, 1.0) * 32767.0).astype(np.int16)
    return out_f32
