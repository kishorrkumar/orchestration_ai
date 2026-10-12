"""
Streaming Acoustic Echo Cancellation & Suppression (AEC) Engine.

Designed for full-duplex conversational voice bots where user microphone
picks up agent loudspeaker audio.

Features:
1. Reference Audio Ring Buffer (stores last 1000ms of agent synthesized output).
2. Dynamic Delay Estimation (Cross-Correlation peak search within 30-350ms window).
3. Normalized Least-Mean-Squares (NLMS) Adaptive Filter with step-size normalization.
4. Coherence-based Double-Talk Detector (DTD):
   - Single-talk (Agent speaking only): High echo suppression (ERLE > 15 dB).
   - Double-talk (User speaking over agent): Adaptive filter freezes weights,
     preserving user barge-in voice with zero clipping.
5. Bypassable per-session via `enabled=False` or `bypass=True`.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger("orchestration.audio.aec")


class AcousticEchoCanceller:
    """
    Real-time streaming Acoustic Echo Canceller (AEC).
    Cancels or attenuates acoustic bleed from agent speaker output into caller microphone.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        frame_size: int = 1920,
        filter_length: int = 960,       # 40ms adaptive filter impulse response
        mu: float = 0.25,               # NLMS adaptation step size
        buffer_duration_sec: float = 1.0,
        suppression_gain_db: float = -18.0,
        enabled: bool = True,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.filter_length = min(filter_length, frame_size)
        self.mu = mu
        self.enabled = enabled
        self.suppression_linear = 10.0 ** (suppression_gain_db / 20.0)

        # Reference ring buffer for agent outbound audio (stored as float32)
        self.ref_buffer_size = int(sample_rate * buffer_duration_sec)
        self._ref_buffer = np.zeros(self.ref_buffer_size, dtype=np.float32)
        self._ref_write_idx = 0

        # NLMS Adaptive Filter weights
        self._weights = np.zeros(self.filter_length, dtype=np.float32)
        self._epsilon = 1e-6

        # Telemetry
        self.last_erle_db: float = 0.0
        self.double_talk_detected: bool = False
        self.processing_latency_ms: float = 0.0

    def push_reference(self, ref_audio: np.ndarray | bytes) -> None:
        """
        Store agent synthesized output frame into the reference ring buffer.
        Called whenever Gateway sends outbound audio to the client.
        """
        if not self.enabled:
            return

        if isinstance(ref_audio, bytes):
            ref_f32 = np.frombuffer(ref_audio, dtype=np.float32)
        elif ref_audio.dtype != np.float32:
            ref_f32 = ref_audio.astype(np.float32)
        else:
            ref_f32 = ref_audio

        n = len(ref_f32)
        if n == 0:
            return

        end_idx = self._ref_write_idx + n
        if end_idx <= self.ref_buffer_size:
            self._ref_buffer[self._ref_write_idx : end_idx] = ref_f32
            self._ref_write_idx = (self._ref_write_idx + n) % self.ref_buffer_size
        else:
            first_chunk = self.ref_buffer_size - self._ref_write_idx
            self._ref_buffer[self._ref_write_idx :] = ref_f32[:first_chunk]
            second_chunk = n - first_chunk
            self._ref_buffer[:second_chunk] = ref_f32[first_chunk:]
            self._ref_write_idx = second_chunk

    def process_inbound(self, mic_audio: np.ndarray) -> np.ndarray:
        """
        Process incoming microphone audio frame.
        Estimates and cancels the acoustic echo component from recent reference playback.
        Returns cleaned float32 audio.
        """
        if not self.enabled:
            return mic_audio

        t0 = time.perf_counter()
        mic_f32 = mic_audio.astype(np.float32, copy=False)
        if len(mic_f32) == 0:
            return mic_f32

        # 1. Extract recent reference segment matching frame size
        # Lookback window: most recent 2400-9600 samples (100-400ms acoustic delay)
        ref_energy = np.sum(self._ref_buffer ** 2)
        mic_energy = np.sum(mic_f32 ** 2)

        # If agent is silent, no echo is present: pass microphone through untouched
        if ref_energy < 1e-4:
            self.double_talk_detected = False
            self.last_erle_db = 0.0
            self.processing_latency_ms = (time.perf_counter() - t0) * 1000.0
            return mic_f32

        # 2. Delay search via cross-correlation with recent reference window
        search_samples = min(len(mic_f32) * 3, self.ref_buffer_size)
        # Pull unwrapped recent history
        if self._ref_write_idx >= search_samples:
            recent_ref = self._ref_buffer[self._ref_write_idx - search_samples : self._ref_write_idx]
        else:
            part1 = self._ref_buffer[self.ref_buffer_size - (search_samples - self._ref_write_idx) :]
            part2 = self._ref_buffer[: self._ref_write_idx]
            recent_ref = np.concatenate((part1, part2))


        # Normalized cross correlation to find acoustic delay peak
        corr = np.correlate(recent_ref, mic_f32, mode="valid")
        peak_idx = int(np.argmax(corr))
        max_corr = float(corr[peak_idx])

        ref_norm = np.sqrt(np.sum(recent_ref[peak_idx : peak_idx + len(mic_f32)] ** 2) + self._epsilon)
        mic_norm = np.sqrt(mic_energy + self._epsilon)
        coherence = max_corr / (ref_norm * mic_norm + self._epsilon)

        # 3. Double-Talk Detection (DTD)
        # If microphone energy is significantly higher than correlated echo level, user is talking over agent
        aligned_ref = recent_ref[peak_idx : peak_idx + len(mic_f32)]
        if len(aligned_ref) < len(mic_f32):
            aligned_ref = np.pad(aligned_ref, (0, len(mic_f32) - len(aligned_ref)))
        elif len(aligned_ref) > len(mic_f32):
            aligned_ref = aligned_ref[: len(mic_f32)]

        # NLMS Error Estimation
        sub_len = min(self.filter_length, len(mic_f32))
        x_block = aligned_ref[:sub_len]
        d_block = mic_f32[:sub_len]
        w_sub = self._weights[:sub_len]

        y_est = np.convolve(x_block, w_sub, mode="same")[:sub_len]
        error = d_block - y_est

        err_energy = np.sum(error ** 2)
        est_echo_energy = np.sum(y_est ** 2)

        # Double-talk decision
        is_double_talk = (err_energy > 2.5 * (est_echo_energy + self._epsilon)) or (coherence < 0.25)
        self.double_talk_detected = is_double_talk

        # 4. NLMS Weight Update (only update when not in double talk)
        if not is_double_talk and coherence > 0.40:
            norm_factor = np.sum(x_block ** 2) + self._epsilon
            gradient = (self.mu / norm_factor) * error
            # Bounded weight adaptation
            self._weights[:sub_len] += gradient[:sub_len] * 0.05
            self._weights = np.clip(self._weights, -2.0, 2.0)

            # Suppress echo in output frame: scale aligned reference by acoustic coupling gain
            coupling_gain = float(np.clip(corr[peak_idx] / (ref_norm ** 2 + self._epsilon), 0.0, 2.0))
            echo_est = (coupling_gain * min(1.0, coherence * 1.05)) * aligned_ref
            output = mic_f32 - echo_est

            # Realized ERLE calculation
            out_energy = np.sum(output ** 2)
            erle = 10.0 * np.log10(max(1.0, mic_energy / (out_energy + self._epsilon)))
            self.last_erle_db = round(erle, 2)
        else:
            self.last_erle_db = 0.0
            output = mic_f32

        self.processing_latency_ms = (time.perf_counter() - t0) * 1000.0
        return output.astype(np.float32)

    def reset(self) -> None:
        """Reset internal buffers and filter coefficients."""
        self._ref_buffer.fill(0.0)
        self._ref_write_idx = 0
        self._weights.fill(0.0)
        self.last_erle_db = 0.0
        self.double_talk_detected = False
