"""
Dual-Channel Session Audio Blackbox Recorder for Real-Time Call Telemetry.
Records caller microphone (Channel 0) and assistant audio (Channel 1) in lockstep
at 24 kHz stereo WAV for deterministic offline diagnosis and replay.
"""

from __future__ import annotations

import logging
import pathlib
import time
from typing import Any

import numpy as np
import soundfile as sf

from .dsp import compute_rms, float32_to_int16

logger = logging.getLogger("orchestration.audio.recorder")

RECORDINGS_DIR = pathlib.Path("data/recordings")
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)


class SessionAudioRecorder:
    """
    In-memory / streaming dual-channel audio recorder for real-time call diagnostics.
    Channel 0 (Left): Inbound caller speech.
    Channel 1 (Right): Outbound assistant speech.
    """

    def __init__(self, session_id: str, sample_rate: int = 24000, enabled: bool = True) -> None:
        self.session_id = session_id
        self.sample_rate = sample_rate
        self.enabled = enabled
        self._inbound_chunks: list[np.ndarray] = []
        self._outbound_chunks: list[np.ndarray] = []
        self._start_time = time.time()
        self._closed = False
        self._file_path: pathlib.Path | None = None

    def record_inbound(self, samples: np.ndarray) -> None:
        """Record a chunk of caller audio (float32 array)."""
        if not self.enabled or self._closed or len(samples) == 0:
            return
        self._inbound_chunks.append(samples.astype(np.float32).copy())

    def record_outbound(self, samples: np.ndarray) -> None:
        """Record a chunk of assistant audio (float32 array)."""
        if not self.enabled or self._closed or len(samples) == 0:
            return
        self._outbound_chunks.append(samples.astype(np.float32).copy())

    def close(self) -> dict[str, Any]:
        """
        Finalizes dual-channel recording, writes WAV file to disk, and computes metrics.
        """
        if self._closed:
            return self.get_summary()

        self._closed = True
        if not self.enabled:
            return {"enabled": False}

        try:
            inbound = np.concatenate(self._inbound_chunks) if self._inbound_chunks else np.zeros(0, dtype=np.float32)
            outbound = np.concatenate(self._outbound_chunks) if self._outbound_chunks else np.zeros(0, dtype=np.float32)

            max_len = max(len(inbound), len(outbound))
            if max_len == 0:
                return {"enabled": True, "recorded": False, "reason": "empty_audio"}

            # Pad shorter track with silence to maintain stereo alignment
            if len(inbound) < max_len:
                inbound = np.pad(inbound, (0, max_len - len(inbound)))
            if len(outbound) < max_len:
                outbound = np.pad(outbound, (0, max_len - len(outbound)))

            stereo = np.column_stack([inbound, outbound])  # [N, 2]
            int16_stereo = float32_to_int16(stereo)

            RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
            self._file_path = RECORDINGS_DIR / f"{self.session_id}.wav"
            sf.write(str(self._file_path), int16_stereo, self.sample_rate, subtype="PCM_16")

            duration = round(max_len / self.sample_rate, 2)
            logger.info(
                f"[SessionAudioRecorder {self.session_id}] Saved dual-channel blackbox recording "
                f"to {self._file_path} ({duration}s, in_rms={compute_rms(inbound):.4f}, out_rms={compute_rms(outbound):.4f})"
            )

            return {
                "enabled": True,
                "recorded": True,
                "file_path": str(self._file_path),
                "duration_sec": duration,
                "inbound_rms": round(compute_rms(inbound), 4),
                "outbound_rms": round(compute_rms(outbound), 4),
                "sample_rate": self.sample_rate,
                "channels": 2,
            }
        except Exception as e:
            logger.error(f"[SessionAudioRecorder {self.session_id}] Error saving audio recording: {e}", exc_info=True)
            return {"enabled": True, "recorded": False, "error": str(e)}

    def get_summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "file_path": str(self._file_path) if self._file_path else None,
            "session_id": self.session_id,
        }
