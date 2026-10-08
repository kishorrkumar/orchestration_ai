"""
Inbound Audio Framing and Control Parser for PersonaPlex S2S WebSocket Streams.

Handles:
- Distinction between 1-byte opcode framing (0x01 + payload) and raw PCM16 samples
  (where byte 0x01 may be a valid audio sample byte).
- Identification and isolation of binary control/metadata frames (0x00, 0x02, 0x03, 0x06).
- Carry-over byte buffering for fragmented or odd-length network chunks.
- Safe PCM16 and G.711 mu-law decoding wrapped in try/except to prevent session drops.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from .codecs import decode_ulaw
from .dsp import int16_to_float32
from .resample import StreamingResampleBuffer

logger = logging.getLogger(__name__)


class InboundAudioFrameProcessor:
    """
    Robust WebSocket binary audio framing processor.
    Protects against odd-byte crashes, control frame confusion, and audio buffer misalignment.
    """

    def __init__(
        self,
        codec: str = "pcm16",
        sample_rate: int = 16000,
        model_sample_rate: int = 24000,
        out_frame_samples: int = 1920,
    ) -> None:
        self.codec = codec
        self.sample_rate = sample_rate
        self.model_sample_rate = model_sample_rate
        self.out_frame_samples = out_frame_samples

        self._carryover = bytearray()
        self.resample_buffer = StreamingResampleBuffer(
            in_rate=sample_rate,
            out_rate=model_sample_rate,
            out_frame_samples=out_frame_samples,
            quality="QQ",
        )

    def process_frame(
        self, data: bytes
    ) -> tuple[Literal["audio", "interrupt", "ping", "control", "empty", "partial", "corrupt"], list[np.ndarray], np.ndarray | None]:
        """
        Process an incoming binary frame from WebSocket client.

        Returns:
            (event_type, resampled_24k_frames, f32_chunk_samples)
        """
        if not data or len(data) == 0:
            return "empty", [], None

        # 1. Non-audio control frame identification
        # Control frames are small packets (<= 4 bytes) starting with non-audio opcodes:
        # 0x00: Handshake, 0x02: Text/metadata, 0x03: Interrupt/control, 0x06: Ping
        if len(data) <= 4 and data[0] in (0x00, 0x02, 0x03, 0x04, 0x05, 0x06):
            opcode = data[0]
            if opcode == 0x03:
                logger.info("Received binary interrupt control frame (0x03)")
                return "interrupt", [], None
            elif opcode == 0x06:
                return "ping", [], None
            else:
                return "control", [], None

        # 2. Check for framed audio vs raw PCM
        # A framing opcode (0x01) can ONLY appear at the start of a new frame when there
        # is no partial sample in carryover (len(self._carryover) == 0).
        # In a framed stream, audio starts with 0x01. If the audio is PCM16 (each sample is 2 bytes),
        # an opcode prefix of 1 byte makes the total frame length ODD (e.g. 1 + 640 = 641 bytes),
        # or it starts with Ogg Opus container '0x01' + 'OggS'.
        # If the length is EVEN, 0x01 is the low byte of an audio sample (e.g. value 1 is 0x01 0x00)
        # and MUST NOT be stripped.
        payload = data
        if len(self._carryover) == 0:
            if data[0] == 0x01 and (len(data) % 2 == 1 or data.startswith(b"\x01OggS")):
                payload = data[1:]


        if len(payload) == 0:
            return "empty", [], None

        # 3. Carry-over buffer for odd/fragmented chunks
        self._carryover.extend(payload)
        if len(self._carryover) == 0:
            return "empty", [], None

        if self.codec == "g711_ulaw":
            # 8-bit samples (1 byte each)
            pcm_bytes = bytes(self._carryover)
            self._carryover.clear()
        else:
            # 16-bit PCM (2 bytes per sample): extract even number of bytes
            usable_len = len(self._carryover) - (len(self._carryover) % 2)
            if usable_len == 0:
                # Need at least 2 bytes for 1 int16 sample
                return "partial", [], None
            pcm_bytes = bytes(self._carryover[:usable_len])
            del self._carryover[:usable_len]

        # 4. Safe PCM decoding wrapped in try/except
        try:
            if self.codec == "g711_ulaw":
                pcm16 = decode_ulaw(pcm_bytes)
                f32_samples = int16_to_float32(pcm16)
            else:
                pcm16 = np.frombuffer(pcm_bytes, dtype=np.int16)
                f32_samples = int16_to_float32(pcm16)
        except Exception as e:
            logger.warning(f"Audio frame decode warning (skipping malformed data): {e}")
            return "corrupt", [], None

        # 5. Push to 24 kHz streaming resampler
        frames_24k = self.resample_buffer.push_chunk(f32_samples)
        return "audio", frames_24k, f32_samples

    def clear(self) -> None:
        """Reset internal carryover buffer."""
        self._carryover.clear()
