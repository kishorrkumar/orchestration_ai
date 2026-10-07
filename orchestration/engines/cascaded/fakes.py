"""
Deterministic Fake Providers for Engine B Testing, CI, and Keyless Demos.
Simulates STT, LLM, and TTS with configurable artificial latencies and zero external network calls.
"""

from __future__ import annotations

import asyncio
import time
from typing import List

import numpy as np
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class FakeSTTService(FrameProcessor):
    """
    Simulated STT: Collects input audio frames and emits scripted transcripts
    when user speech ends (simulated via frame counter or silence).
    """

    def __init__(
        self,
        scripted_transcripts: List[str] | None = None,
        turn_delay_ms: float = 20.0,
        sample_rate: int = 16000,
    ) -> None:
        super().__init__()
        self.scripted_transcripts = scripted_transcripts or [
            "Hello, what time is it?",
            "Can you tell me about the weather?",
            "Thank you, goodbye!",
        ]
        self._turn_idx = 0
        self.turn_delay_ms = turn_delay_ms
        self.sample_rate = sample_rate
        self._audio_frames_received = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            self._audio_frames_received += 1
            # Every ~25 frames (500ms of audio), emit a simulated transcript if available
            if self._audio_frames_received % 25 == 0 and self._turn_idx < len(self.scripted_transcripts):
                text = self.scripted_transcripts[self._turn_idx]
                self._turn_idx += 1
                await asyncio.sleep(self.turn_delay_ms / 1000.0)
                # Emit transcription frame
                await self.push_frame(
                    TranscriptionFrame(
                        text=text,
                        user_id="caller",
                        timestamp=str(time.time()),
                    )
                )


class FakeLLMService(FrameProcessor):
    """
    Simulated LLM: Receives user text and streams back response words
    with controlled token latency.
    """

    def __init__(
        self,
        token_delay_ms: float = 15.0,
        replies: List[str] | None = None,
    ) -> None:
        super().__init__()
        self.token_delay_ms = token_delay_ms
        self.replies = replies or [
            "Hello! I am your AI assistant, ready to help you.",
            "The current weather is pleasant and clear today.",
            "It was a pleasure speaking with you. Have a great day!",
        ]
        self._reply_idx = 0
        self.interrupted = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            self.interrupted = True
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            self.interrupted = False
            reply = self.replies[self._reply_idx % len(self.replies)]
            self._reply_idx += 1

            await self.push_frame(LLMFullResponseStartFrame())
            words = reply.split()
            for w in words:
                if self.interrupted:
                    break
                await asyncio.sleep(self.token_delay_ms / 1000.0)
                await self.push_frame(TextFrame(text=w + " "))
            await self.push_frame(LLMFullResponseEndFrame())


class FakeTTSService(FrameProcessor):
    """
    Simulated TTS: Receives TextFrames and streams raw 16kHz audio frames
    matching the spoken length.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 20,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.frame_samples = int(sample_rate * (chunk_duration_ms / 1000.0))
        self.frame_bytes = self.frame_samples * 2  # 16-bit PCM = 2 bytes/sample
        self.interrupted = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            self.interrupted = True
            await self.push_frame(TTSStoppedFrame())
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)

        if isinstance(frame, TextFrame):
            text = frame.text.strip()
            if not text:
                return

            self.interrupted = False
            await self.push_frame(TTSStartedFrame())

            # Synthesize ~4 frames (80ms) per word of gentle simulated audio
            num_frames = max(2, len(text) // 2)
            # Create a 440 Hz gentle sine tone
            t = np.linspace(0, self.chunk_duration_ms / 1000.0, self.frame_samples, endpoint=False)
            tone = (np.sin(2 * np.pi * 440 * t) * 16000 * 0.1).astype(np.int16).tobytes()

            for _ in range(num_frames):
                if self.interrupted:
                    break
                await asyncio.sleep(0.005)
                await self.push_frame(
                    TTSAudioRawFrame(
                        audio=tone,
                        sample_rate=self.sample_rate,
                        num_channels=1,
                    )
                )

            await self.push_frame(TTSStoppedFrame())
