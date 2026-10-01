"""
Lightweight Streaming Caller ASR and Conversational Turn Segmenter.

Key Responsibilities:
1. Caller Speech Transcription:
   - Uses faster-whisper (tiny.en/distil int8 on CPU or minimal CUDA)
   - Zero-VRAM CPU execution by default, preventing GPU contention with PersonaPlex 7B
   - Executes in separate thread/task queue, never blocking the 12.5 Hz (80 ms) audio pipeline
2. Turn Segmentation & Timestamps:
   - Segments agent output into turns based on output silence (>600ms) or caller speech onset
   - Formats every turn with separate "You" vs "Agent" bubbles and timestamps [HH:MM:SS]
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Coroutine, Any
import numpy as np

logger = logging.getLogger(__name__)

_GLOBAL_WHISPER_MODEL = None
_WHISPER_LOCK = asyncio.Lock()


def get_shared_whisper_model(model_size: str = "tiny.en", device: str = "cpu", compute_type: str = "int8"):
    """Lazy-load a shared CPU int8 faster-whisper model (0 MB GPU VRAM, ~75 MB RAM)."""
    global _GLOBAL_WHISPER_MODEL
    if _GLOBAL_WHISPER_MODEL is None:
        try:
            from faster_whisper import WhisperModel
            logger.info(f"Loading lightweight caller ASR model: faster-whisper {model_size} ({device}, {compute_type})")
            _GLOBAL_WHISPER_MODEL = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                cpu_threads=2,
            )
            logger.info("Caller ASR model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load faster-whisper: {e}. Fallback to mock transcriber.")
            _GLOBAL_WHISPER_MODEL = False
    return _GLOBAL_WHISPER_MODEL


class CallerTranscriber:
    """
    Asynchronous, non-blocking caller audio transcriber and turn manager.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        on_transcript_callback: Callable[[dict[str, Any]], Coroutine[Any, Any, None]] | None = None,
        model_size: str = "tiny.en",
    ):
        self.sample_rate = sample_rate
        self.on_transcript = on_transcript_callback
        self.model_size = model_size

        # Queue for inbound caller audio segments ready for STT
        self._transcribe_queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._is_running = False

        # Agent turn tracking
        self._agent_turn_tokens: list[str] = []
        self._agent_turn_start_time: float = 0.0
        self._last_agent_audio_time: float = 0.0

    def start(self):
        if not self._is_running:
            self._is_running = True
            self._worker_task = asyncio.create_task(self._transcription_worker())

    async def stop(self):
        self._is_running = False
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except (asyncio.CancelledError, Exception):
                pass

    def enqueue_caller_speech(self, audio: np.ndarray):
        """Enqueue completed user speech utterance for background transcription."""
        if self._is_running and len(audio) > 0:
            try:
                self._transcribe_queue.put_nowait(audio.copy())
            except Exception as e:
                logger.warning(f"Failed to queue speech for transcription: {e}")

    async def _transcription_worker(self):
        """Background worker that executes faster-whisper on CPU without blocking audio loop."""
        while self._is_running:
            try:
                audio_chunk = await self._transcribe_queue.get()
                text = await self._transcribe_audio(audio_chunk)
                if text and text.strip():
                    ts = time.strftime("%H:%M:%S")
                    logger.debug(f"[ASR Caller] Transcribed: '{text}' ({ts})")
                    if self.on_transcript:
                        await self.on_transcript({
                            "event": "transcript",
                            "role": "user",
                            "text": text.strip(),
                            "is_final": True,
                            "timestamp": ts,
                        })
                self._transcribe_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Error in caller transcription worker: {e}")
                await asyncio.sleep(0.05)

    async def _transcribe_audio(self, audio: np.ndarray) -> str:
        """Run STT in thread pool."""
        def _run_stt() -> str:
            model = get_shared_whisper_model(model_size=self.model_size)
            if not model:
                return ""

            # Ensure 16kHz for whisper if needed
            audio_16k = audio
            if self.sample_rate != 16000:
                try:
                    from scipy.signal import resample_poly
                    # 24k -> 16k is ratio 2/3
                    gcd = 8000
                    up = 16000 // gcd
                    down = self.sample_rate // gcd
                    audio_16k = resample_poly(audio, up, down).astype(np.float32)
                except Exception:
                    pass

            segments, _ = model.transcribe(
                audio_16k,
                beam_size=1,
                language="en",
                temperature=0.0,
                vad_filter=False,
            )
            return " ".join(seg.text for seg in segments).strip()

        return await asyncio.to_thread(_run_stt)

    # -------------------------------------------------------------
    # Agent Turn Segmentation
    # -------------------------------------------------------------
    def on_agent_token(self, token: str):
        """Accumulate token into active agent turn."""
        if not self._agent_turn_tokens:
            self._agent_turn_start_time = time.time()
        self._agent_turn_tokens.append(token)

    def on_agent_audio_frame(self):
        self._last_agent_audio_time = time.time()

    async def finalize_agent_turn(self, reason: str = "silence") -> dict[str, Any] | None:
        """Finalize current agent turn into a completed bubble with timestamp."""
        if not self._agent_turn_tokens:
            return None

        turn_text = "".join(self._agent_turn_tokens).strip()
        self._agent_turn_tokens = []
        if not turn_text:
            return None

        ts = time.strftime("%H:%M:%S")
        logger.debug(f"[Agent Turn] Finalized ({reason}): '{turn_text}' at {ts}")
        payload = {
            "event": "transcript",
            "role": "agent",
            "text": turn_text,
            "is_final": True,
            "timestamp": ts,
        }
        if self.on_transcript:
            await self.on_transcript(payload)
        return payload

    async def check_agent_silence_timeout(self, silence_threshold_sec: float = 0.65):
        """If agent hasn't generated audio for > 650ms and has tokens, finalize the turn."""
        if self._agent_turn_tokens and self._last_agent_audio_time > 0:
            if time.time() - self._last_agent_audio_time >= silence_threshold_sec:
                await self.finalize_agent_turn(reason="silence_timeout")
