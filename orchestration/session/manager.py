"""
Voice Session and Lifecycle State Machine Manager.

Coordinates:
- State transitions (INITIALIZING -> PROMPTING -> ACTIVE -> INTERRUPTED -> COMPLETED)
- Bi-directional audio & text pipelining between client WebSocket and worker
- 24kHz 1920-sample audio frame alignment
- Barge-in & interruption detection
- Session telemetry and real-time observability
"""

from __future__ import annotations
import asyncio
from enum import Enum
import logging
import time
import uuid
from typing import Dict, List, Optional, Any

import numpy as np

from ..persona.registry import PersonaConfig
from ..protocol.messages import (
    WSMessage,
    MessageType,
    HandshakeMessage,
    AudioMessage,
    TextMessage,
    ControlMessage,
    ControlAction,
    MetadataMessage,
    ErrorMessage,
    encode_message,
    decode_message,
)
from ..protocol.audio import (
    FRAME_SIZE,
    SAMPLE_RATE,
    AudioFrameBuffer,
    compute_rms,
    AdaptiveNoiseCanceller,
)
from ..worker.client import PersonaPlexWorkerClient
from ..worker.pool import WorkerPool

logger = logging.getLogger(__name__)


class SessionState(str, Enum):
    INITIALIZING = "INITIALIZING"
    PROMPTING = "PROMPTING"
    ACTIVE = "ACTIVE"
    INTERRUPTED = "INTERRUPTED"
    CLOSING = "CLOSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class SessionMetrics:
    def __init__(self, session_id: str, persona_id: str, worker_id: str):
        self.session_id = session_id
        self.persona_id = persona_id
        self.worker_id = worker_id
        self.state = SessionState.INITIALIZING
        self.created_at = time.time()
        self.connected_at: Optional[float] = None
        self.ended_at: Optional[float] = None

        # 1. Listening
        self.user_frames_in = 0
        self.dropped_frames = 0
        self.vad_speech_frames = 0

        # 2. Understanding
        self.text_tokens_out = 0
        self.transcript_tokens: List[str] = []

        # 3. Reasoning & Goals
        self.business_goal: Optional[Dict[str, Any]] = None
        self.reasoning_notes: List[str] = []

        # 4. Speaking
        self.agent_frames_out = 0
        self.clipping_events = 0
        self.underrun_events = 0

        # 5. Latency
        self.ttfa_history_ms: List[float] = []
        self.frame_step_times_ms: List[float] = []

        # 6. Conversation Dynamics
        self.barge_in_events = 0
        self.backchannel_events = 0
        self.turn_count = 0
        self.silence_frames = 0

        # 7. Task Success & Outcome Tagging
        self.outcome_tag: str = "in_progress"  # in_progress, success, resolved, escalated, dropped
        self.outcome_metadata: Dict[str, Any] = {}

    @property
    def duration_sec(self) -> float:
        end = self.ended_at or time.time()
        start = self.connected_at or self.created_at
        return max(0.0, end - start)

    @property
    def full_transcript(self) -> str:
        return "".join(self.transcript_tokens).strip()

    @property
    def ttfa_p50_ms(self) -> float:
        if not self.ttfa_history_ms:
            return 0.0
        s = sorted(self.ttfa_history_ms)
        return round(s[len(s) // 2], 2)

    @property
    def frame_step_avg_ms(self) -> float:
        if not self.frame_step_times_ms:
            return 0.0
        return round(sum(self.frame_step_times_ms) / len(self.frame_step_times_ms), 2)

    def record_ttfa(self, ttfa_ms: float) -> None:
        self.ttfa_history_ms.append(round(ttfa_ms, 2))

    def record_frame_step(self, step_ms: float) -> None:
        self.frame_step_times_ms.append(round(step_ms, 2))

    def tag_outcome(self, outcome: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.outcome_tag = outcome
        if metadata:
            self.outcome_metadata.update(metadata)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "persona_id": self.persona_id,
            "worker_id": self.worker_id,
            "state": self.state.value,
            "created_at": self.created_at,
            "connected_at": self.connected_at,
            "ended_at": self.ended_at,
            "duration_sec": round(self.duration_sec, 2),
            # 7 Dimensions
            "listening": {
                "user_frames_in": self.user_frames_in,
                "dropped_frames": self.dropped_frames,
                "vad_speech_frames": self.vad_speech_frames,
            },
            "understanding": {
                "transcript": self.full_transcript,
                "text_tokens_out": self.text_tokens_out,
                "word_count": len(self.full_transcript.split()),
            },
            "reasoning": {
                "persona_id": self.persona_id,
                "business_goal": self.business_goal,
                "reasoning_notes": self.reasoning_notes,
            },
            "speaking": {
                "agent_frames_out": self.agent_frames_out,
                "clipping_events": self.clipping_events,
                "underrun_events": self.underrun_events,
            },
            "latency": {
                "ttfa_p50_ms": self.ttfa_p50_ms,
                "frame_step_avg_ms": self.frame_step_avg_ms,
                "target_ttfa_met": self.ttfa_p50_ms < 300.0 if self.ttfa_history_ms else True,
                "target_step_met": self.frame_step_avg_ms < 80.0 if self.frame_step_times_ms else True,
            },
            "conversation": {
                "barge_in_events": self.barge_in_events,
                "backchannel_events": self.backchannel_events,
                "turn_count": self.turn_count,
            },
            "task_success": {
                "outcome_tag": self.outcome_tag,
                "outcome_metadata": self.outcome_metadata,
            },
            # Top-level backwards compatibility fields
            "user_frames_in": self.user_frames_in,
            "agent_frames_out": self.agent_frames_out,
            "text_tokens_out": self.text_tokens_out,
            "barge_in_events": self.barge_in_events,
            "underrun_events": self.underrun_events,
            "transcript": self.full_transcript,
        }


class VoiceSession:
    """
    Manages a single client-agent interaction session.
    """

    def __init__(
        self,
        session_id: str,
        persona: PersonaConfig,
        worker: PersonaPlexWorkerClient,
        pool: WorkerPool,
        barge_in_rms_threshold: float = 0.03,
    ):
        self.session_id = session_id
        self.persona = persona
        self.worker = worker
        self.pool = pool
        self.barge_in_rms_threshold = barge_in_rms_threshold

        self.metrics = SessionMetrics(session_id, persona.id, worker.worker_id)
        self.inbound_buffer = AudioFrameBuffer(dtype=np.float32)
        self.noise_canceller = AdaptiveNoiseCanceller()
        from ..audio.cleaner import CallerAudioCleaner
        self.cleaner = CallerAudioCleaner(sample_rate=SAMPLE_RATE, frame_size=FRAME_SIZE)

        self._state = SessionState.INITIALIZING
        self._stop_event = asyncio.Event()
        self._user_speaking = False
        self._last_user_speech_time = 0.0
        self._speech_frames_count = 0

    @property
    def state(self) -> SessionState:
        return self._state

    def set_state(self, new_state: SessionState) -> None:
        self._state = new_state
        self.metrics.state = new_state
        logger.debug(f"Session {self.session_id} state -> {new_state.value}")

    async def start(self) -> None:
        """Connect to worker and complete handshake."""
        self.set_state(SessionState.PROMPTING)
        try:
            await self.worker.connect(self.session_id, self.persona)
            self.set_state(SessionState.ACTIVE)
            self.metrics.connected_at = time.time()
        except Exception as e:
            self.set_state(SessionState.FAILED)
            raise e

    async def ingest_client_message(self, raw_bytes: bytes) -> Optional[WSMessage]:
        """
        Process an incoming raw binary message from client:
        - If audio (0x01): de-noise, buffer, slice into 1920-sample frames, check barge-in, push to worker.
        - If text (0x02): forward to worker.
        - If control (0x03): forward to worker.
        """
        try:
            msg = decode_message(raw_bytes)
        except Exception as e:
            return ErrorMessage(error=f"Malformed frame: {str(e)}")

        if self._state not in (SessionState.ACTIVE, SessionState.INTERRUPTED):
            if msg.type == MessageType.AUDIO:
                return ErrorMessage(error="Handshake not completed. Audio not accepted before session is active.")
            elif msg.type == MessageType.HANDSHAKE:
                return None
            return None

        if msg.type == MessageType.AUDIO:
            audio_data = msg.data
            self.inbound_buffer.push_pcm_bytes(audio_data, is_int16=False)

            # Pop all available 1920-sample frames and feed to worker
            frames = self.inbound_buffer.pop_all_available_frames()
            for raw_frame in frames:
                self.metrics.user_frames_in += 1
                rms = compute_rms(raw_frame)

                # Voice Activity & Barge-in tracking
                if rms > self.barge_in_rms_threshold:
                    if not self._user_speaking:
                        self._user_speaking = True
                        self.metrics.barge_in_events += 1
                        self.set_state(SessionState.INTERRUPTED)
                    self._last_user_speech_time = time.time()
                elif self._user_speaking and (time.time() - self._last_user_speech_time > 0.6):
                    # 600ms conversational hangtime before taking the floor
                    self._user_speaking = False
                    self.set_state(SessionState.ACTIVE)

                await self.worker.send_audio(raw_frame)

        elif msg.type == MessageType.TEXT:
            await self.worker.send_text(msg.text)
        elif msg.type == MessageType.CONTROL:
            await self.worker.send_control(msg.action)

        return msg

    async def run_worker_forwarder(self, send_to_client_fn) -> None:
        """
        Receive generated frames and text tokens from worker and forward to client.
        """
        try:
            async for worker_msg in self.worker.recv_messages():
                if self._stop_event.is_set():
                    break

                if worker_msg.type == MessageType.AUDIO:
                    self.metrics.agent_frames_out += 1
                    # Forward agent audio 0x01 to client
                    await send_to_client_fn(encode_message(worker_msg))

                elif worker_msg.type == MessageType.TEXT:
                    self.metrics.text_tokens_out += 1
                    self.metrics.transcript_tokens.append(worker_msg.text)
                    # Forward text token 0x02 to client
                    await send_to_client_fn(encode_message(worker_msg))

                elif worker_msg.type == MessageType.METADATA:
                    await send_to_client_fn(encode_message(worker_msg))

                elif worker_msg.type == MessageType.ERROR:
                    await send_to_client_fn(encode_message(worker_msg))
                    break

        except Exception as e:
            logger.error(f"Error in session {self.session_id} worker forwarder: {e}")
            self.set_state(SessionState.FAILED)
        finally:
            self._stop_event.set()

    async def close(self) -> None:
        """Cleanly close session, update metrics, and release worker to pool."""
        if self._state in (SessionState.COMPLETED, SessionState.CLOSING):
            return

        self.set_state(SessionState.CLOSING)
        self._stop_event.set()
        self.metrics.ended_at = time.time()

        try:
            await self.pool.release_worker(self.worker.worker_id)
        except Exception as e:
            logger.warning(f"Error releasing worker {self.worker.worker_id}: {e}")

        self.set_state(SessionState.COMPLETED)
        logger.info(
            f"Session {self.session_id} ended. "
            f"Duration: {self.metrics.duration_sec:.1f}s | "
            f"Frames In: {self.metrics.user_frames_in} | "
            f"Frames Out: {self.metrics.agent_frames_out} | "
            f"Barge-ins: {self.metrics.barge_in_events}"
        )


class SessionManager:
    """Manages active and historical sessions."""

    def __init__(self, pool: WorkerPool):
        self.pool = pool
        self._active_sessions: Dict[str, VoiceSession] = {}
        self._session_history: List[dict] = []

    async def create_session(
        self,
        persona: PersonaConfig,
        session_id: Optional[str] = None,
        timeout: float = 5.0,
    ) -> VoiceSession:
        """Acquire a worker and initialize a new VoiceSession."""
        sid = session_id or f"sess_{uuid.uuid4().hex[:12]}"
        worker = await self.pool.acquire_worker(session_id=sid, timeout=timeout)
        session = VoiceSession(session_id=sid, persona=persona, worker=worker, pool=self.pool)
        self._active_sessions[sid] = session
        return session

    def get_session(self, session_id: str) -> Optional[VoiceSession]:
        return self._active_sessions.get(session_id)

    async def end_session(self, session_id: str) -> Optional[dict]:
        session = self._active_sessions.pop(session_id, None)
        if session:
            await session.close()
            metrics_dict = session.metrics.to_dict()
            self._session_history.append(metrics_dict)
            if len(self._session_history) > 1000:
                self._session_history.pop(0)
            return metrics_dict
        return None

    def list_active_sessions(self) -> List[dict]:
        return [sess.metrics.to_dict() for sess in self._active_sessions.values()]

    def list_recent_history(self, limit: int = 50) -> List[dict]:
        return self._session_history[-limit:]


# Global default session manager placeholder (bound when pool is initialized)
default_session_manager: Optional[SessionManager] = None
