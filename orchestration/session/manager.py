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
import logging
import time
import uuid
from enum import StrEnum
from typing import Any

import numpy as np

from ..persona.registry import PersonaConfig
from ..protocol.audio import (
    FRAME_SIZE,
    SAMPLE_RATE,
    AdaptiveNoiseCanceller,
    AudioFrameBuffer,
    compute_rms,
)
from ..protocol.messages import (
    AudioMessage,
    ControlAction,
    ControlMessage,
    ErrorMessage,
    HandshakeMessage,
    MetadataMessage,
    TextMessage,
    WSMessage,
    decode_message,
    encode_message,
)
from ..worker.client import PersonaPlexWorkerClient
from ..worker.pool import WorkerPool

logger = logging.getLogger(__name__)


class SessionState(StrEnum):
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
        self.connected_at: float | None = None
        self.ended_at: float | None = None

        # 1. Listening
        self.user_frames_in = 0
        self.dropped_frames = 0
        self.vad_speech_frames = 0

        # 2. Understanding
        self.text_tokens_out = 0
        self.transcript_tokens: list[str] = []

        # 3. Reasoning & Goals
        self.business_goal: dict[str, Any] | None = None
        self.reasoning_notes: list[str] = []

        # 4. Speaking
        self.agent_frames_out = 0
        self.clipping_events = 0
        self.underrun_events = 0

        # 5. Latency
        self.ttfa_history_ms: list[float] = []
        self.frame_step_times_ms: list[float] = []

        # 6. Conversation Dynamics
        self.barge_in_events = 0
        self.backchannel_events = 0
        self.turn_count = 0
        self.silence_frames = 0
        self.handshake_time_sec: float = 0.0
        self.claimed_from_standby: bool = False

        # 7. Task Success & Outcome Tagging
        self.outcome_tag: str = "in_progress"  # in_progress, success, resolved, escalated, dropped
        self.outcome_metadata: dict[str, Any] = {}

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

    def tag_outcome(self, outcome: str, metadata: dict[str, Any] | None = None) -> None:
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
            "handshake_time_sec": self.handshake_time_sec,
            "claimed_from_standby": self.claimed_from_standby,
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
        barge_in_warmup_sec: float = 0.0,
    ):
        self.session_id = session_id
        self.persona = persona
        self.worker = worker
        self.pool = pool
        self.barge_in_rms_threshold = barge_in_rms_threshold
        self.barge_in_warmup_sec = barge_in_warmup_sec

        self.metrics = SessionMetrics(session_id, persona.id, worker.worker_id)
        self.inbound_buffer = AudioFrameBuffer(dtype=np.float32)
        self.noise_canceller = AdaptiveNoiseCanceller()
        from ..audio.cleaner import CallerAudioCleaner
        from ..audio.caller_transcriber import CallerTranscriber
        from ..audio.turn_detector import TurnDetector

        self.cleaner = CallerAudioCleaner(sample_rate=SAMPLE_RATE, frame_size=FRAME_SIZE)
        self._send_to_client_fn = None
        self.turn_detector = TurnDetector(
            sample_rate=SAMPLE_RATE,
            frame_size=FRAME_SIZE,
            base_silence_sec=0.60,
            speech_rms_threshold=self.barge_in_rms_threshold,
        )
        self.transcriber = CallerTranscriber(
            sample_rate=SAMPLE_RATE,
            on_transcript_callback=self._broadcast_transcript,
        )

        self._state = SessionState.INITIALIZING
        self._stop_event = asyncio.Event()
        self._feeder_task: asyncio.Task | None = None
        self._user_speaking = False
        self._last_user_speech_time = 0.0
        self._speech_frames_count = 0
        self.is_standby: bool = False
        self.is_claimed_from_standby: bool = False

        # Telemetry flags
        self._first_input_frame_sent = False
        self._first_output_frame_received = False
        self._first_text_token_received = False
        self._last_telemetry_log_time = time.time()

    async def _broadcast_transcript(self, payload: dict[str, Any]) -> None:
        if self._send_to_client_fn:
            try:
                await self._send_to_client_fn(encode_message(MetadataMessage(data=payload)))
            except Exception as e:
                logger.debug(f"Failed to forward transcript metadata: {e}")

    @property
    def state(self) -> SessionState:
        return self._state

    def set_state(self, new_state: SessionState) -> None:
        self._state = new_state
        self.metrics.state = new_state
        logger.debug(f"Session {self.session_id} state -> {new_state.value}")

    async def start(self) -> None:
        """Connect to worker, await handshake (priming phase), clear buffer, and start feeder."""
        self.transcriber.start()
        if self.is_claimed_from_standby:
            logger.info(f"Session {self.session_id} was claimed from primed standby. Instant start (TTFA < 1.5s)!")
            self.set_state(SessionState.ACTIVE)
            self.metrics.connected_at = time.time()
            return

        self.set_state(SessionState.PROMPTING)
        t_start = time.time()
        try:
            await self.worker.connect(self.session_id, self.persona)
            t_handshake = time.time() - t_start
            self.metrics.handshake_time_sec = round(t_handshake, 3)
            logger.info(f"Session {self.session_id} worker handshake complete in {t_handshake:.2f}s")
            
            # Drop any audio frames accumulated during the priming wait (prevents burst to model)
            self.inbound_buffer.clear()

            self.set_state(SessionState.ACTIVE)
            self.metrics.connected_at = time.time()

            # Start dedicated continuous 12.5 Hz (80 ms) feeder loop to upstream worker
            self._feeder_task = asyncio.create_task(self._continuous_worker_feeder())
        except Exception as e:
            self.set_state(SessionState.FAILED)
            try:
                await self.pool.release_worker(self.worker.worker_id)
            except Exception as rel_err:
                logger.warning(f"Error releasing worker after start failure: {rel_err}")
            raise e

    async def _continuous_worker_feeder(self) -> None:
        """
        Continuously stream 80ms frames (1,920 samples @ 24kHz) to the worker at 12.5 Hz cadence.
        If user audio frames are queued, pops and sends them.
        If user buffer is empty, sends silence frames to drive Moshi's autoregressive generation loop.
        """
        logger.info(f"Session {self.session_id}: Continuous worker feeder started at 12.5 Hz cadence")
        silence_frame = np.zeros(FRAME_SIZE, dtype=np.float32)

        try:
            while not self._stop_event.is_set():
                t_frame_start = time.time()

                # Pop user frame if available; otherwise supply silence frame
                frame = self.inbound_buffer.pop_frame()
                if frame is None:
                    frame = silence_frame
                    self.metrics.silence_frames += 1

                # Send frame to worker
                try:
                    await self.worker.send_audio(frame)
                    if not self._first_input_frame_sent:
                        self._first_input_frame_sent = True
                        elapsed = time.time() - (self.metrics.connected_at or self.metrics.created_at)
                        logger.info(f"Session {self.session_id}: First input audio frame sent to worker at {elapsed:.2f}s")
                except Exception as ex:
                    logger.warning(f"Session {self.session_id} worker feeder send error: {ex}")
                    break

                # Periodic telemetry logging every 5 seconds
                now = time.time()
                if now - self._last_telemetry_log_time >= 5.0:
                    self._last_telemetry_log_time = now
                    logger.info(
                        f"Session {self.session_id} [5s Telemetry] "
                        f"Frames In: {self.metrics.user_frames_in} | "
                        f"Silence Frames: {self.metrics.silence_frames} | "
                        f"Frames Out: {self.metrics.agent_frames_out} | "
                        f"Tokens Out: {self.metrics.text_tokens_out} | "
                        f"Barge-ins: {self.metrics.barge_in_events} | "
                        f"State: {self.state.value}"
                    )

                # Maintain strict 80ms (12.5 Hz) frame interval
                elapsed = time.time() - t_frame_start
                sleep_dur = max(0.0, 0.080 - elapsed)
                await asyncio.sleep(sleep_dur)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in session {self.session_id} continuous feeder: {e}")

    async def ingest_client_message(self, raw_bytes: bytes) -> WSMessage | None:
        """
        Process an incoming raw binary message from client:
        - If audio (0x01): buffer, slice into 1920-sample frames, check barge-in, queue in inbound_buffer.
        - If text (0x02): forward to worker.
        - If control (0x03): forward to worker.
        """
        try:
            msg = decode_message(raw_bytes)
        except Exception as e:
            return ErrorMessage(error=f"Malformed frame: {e!s}")

        if self._state not in (SessionState.ACTIVE, SessionState.INTERRUPTED):
            if isinstance(msg, AudioMessage):
                return ErrorMessage(error="Handshake not completed. Audio not accepted before session is active.")
            elif isinstance(msg, HandshakeMessage):
                return None
            return None

        if isinstance(msg, AudioMessage):
            audio_data = msg.data
            prev_frames = self.inbound_buffer.available_frames
            self.inbound_buffer.push_pcm_bytes(audio_data, is_int16=False)
            curr_frames = self.inbound_buffer.available_frames
            new_frames = max(0, curr_frames - prev_frames)

            # Cap queued frames to keep live latency under 250ms (drop excess older frames)
            while self.inbound_buffer.available_frames > 4:
                self.inbound_buffer.pop_frame()
                self.metrics.dropped_frames += 1

            if new_frames > 0:
                self.metrics.user_frames_in += new_frames
                avail = self.inbound_buffer.available_frames
                start_idx = max(0, avail - new_frames)
                now = time.time()
                session_age = now - (self.metrics.connected_at or self.metrics.created_at)

                for idx in range(start_idx, avail):
                    frame_data = self.inbound_buffer.peek_frame(idx)
                    if frame_data is None:
                        continue

                    # Record to cleaner for A/B comparison and optional CPU denoise
                    self.cleaner.process_chunk(frame_data)

                    rms = compute_rms(frame_data)
                    is_agent_spk = (self.metrics.agent_frames_out > 0 and (now - self.transcriber._last_agent_audio_time < 0.35))
                    spk_now, turn_done, completed_audio = self.turn_detector.push_frame(
                        frame_data,
                        rms=rms,
                        is_agent_speaking=is_agent_spk,
                    )

                    # Guard: require warmup delay (if configured) before enabling barge-in
                    if session_age >= self.barge_in_warmup_sec and rms > self.barge_in_rms_threshold:
                        if not self._user_speaking:
                            self._user_speaking = True
                            self.metrics.barge_in_events += 1
                            logger.info(
                                f"Session {self.session_id} barge-in triggered (RMS={rms:.4f} > {self.barge_in_rms_threshold:.4f}, age={session_age:.2f}s)"
                            )
                            self.set_state(SessionState.INTERRUPTED)
                            asyncio.create_task(self.transcriber.finalize_agent_turn(reason="caller_barge_in"))
                            try:
                                await self.worker.send_control(ControlAction.PAUSE)
                            except Exception:
                                pass
                        self._last_user_speech_time = now
                    elif self._user_speaking and (now - self._last_user_speech_time > 0.6):
                        self._user_speaking = False
                        self.set_state(SessionState.ACTIVE)

                    # Caller speech turn finalized: submit for lightweight CPU transcription
                    if turn_done and completed_audio is not None and len(completed_audio) > 0:
                        self.transcriber.enqueue_caller_speech(completed_audio)

        elif isinstance(msg, TextMessage):
            await self.worker.send_text(msg.text)
        elif isinstance(msg, ControlMessage):
            await self.worker.send_control(msg.action)

        return msg

    async def run_worker_forwarder(self, send_to_client_fn) -> None:
        """
        Receive generated frames and text tokens from worker and forward to client.
        Segments agent text into separate turns based on audio silence and user onset.
        """
        self._send_to_client_fn = send_to_client_fn

        # Background silence watchdog to finalize agent turns after speech ends (>600ms)
        async def agent_silence_watchdog():
            while not self._stop_event.is_set():
                try:
                    await asyncio.sleep(0.15)
                    await self.transcriber.check_agent_silence_timeout(silence_threshold_sec=0.60)
                except asyncio.CancelledError:
                    break
                except Exception:
                    pass

        watchdog_task = asyncio.create_task(agent_silence_watchdog())

        try:
            async for worker_msg in self.worker.recv_messages():
                if self._stop_event.is_set():
                    break

                if isinstance(worker_msg, AudioMessage):
                    self.metrics.agent_frames_out += 1
                    self.transcriber.on_agent_audio_frame()
                    if not self._first_output_frame_received:
                        self._first_output_frame_received = True
                        elapsed = time.time() - (self.metrics.connected_at or self.metrics.created_at)
                        logger.info(f"Session {self.session_id}: First agent audio frame received at {elapsed:.2f}s")

                    # Barge-in: immediately suppress forwarding agent audio while user is interrupting
                    if self._state == SessionState.INTERRUPTED:
                        continue
                    # Forward agent audio 0x01 to client
                    await send_to_client_fn(encode_message(worker_msg))

                elif isinstance(worker_msg, TextMessage):
                    self.metrics.text_tokens_out += 1
                    self.transcriber.on_agent_token(worker_msg.text)
                    if not self._first_text_token_received:
                        self._first_text_token_received = True
                        elapsed = time.time() - (self.metrics.connected_at or self.metrics.created_at)
                        logger.info(f"Session {self.session_id}: First text token received at {elapsed:.2f}s: {worker_msg.text!r}")
                    self.metrics.transcript_tokens.append(worker_msg.text)
                    # Forward live partial text token 0x02 to client
                    await send_to_client_fn(encode_message(worker_msg))

                elif isinstance(worker_msg, MetadataMessage):
                    await send_to_client_fn(encode_message(worker_msg))

                elif isinstance(worker_msg, ErrorMessage):
                    logger.warning(f"Session {self.session_id} received worker error: {worker_msg.error}")
                    await send_to_client_fn(encode_message(worker_msg))
                    break

        except Exception as e:
            logger.error(f"Error in session {self.session_id} worker forwarder: {e}")
            self.set_state(SessionState.FAILED)
        finally:
            watchdog_task.cancel()
            self._stop_event.set()

    async def close(self) -> None:
        """Cleanly close session, update metrics, stop feeder and transcriber, and release worker."""
        if self._state in (SessionState.COMPLETED, SessionState.CLOSING):
            return

        self.set_state(SessionState.CLOSING)
        self._stop_event.set()
        self.metrics.ended_at = time.time()

        # Stop transcriber and finalize any pending agent turn
        await self.transcriber.finalize_agent_turn(reason="session_close")
        await self.transcriber.stop()

        if self._feeder_task and not self._feeder_task.done():
            self._feeder_task.cancel()
            try:
                await self._feeder_task
            except (asyncio.CancelledError, Exception):
                pass

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
    """Manages active and historical sessions, including pre-primed standby sessions."""

    def __init__(self, pool: WorkerPool, enable_standby: bool = False):
        self.pool = pool
        self.enable_standby = enable_standby
        self._active_sessions: dict[str, VoiceSession] = {}
        self._session_history: list[dict] = []
        self._standby_session: VoiceSession | None = None
        self._standby_lock = asyncio.Lock()
        self._is_prepriming = False

    async def preprime_standby(self, persona: PersonaConfig | None = None) -> VoiceSession | None:
        """
        Pre-connects and primes a standby session in the background.
        Maintains real-time 80ms silence frames so worker is primed and ready
        for instant attachment by incoming calls (reducing TTFA from ~12.5s to < 1.5s).
        """
        self.enable_standby = True
        if self._is_prepriming:
            return None

        async with self._standby_lock:
            # Check if existing standby session is still alive and healthy
            if self._standby_session is not None:
                if self._standby_session.state == SessionState.ACTIVE and self._standby_session.worker.status == WorkerStatus.BUSY:
                    if persona is None or self._standby_session.persona.get_normalized_voice_prompt() == persona.get_normalized_voice_prompt():
                        return self._standby_session
                # Stale or mismatched standby: tear down cleanly
                old = self._standby_session
                self._standby_session = None
                await old.close()

            # Verify pool has idle capacity
            pool_stats = self.pool.get_stats()
            if pool_stats["idle_workers"] <= 0:
                logger.debug("No idle workers available to establish standby session")
                return None

            from ..persona.registry import default_registry
            target_persona = persona or default_registry.get("casual_friend") or list(default_registry.list_all())[0]

            self._is_prepriming = True
            standby_sid = f"standby_{uuid.uuid4().hex[:8]}"
            try:
                logger.info(f"[STANDBY] Pre-priming standby session {standby_sid} with voice {target_persona.get_normalized_voice_prompt()}...")
                session = await self.create_session(
                    persona=target_persona,
                    session_id=standby_sid,
                    timeout=10.0,
                    use_standby=False,
                    is_standby=True,
                )
                session.is_standby = True
                await session.start()
                self._standby_session = session
                logger.info(f"[STANDBY] Standby session {standby_sid} primed and running! Instant TTFA ready.")
                return session
            except Exception as e:
                logger.warning(f"[STANDBY] Failed to pre-prime standby session: {e}")
                self._standby_session = None
                return None
            finally:
                self._is_prepriming = False

    async def create_session(
        self,
        persona: PersonaConfig,
        session_id: str | None = None,
        timeout: float = 30.0,
        barge_in_warmup_sec: float = 0.0,
        use_standby: bool = True,
        is_standby: bool = False,
    ) -> VoiceSession:
        """Acquire a worker and initialize a new VoiceSession, attaching to standby if available."""
        sid = session_id or f"sess_{uuid.uuid4().hex[:12]}"

        # Instant Standby Claiming
        if use_standby and not is_standby:
            async with self._standby_lock:
                if self._standby_session is not None and self._standby_session.state == SessionState.ACTIVE:
                    standby_voice = self._standby_session.persona.get_normalized_voice_prompt()
                    req_voice = persona.get_normalized_voice_prompt()
                    if standby_voice == req_voice:
                        session = self._standby_session
                        self._standby_session = None
                        session.session_id = sid
                        session.metrics.session_id = sid
                        session.metrics.persona_id = persona.id
                        session.metrics.claimed_from_standby = True
                        session.is_standby = False
                        session.is_claimed_from_standby = True
                        session.persona = persona
                        self._active_sessions[sid] = session
                        logger.info(f"[STANDBY] Claimed pre-primed session for {sid}! TTFA will be < 1.5s.")

                        # Automatically trigger background replacement priming
                        asyncio.create_task(self.preprime_standby(persona))
                        return session

        worker = await self.pool.acquire_worker(session_id=sid, timeout=timeout)
        session = VoiceSession(
            session_id=sid,
            persona=persona,
            worker=worker,
            pool=self.pool,
            barge_in_warmup_sec=barge_in_warmup_sec,
        )
        session.is_standby = is_standby
        self._active_sessions[sid] = session
        return session

    def get_session(self, session_id: str) -> VoiceSession | None:
        return self._active_sessions.get(session_id)

    async def end_session(self, session_id: str) -> dict | None:
        session = self._active_sessions.pop(session_id, None)
        if session:
            await session.close()
            metrics_dict = session.metrics.to_dict()
            self._session_history.append(metrics_dict)
            if len(self._session_history) > 1000:
                self._session_history.pop(0)

            # Auto-prime replacement standby session in background if enabled
            if self.enable_standby:
                asyncio.create_task(self.preprime_standby())
            return metrics_dict
        return None

    def has_standby(self) -> bool:
        """Returns True if a pre-primed standby session is ready for instant connection."""
        return (
            self._standby_session is not None
            and self._standby_session.state == SessionState.ACTIVE
            and not self._standby_session.is_claimed_from_standby
        )

    def list_active_sessions(self) -> list[dict]:
        return [sess.metrics.to_dict() for sess in self._active_sessions.values()]

    def list_recent_history(self, limit: int = 50) -> list[dict]:
        return self._session_history[-limit:]


# Global default session manager placeholder (bound when pool is initialized)
default_session_manager: SessionManager | None = None
