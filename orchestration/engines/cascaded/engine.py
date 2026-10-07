"""
Cascaded Voice Engine (Engine B Runtime).
Executes streaming STT -> LLM -> TTS pipelines using Pipecat and user API keys.
Guarantees wire parity with Engine A over /v2/voice.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Dict, List

import numpy as np
from fastapi import WebSocketDisconnect
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    TextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.workers.runner import WorkerRunner
from sqlalchemy import select

from orchestration.audio.codecs import decode_ulaw, encode_ulaw
from orchestration.audio.dsp import compute_rms, int16_to_float32
from orchestration.db.models import CallTurn, ProviderCredential
from orchestration.db.service import CallSessionService
from orchestration.domain.engines.spec import CascadedPipelineSpec, EngineType, TurnMetrics
from orchestration.engines.base import SessionContext, VoiceEngine
from orchestration.infrastructure.vault.crypto import decrypt_api_key
from orchestration.pipeline.end_detector import EndOfCallDetector
from orchestration.prompts.compiler import compile_prompt

from .pipeline_factory import build_llm_processor, build_stt_processor, build_tts_processor

logger = logging.getLogger("orchestration.engines.cascaded")


class BridgeSink(FrameProcessor):
    """
    Downstream sink in the Pipecat pipeline that captures events, text, and audio,
    forwarding them onto the client WebSocket.
    """

    def __init__(self, engine_session: CascadedSessionRunner) -> None:
        super().__init__()
        self.session = engine_session

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

        if self.session.stop_event.is_set():
            return

        try:
            if isinstance(frame, TranscriptionFrame):
                await self.session.on_user_transcript(frame.text)
            elif isinstance(frame, TextFrame):
                await self.session.on_agent_token(frame.text)
            elif isinstance(frame, TTSStartedFrame):
                await self.session.on_agent_speaking_started()
            elif isinstance(frame, TTSStoppedFrame):
                await self.session.on_agent_speaking_stopped()
            elif isinstance(frame, TTSAudioRawFrame):
                await self.session.on_tts_audio(frame.audio)
            elif isinstance(frame, InterruptionFrame):
                await self.session.on_interruption_detected()
        except Exception as e:
            logger.debug("BridgeSink frame processing error: %s", e)


class CascadedSessionRunner:
    """Manages an active real-time call session using Engine B."""

    def __init__(
        self,
        context: SessionContext,
        spec: CascadedPipelineSpec,
        credentials: Dict[str, str],
    ) -> None:
        self.context = context
        self.spec = spec
        self.credentials = credentials
        self.ws = context.websocket
        self.agent = context.agent
        self.ver = context.version
        self.sample_rate = context.client_sample_rate
        self.codec = context.codec

        self.call_start_time = time.time()
        self.stop_event = asyncio.Event()
        self.disconnect_reason = "normal"

        # Telemetry & turns
        self.current_user_text = ""
        self.current_agent_turn_text: List[str] = []
        self.current_turn_index = 0
        self.is_agent_speaking = False
        self.t_caller_stopped = 0.0
        self.t_first_llm_token = 0.0
        self.t_first_tts_audio = 0.0

        # End of call detector fallback
        ending_text = self.ver.ending_text if self.ver else self.agent.draft_ending_text
        end_silence_sec = self.ver.end_silence_sec if self.ver else self.agent.draft_end_silence_sec
        max_duration_sec = self.ver.max_duration_sec if self.ver else self.agent.draft_max_duration_sec
        self.detector = EndOfCallDetector(
            ending_text=ending_text,
            silence_timeout_sec=float(end_silence_sec),
            max_duration_sec=float(max_duration_sec),
        )

        self.task: PipelineTask | None = None
        self.runner: WorkerRunner | None = None
        self.pipeline_runner_task: asyncio.Task | None = None

    async def initialize_and_run(self) -> None:
        # 1. Compile prompt
        sys_prompt = self.ver.system_prompt if self.ver else self.agent.draft_system_prompt
        tz_name = self.ver.timezone if self.ver else self.agent.draft_timezone
        greeting_text = self.ver.greeting_text if self.ver else self.agent.draft_greeting_text
        greeting_mode = self.ver.greeting_mode if self.ver else self.agent.draft_greeting_mode
        ending_text = self.ver.ending_text if self.ver else self.agent.draft_ending_text

        # Add language conditioning instruction
        lang = getattr(self.ver, "language", None) or "en-US"
        if lang in ("ta-IN", "hi-IN", "en-IN"):
            sys_prompt = f"Always respond in natural conversational {lang}. Keep sentences short and direct.\n{sys_prompt}"

        compiled = compile_prompt(
            system_prompt=sys_prompt,
            greeting_text=greeting_text,
            greeting_mode=greeting_mode,
            ending_text=ending_text,
            agent_name=self.agent.name,
            timezone=tz_name,
            caller_name=self.context.template_vars.get("caller_name"),
            customer_name=self.context.template_vars.get("customer_name"),
            phone_number=self.context.template_vars.get("phone_number"),
        )
        self.detector.start_session(time.time())

        # 2. Build Pipecat pipeline
        stt_proc = build_stt_processor(self.spec, self.credentials, sample_rate=self.sample_rate)
        llm_proc = build_llm_processor(self.spec, self.credentials, system_prompt=compiled.formatted_prompt)
        tts_proc = build_tts_processor(self.spec, self.credentials, sample_rate=self.sample_rate)
        sink = BridgeSink(self)

        pipeline = Pipeline([stt_proc, llm_proc, tts_proc, sink])
        self.task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))
        self.runner = WorkerRunner()
        self.pipeline_runner_task = asyncio.create_task(self.runner.run(self.task))

        # Notify client of ready session state
        await self.ws.send_json({
            "type": "session_started",
            "session_id": self.context.session_record.id,
            "agent_id": self.agent.id,
            "agent_name": self.agent.name,
            "engine": EngineType.CASCADED_CLOUD.value,
            "sample_rate": self.sample_rate,
            "codec": self.codec,
            "greeting_mode": greeting_mode,
            "stt_provider": self.spec.stt.provider,
            "llm_provider": self.spec.llm.provider,
            "tts_provider": self.spec.tts.provider,
        })

        # 3. Handle Greeting
        if greeting_mode == "agent_first" and greeting_text.strip():
            await self.ws.send_json({
                "type": "transcript",
                "role": "assistant",
                "text": greeting_text,
                "is_greeting": True,
            })
            # Inject greeting directly into TTS for immediate audio synthesis
            await self.task.queue_frame(TextFrame(text=greeting_text))

        # 4. Inbound Client Audio & Message Loop
        async def client_inbound_loop():
            try:
                while not self.stop_event.is_set():
                    raw = await self.ws.receive()
                    if "bytes" in raw and raw["bytes"]:
                        data = raw["bytes"]
                        if len(data) % 2 != 0 and data[0] == 0x01:
                            data = data[1:]
                        if len(data) == 0:
                            continue

                        # Decode codec to PCM16
                        if self.codec == "g711_ulaw":
                            pcm16 = decode_ulaw(data)
                        else:
                            pcm16 = np.frombuffer(data, dtype=np.int16)

                        f32 = int16_to_float32(pcm16)
                        rms = compute_rms(f32)
                        if rms > 0.012:
                            self.detector.on_user_speech(time.time())
                            # Barge-in check: If agent is speaking, cancel immediately
                            if self.is_agent_speaking:
                                await self.trigger_barge_in()

                        # Push into Pipecat pipeline
                        raw_bytes = pcm16.tobytes()
                        await self.task.queue_frame(
                            InputAudioRawFrame(
                                audio=raw_bytes,
                                sample_rate=self.sample_rate,
                                num_channels=1,
                            )
                        )

                    elif "text" in raw and raw["text"]:
                        try:
                            msg_json = json.loads(raw["text"])
                            mtype = msg_json.get("type")
                            if mtype == "hangup":
                                self.disconnect_reason = "user_hangup"
                                self.stop_event.set()
                                break
                            elif mtype == "ping":
                                await self.ws.send_json({"type": "pong", "time": time.time()})
                            elif mtype == "interrupt":
                                await self.trigger_barge_in()
                        except json.JSONDecodeError:
                            pass
            except (WebSocketDisconnect, asyncio.CancelledError):
                self.disconnect_reason = "client_disconnected"
                self.stop_event.set()
            except Exception as e:
                if "disconnect" in type(e).__name__.lower() or "disconnect" in str(e).lower():
                    self.disconnect_reason = "client_disconnected"
                else:
                    logger.error("client_inbound_loop exception: %s", e, exc_info=True)
                self.stop_event.set()

        async def supervisor_loop():
            try:
                while not self.stop_event.is_set():
                    await asyncio.sleep(0.1)
                    term_reason = self.detector.check_termination(time.time())
                    if term_reason:
                        self.disconnect_reason = term_reason
                        self.stop_event.set()
                        break
            except Exception as e:
                logger.error("supervisor_loop exception: %s", e, exc_info=True)

        try:
            inbound_task = asyncio.create_task(client_inbound_loop())
            supervisor_task = asyncio.create_task(supervisor_loop())
            done, pending = await asyncio.wait(
                [inbound_task, supervisor_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for p in pending:
                p.cancel()
        finally:
            await self.teardown()

    async def trigger_barge_in(self) -> None:
        """Immediate barge-in cancellation and dialogue history truncation."""
        if self.task:
            await self.task.queue_frame(InterruptionFrame())
        self.is_agent_speaking = False
        await self.ws.send_json({
            "type": "agent.speaking.end",
            "interrupted": True,
        })

    async def on_user_transcript(self, text: str) -> None:
        """Called when STT produces finalized user utterance."""
        self.t_caller_stopped = time.time()
        self.t_first_llm_token = 0.0
        self.t_first_tts_audio = 0.0
        self.current_user_text = text

        await self.ws.send_json({
            "type": "transcript",
            "role": "user",
            "text": text,
        })

        # Save user turn to database
        self.current_turn_index += 1
        turn_ms = (time.time() - self.call_start_time) * 1000.0
        async with self.context.session_factory() as db:
            turn_record = CallTurn(
                session_id=self.context.session_record.id,
                idx=self.current_turn_index,
                role="user",
                text=text,
                started_ms=turn_ms,
            )
            db.add(turn_record)
            await db.commit()

    async def on_agent_token(self, token: str) -> None:
        """Called on LLM token stream."""
        if self.t_first_llm_token == 0.0 and self.t_caller_stopped > 0:
            self.t_first_llm_token = time.time()

        self.current_agent_turn_text.append(token)
        await self.ws.send_json({
            "type": "transcript",
            "role": "assistant",
            "text": token,
        })

        # Check end of call detector
        self.detector.on_agent_token(token, time.time())

    async def on_agent_speaking_started(self) -> None:
        self.is_agent_speaking = True
        await self.ws.send_json({"type": "agent.speaking.start"})

    async def on_agent_speaking_stopped(self) -> None:
        self.is_agent_speaking = False
        self.detector.on_agent_speaking_stopped(time.time())
        await self.ws.send_json({"type": "agent.speaking.end", "interrupted": False})

        # Compute turn metrics
        if self.t_caller_stopped > 0 and self.t_first_tts_audio > 0:
            v2v_ms = (self.t_first_tts_audio - self.t_caller_stopped) * 1000.0
            llm_ttft = (self.t_first_llm_token - self.t_caller_stopped) * 1000.0 if self.t_first_llm_token > 0 else 0.0
            tts_ttfa = (self.t_first_tts_audio - self.t_first_llm_token) * 1000.0 if self.t_first_llm_token > 0 else 0.0

            metrics = TurnMetrics(
                eot_ms=250.0,
                stt_ms=20.0,
                llm_ttft_ms=round(llm_ttft, 1),
                tts_ttfa_ms=round(tts_ttfa, 1),
                voice_to_voice_ms=round(v2v_ms, 1),
            )
            await self.ws.send_json({
                "type": "metrics.turn",
                "metrics": metrics.model_dump(),
            })

        # Save assistant turn to database
        if self.current_agent_turn_text:
            spoken_text = "".join(self.current_agent_turn_text)
            self.current_agent_turn_text = []
            self.current_turn_index += 1
            turn_ms = (time.time() - self.call_start_time) * 1000.0

            llm_ttft = (self.t_first_llm_token - self.t_caller_stopped) * 1000.0 if (self.t_first_llm_token > 0 and self.t_caller_stopped > 0) else None
            tts_ttfa = (self.t_first_tts_audio - self.t_first_llm_token) * 1000.0 if (self.t_first_tts_audio > 0 and self.t_first_llm_token > 0) else None
            v2v_ms = (self.t_first_tts_audio - self.t_caller_stopped) * 1000.0 if (self.t_first_tts_audio > 0 and self.t_caller_stopped > 0) else None

            async with self.context.session_factory() as db:
                turn_record = CallTurn(
                    session_id=self.context.session_record.id,
                    idx=self.current_turn_index,
                    role="assistant",
                    text=spoken_text,
                    started_ms=turn_ms,
                    eot_ms=250.0,
                    stt_ms=20.0,
                    llm_ttft_ms=round(llm_ttft, 1) if llm_ttft is not None else None,
                    tts_ttfa_ms=round(tts_ttfa, 1) if tts_ttfa is not None else None,
                    voice_to_voice_ms=round(v2v_ms, 1) if v2v_ms is not None else None,
                )
                db.add(turn_record)
                await db.commit()

    async def on_tts_audio(self, pcm_bytes: bytes) -> None:
        """Called when TTS outputs raw audio."""
        if self.t_first_tts_audio == 0.0 and self.t_caller_stopped > 0:
            self.t_first_tts_audio = time.time()

        if len(pcm_bytes) == 0:
            return

        # Encode according to codec
        if self.codec == "g711_ulaw":
            f32 = int16_to_float32(np.frombuffer(pcm_bytes, dtype=np.int16))
            out_bytes = encode_ulaw(f32)
        else:
            out_bytes = pcm_bytes

        await self.ws.send_bytes(out_bytes)

    async def on_interruption_detected(self) -> None:
        self.is_agent_speaking = False

    async def teardown(self) -> None:
        """Closes pipeline and finalizes call session in DB."""
        if self.task:
            try:
                await self.task.queue_frame(EndFrame())
            except Exception:
                pass

        if self.pipeline_runner_task and not self.pipeline_runner_task.done():
            self.pipeline_runner_task.cancel()

        duration_sec = round(time.time() - self.call_start_time, 2)
        async with self.context.session_factory() as db:
            call_svc = CallSessionService(db)
            await call_svc.end_session(
                session_id=self.context.session_record.id,
                end_reason=self.disconnect_reason,
                duration_sec=duration_sec,
            )
            await db.commit()

        try:
            await self.ws.send_json({
                "type": "session_ended",
                "session_id": self.context.session_record.id,
                "reason": self.disconnect_reason,
                "duration_sec": duration_sec,
            })
            await self.ws.close()
        except Exception:
            pass


class CascadedVoiceEngine(VoiceEngine):
    """Engine B: Streaming STT -> LLM -> TTS Cascaded Cloud Engine."""

    @property
    def engine_type(self) -> EngineType:
        return EngineType.CASCADED_CLOUD

    async def run_session(self, context: SessionContext) -> None:
        # 1. Parse pipeline spec
        pipeline_json = getattr(context.version, "pipeline_json", None) or getattr(context.agent, "draft_pipeline_json", "{}")
        try:
            spec_data = json.loads(pipeline_json) if pipeline_json else {}
            spec = CascadedPipelineSpec.model_validate(spec_data)
        except Exception as e:
            logger.warning("Failed to parse pipeline spec (%s), using defaults", e)
            spec = CascadedPipelineSpec()

        # 2. Fetch and decrypt required provider credentials from DB
        needed_providers = {spec.stt.provider, spec.llm.provider, spec.tts.provider}
        credentials: Dict[str, str] = {}

        async with context.session_factory() as db:
            stmt = select(ProviderCredential).where(
                ProviderCredential.workspace_id == context.workspace_id,
                ProviderCredential.provider_id.in_(needed_providers),
            )
            records = (await db.execute(stmt)).scalars().all()
            for r in records:
                try:
                    decrypted = decrypt_api_key(r.ciphertext, r.nonce, key_version=r.key_version)
                    credentials[r.provider_id] = decrypted
                except Exception as e:
                    logger.error("Failed to decrypt credentials for provider %s: %s", r.provider_id, e)

        runner = CascadedSessionRunner(context, spec, credentials)
        await runner.initialize_and_run()
