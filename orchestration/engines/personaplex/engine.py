"""
PersonaPlex S2S Voice Engine (Engine A Adapter).
Wraps existing native GPU/Mock worker pool and Mimi 24kHz audio pipeline.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

import numpy as np
from fastapi import WebSocketDisconnect, status

from orchestration.audio.codecs import decode_ulaw, encode_ulaw
from orchestration.audio.dsp import compute_rms, float32_to_int16, int16_to_float32, soft_clip
from orchestration.audio.resample import AudioResampler, StreamingResampleBuffer
from orchestration.db.service import CallSessionService
from orchestration.domain.engines.spec import EngineType
from orchestration.engines.base import SessionContext, VoiceEngine
from orchestration.persona.registry import PersonaConfig
from orchestration.pipeline.end_detector import EndOfCallDetector
from orchestration.prompts.compiler import compile_prompt
from orchestration.protocol.audio import MODEL_SAMPLE_RATE
from orchestration.protocol.messages import AudioMessage, ControlAction, TextMessage
from orchestration.worker.client import PersonaPlexWorkerClient
from orchestration.worker.pool import WorkerPool

logger = logging.getLogger("orchestration.engines.personaplex")


class PersonaPlexVoiceEngine(VoiceEngine):
    """Engine A: Native PersonaPlex Speech-to-Speech foundation model."""

    def __init__(self, pool: WorkerPool | None = None) -> None:
        self.pool = pool

    @property
    def engine_type(self) -> EngineType:
        return EngineType.PERSONAPLEX_S2S

    async def run_session(self, context: SessionContext) -> None:
        ws = context.websocket
        agent = context.agent
        target_ver = context.version
        session_factory = context.session_factory
        call_session = context.session_record
        sample_rate = context.client_sample_rate
        codec = context.codec

        worker_pool: WorkerPool = self.pool or getattr(ws.app.state, "pool", None)
        if not worker_pool:
            await ws.send_json({"type": "error", "message": "Worker pool not available on gateway"})
            await ws.close(code=status.WS_1011_INTERNAL_ERROR)
            return

        # 1. Configuration values
        voice_id = target_ver.voice_id if target_ver else agent.draft_voice_id
        greeting_text = target_ver.greeting_text if target_ver else agent.draft_greeting_text
        greeting_mode = target_ver.greeting_mode if target_ver else agent.draft_greeting_mode
        system_prompt = target_ver.system_prompt if target_ver else agent.draft_system_prompt
        ending_text = target_ver.ending_text if target_ver else agent.draft_ending_text
        end_silence_sec = target_ver.end_silence_sec if target_ver else agent.draft_end_silence_sec
        max_duration_sec = target_ver.max_duration_sec if target_ver else agent.draft_max_duration_sec
        tz_name = target_ver.timezone if target_ver else agent.draft_timezone

        # 2. Compile prompt with dynamic local time
        compiled = compile_prompt(
            system_prompt=system_prompt,
            greeting_text=greeting_text,
            greeting_mode=greeting_mode,
            ending_text=ending_text,
            agent_name=agent.name,
            timezone=tz_name,
            caller_name=context.template_vars.get("caller_name"),
            customer_name=context.template_vars.get("customer_name"),
            phone_number=context.template_vars.get("phone_number"),
        )

        detector = EndOfCallDetector(
            ending_text=ending_text,
            silence_timeout_sec=float(end_silence_sec),
            max_duration_sec=float(max_duration_sec),
        )

        call_start_time = time.time()
        in_buffer = StreamingResampleBuffer(
            in_rate=sample_rate,
            out_rate=MODEL_SAMPLE_RATE,
            out_frame_samples=480,
            quality="QQ",
        )
        out_resampler = AudioResampler(in_rate=MODEL_SAMPLE_RATE, out_rate=sample_rate, quality="QQ")

        # 3. Acquire worker from pool
        worker_client: PersonaPlexWorkerClient | None = None
        try:
            worker_client = await worker_pool.acquire_worker(session_id=call_session.id, timeout=10.0)
        except Exception as e:
            logger.error(f"Failed to acquire worker for session {call_session.id}: {e}")
            await ws.send_json({"type": "error", "message": "All inference workers are busy. Please retry."})
            await ws.close(code=status.WS_1013_TRY_AGAIN_LATER)
            return

        persona_cfg = PersonaConfig(
            id=agent.id,
            name=agent.name,
            voice_prompt=voice_id,
            text_prompt=compiled.formatted_prompt,
            system_prompt=compiled.formatted_prompt,
        )

        try:
            await worker_client.connect(session_id=call_session.id, persona=persona_cfg)
        except Exception as e:
            logger.error(f"Failed to connect worker client: {e}")
            await worker_pool.release_worker(worker_client.worker_id)
            await ws.send_json({"type": "error", "message": f"Worker connection failed: {e}"})
            await ws.close(code=status.WS_1011_INTERNAL_ERROR)
            return

        # Notify client of ready session state
        await ws.send_json({
            "type": "session_started",
            "session_id": call_session.id,
            "agent_id": agent.id,
            "agent_name": agent.name,
            "voice_id": voice_id,
            "timezone": tz_name,
            "sample_rate": sample_rate,
            "codec": codec,
            "token_count": compiled.token_count,
            "greeting_mode": greeting_mode,
            "engine": EngineType.PERSONAPLEX_S2S.value,
        })

        if greeting_mode == "agent_first" and greeting_text.strip():
            await ws.send_json({
                "type": "transcript",
                "role": "assistant",
                "text": greeting_text,
                "is_greeting": True,
            })

        stop_event = asyncio.Event()
        disconnect_reason = "normal"
        turn_counter = 0
        current_agent_turn_text: list[str] = []
        current_agent_turn_start = 0.0

        async def client_to_worker_loop():
            nonlocal disconnect_reason
            try:
                while not stop_event.is_set():
                    raw = await ws.receive()
                    if "bytes" in raw and raw["bytes"]:
                        data = raw["bytes"]
                        if len(data) > 0 and data[0] == 0x01:
                            data = data[1:]
                        if len(data) == 0:
                            continue

                        if codec == "g711_ulaw":
                            pcm16 = decode_ulaw(data)
                        else:
                            pcm16 = np.frombuffer(data, dtype=np.int16)
                        f32_samples = int16_to_float32(pcm16)

                        rms = compute_rms(f32_samples)
                        if rms > 0.012:
                            detector.on_user_speech(time.time())
                            if current_agent_turn_text:
                                await worker_client.send_control(ControlAction.PAUSE)

                        frames_24k = in_buffer.push_chunk(f32_samples)
                        for frame_24k in frames_24k:
                            await worker_client.send_audio(frame_24k)

                    elif "text" in raw and raw["text"]:
                        try:
                            msg_json = json.loads(raw["text"])
                            mtype = msg_json.get("type", "")
                            if mtype == "hangup":
                                disconnect_reason = "user_hangup"
                                stop_event.set()
                                break
                            elif mtype == "ping":
                                await ws.send_json({"type": "pong", "time": time.time()})
                            elif mtype == "interrupt":
                                await worker_client.send_control(ControlAction.PAUSE)
                        except json.JSONDecodeError:
                            pass
            except (WebSocketDisconnect, asyncio.CancelledError):
                disconnect_reason = "client_disconnected"
                stop_event.set()
            except Exception as e:
                logger.debug(f"client_to_worker_loop error: {e}")
                stop_event.set()

        async def worker_to_client_loop():
            nonlocal disconnect_reason, turn_counter, current_agent_turn_text, current_agent_turn_start
            try:
                async for msg in worker_client.recv_messages():
                    if stop_event.is_set():
                        break

                    if isinstance(msg, TextMessage):
                        token = msg.text
                        if not current_agent_turn_text:
                            current_agent_turn_start = (time.time() - call_start_time) * 1000.0
                        current_agent_turn_text.append(token)

                        await ws.send_json({
                            "type": "transcript",
                            "role": "assistant",
                            "text": token,
                        })

                        matched = detector.on_agent_token(token, time.time())
                        if matched:
                            detector.on_agent_speaking_stopped(time.time())

                    elif isinstance(msg, AudioMessage):
                        raw_audio = msg.data
                        if len(raw_audio) == 0:
                            continue

                        resampled = out_resampler.resample_chunk(raw_audio, last=False)
                        if len(resampled) == 0:
                            continue

                        clipped_f32 = soft_clip(resampled, threshold=0.92)
                        if codec == "g711_ulaw":
                            pcm_bytes = encode_ulaw(clipped_f32)
                        else:
                            pcm16_out = float32_to_int16(clipped_f32)
                            pcm_bytes = pcm16_out.tobytes()

                        await ws.send_bytes(pcm_bytes)
            except (WebSocketDisconnect, asyncio.CancelledError):
                disconnect_reason = "client_disconnected"
                stop_event.set()
            except Exception as e:
                logger.debug(f"worker_to_client_loop error: {e}")
                stop_event.set()

        async def detector_loop():
            nonlocal disconnect_reason
            while not stop_event.is_set():
                await asyncio.sleep(0.1)
                should_end, reason = detector.should_end_call(time.time())
                if should_end:
                    disconnect_reason = reason
                    stop_event.set()
                    break

        try:
            await asyncio.gather(
                client_to_worker_loop(),
                worker_to_client_loop(),
                detector_loop(),
                return_exceptions=True,
            )
        finally:
            if worker_client:
                try:
                    await worker_client.close()
                except Exception:
                    pass
                await worker_pool.release_worker(worker_client.worker_id)

            duration_sec = round(time.time() - call_start_time, 2)
            try:
                await ws.send_json({
                    "type": "session_ended",
                    "session_id": call_session.id,
                    "reason": disconnect_reason,
                    "duration_sec": duration_sec,
                })
                await ws.close()
            except Exception:
                pass

            async with session_factory() as db:
                call_svc = CallSessionService(db)
                await call_svc.end_call_session(
                    session_id=call_session.id,
                    end_reason=disconnect_reason,
                    duration_sec=duration_sec,
                    ttfa_ms=0.0,
                )
