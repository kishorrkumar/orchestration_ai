"""
Lean S2S Voice Session Runtime Router (/v2/voice).
Handles full-duplex WebSocket streaming for 16 kHz web clients and 8 kHz telephony callers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

import numpy as np
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status

from ..audio.codecs import decode_ulaw, encode_ulaw
from ..audio.dsp import compute_rms, float32_to_int16, int16_to_float32, soft_clip
from ..audio.resample import AudioResampler, StreamingResampleBuffer
from ..db.models import AgentVersion
from ..db.service import AgentService, CallSessionService
from ..db.session import get_session_factory
from ..domain.engines.spec import EngineType
from ..engines.base import SessionContext

try:
    from ..engines.cascaded.engine import CascadedVoiceEngine
except ImportError:
    CascadedVoiceEngine = None  # type: ignore
from ..persona.registry import PersonaConfig
from ..pipeline.end_detector import EndOfCallDetector
from ..prompts.compiler import compile_prompt
from ..protocol.audio import (
    CLIENT_SAMPLE_RATE,
    MODEL_SAMPLE_RATE,
)
from ..protocol.messages import AudioMessage, ControlAction, TextMessage
from ..worker.client import PersonaPlexWorkerClient
from ..worker.pool import WorkerPool

logger = logging.getLogger("orchestration.voice_v2")

router = APIRouter(tags=["Voice V2"])


@router.websocket("/v2/voice")
async def voice_v2_endpoint(
    websocket: WebSocket,
    agent_id: str = Query(..., description="ID of the voice agent to talk with"),
    version: int | None = Query(None, description="Specific version number (defaults to published version)"),
    sample_rate: int = Query(CLIENT_SAMPLE_RATE, description="Client audio sample rate (16000 or 8000)"),
    codec: str = Query("pcm16", description="Client audio codec ('pcm16' or 'g711_ulaw')"),
    caller_name: str | None = Query(None, description="Optional caller name for template interpolation"),
    customer_name: str | None = Query(None, description="Optional customer name for template interpolation"),
    phone_number: str | None = Query(None, description="Optional phone number for template interpolation"),
):
    """
    Production S2S WebSocket endpoint.
    Streaming 16 kHz (web) or 8 kHz (telephony G.711) full-duplex conversational voice.
    """
    await websocket.accept()

    session_factory = get_session_factory()

    # 1. Resolve Agent and Version snapshot from Database
    async with session_factory() as db:
        agent_svc = AgentService(db)
        agent = await agent_svc.get_agent(agent_id)
        if not agent:
            await websocket.send_json({"type": "error", "message": f"Agent '{agent_id}' not found"})
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        # Resolve version snapshot
        target_ver: AgentVersion | None = None
        if version is not None:
            target_ver = await agent_svc.get_version(agent_id, version)
        elif agent.published_version_id and agent.versions:
            target_ver = next((v for v in agent.versions if v.id == agent.published_version_id), None)

        if target_ver is None and agent.versions:
            # Fallback to latest version
            target_ver = agent.versions[-1]

    # Check Engine type
    resolved_engine = target_ver.engine if (target_ver and hasattr(target_ver, "engine")) else getattr(agent, "draft_engine", "personaplex_s2s")

    # If Engine B (Cascaded Cloud pipeline)
    if resolved_engine == EngineType.CASCADED_CLOUD.value or resolved_engine == "cascaded_cloud":
        call_start_time = time.time()
        call_session_id = f"call_{int(call_start_time)}_{agent.id[:8]}"
        agent_version_id = target_ver.id if target_ver else None
        async with session_factory() as db:
            call_svc = CallSessionService(db)
            session_record = await call_svc.create_session(
                agent_id=agent.id,
                agent_version_id=agent_version_id,
                session_id=call_session_id,
                engine=EngineType.CASCADED_CLOUD.value,
            )
            await db.commit()

        context = SessionContext(
            websocket=websocket,
            agent=agent,
            version=target_ver,
            session_record=session_record,
            session_factory=session_factory,
            client_sample_rate=sample_rate,
            codec=codec,
            template_vars={
                "caller_name": caller_name or "",
                "customer_name": customer_name or "",
                "phone_number": phone_number or "",
            },
            workspace_id=agent.workspace_id,
        )

        if CascadedVoiceEngine is None:
            await websocket.send_json({
                "type": "error",
                "message": "Engine B requires Pipecat. Install via: pip install 'pipecat-ai>=1.12.0'",
            })
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
            return

        engine_runner = CascadedVoiceEngine()
        await engine_runner.run_session(context)
        return

    # Engine A (PersonaPlex S2S): requires GPU/Mock Worker Pool
    worker_pool: WorkerPool = getattr(websocket.app.state, "pool", None)
    if not worker_pool:
        await websocket.send_json({"type": "error", "message": "Worker pool not available on gateway"})
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        return

    # Extract active configuration values
    voice_id = target_ver.voice_id if target_ver else agent.draft_voice_id
    greeting_text = target_ver.greeting_text if target_ver else agent.draft_greeting_text
    greeting_mode = target_ver.greeting_mode if target_ver else agent.draft_greeting_mode
    system_prompt = target_ver.system_prompt if target_ver else agent.draft_system_prompt
    ending_text = target_ver.ending_text if target_ver else agent.draft_ending_text
    end_silence_sec = target_ver.end_silence_sec if target_ver else agent.draft_end_silence_sec
    max_duration_sec = target_ver.max_duration_sec if target_ver else agent.draft_max_duration_sec
    tz_name = target_ver.timezone if target_ver else agent.draft_timezone
    agent_version_id = target_ver.id if target_ver else None

    # 2. Compile prompt with dynamic local time and caller interpolation
    compiled = compile_prompt(
        system_prompt=system_prompt,
        greeting_text=greeting_text,
        greeting_mode=greeting_mode,
        ending_text=ending_text,
        agent_name=agent.name,
        timezone=tz_name,
        caller_name=caller_name,
        customer_name=customer_name,
        phone_number=phone_number,
    )

    # 3. Create CallSession in Database
    call_start_time = time.time()
    call_session_id = f"call_{int(call_start_time)}_{agent.id[:8]}"
    async with session_factory() as db:
        call_svc = CallSessionService(db)
        await call_svc.create_session(
            agent_id=agent.id,
            agent_version_id=agent_version_id,
            session_id=call_session_id,
        )
        await db.commit()

    # 4. Initialize EndOfCallDetector
    detector = EndOfCallDetector(
        ending_text=ending_text,
        silence_timeout_sec=end_silence_sec,
        max_duration_sec=max_duration_sec,
    )
    detector.start_session(time.time())

    # 5. Initialize Resamplers
    in_buffer = StreamingResampleBuffer(
        in_rate=sample_rate,
        out_rate=MODEL_SAMPLE_RATE,
        out_frame_samples=480,  # 20ms at 24kHz = exact allowed Opus frame
        quality="QQ",
    )
    out_resampler = AudioResampler(in_rate=MODEL_SAMPLE_RATE, out_rate=sample_rate, quality="QQ")

    # 6. Acquire Worker from Pool
    worker_client: PersonaPlexWorkerClient | None = None
    try:
        worker_client = await worker_pool.acquire_worker(session_id=call_session_id, timeout=10.0)
    except Exception as e:
        logger.error(f"Failed to acquire worker for session {call_session_id}: {e}")
        await websocket.send_json({"type": "error", "message": "All inference workers are busy. Please retry."})
        await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER)
        return

    # 7. Connect Worker with S2S Persona configuration
    persona_cfg = PersonaConfig(
        id=agent.id,
        name=agent.name,
        voice_prompt=voice_id,
        text_prompt=compiled.formatted_prompt,
        system_prompt=compiled.formatted_prompt,
    )

    try:
        await worker_client.connect(session_id=call_session_id, persona=persona_cfg)
    except Exception as e:
        logger.error(f"Failed to connect worker client: {e}")
        await worker_pool.release_worker(worker_client.worker_id)
        await websocket.send_json({"type": "error", "message": f"Worker connection failed: {e}"})
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        return

    # Notify client of ready session state
    await websocket.send_json({
        "type": "session_started",
        "session_id": call_session_id,
        "agent_id": agent.id,
        "agent_name": agent.name,
        "voice_id": voice_id,
        "timezone": tz_name,
        "sample_rate": sample_rate,
        "codec": codec,
        "token_count": compiled.token_count,
        "greeting_mode": greeting_mode,
    })

    # Greeting delivery if agent speaks first
    if greeting_mode == "agent_first" and greeting_text.strip():
        await websocket.send_json({
            "type": "transcript",
            "role": "assistant",
            "text": greeting_text,
            "is_greeting": True,
        })

    # 8. Full-Duplex Bi-Directional Streaming Engine
    stop_event = asyncio.Event()
    disconnect_reason = "normal"
    turn_counter = 0
    current_agent_turn_text = []
    current_agent_turn_start = 0.0

    async def client_to_worker_loop():
        nonlocal disconnect_reason
        try:
            while not stop_event.is_set():
                raw = await websocket.receive()
                if "bytes" in raw and raw["bytes"]:
                    data = raw["bytes"]
                    # Handle optional 0x01 prefix
                    if len(data) > 0 and data[0] == 0x01:
                        data = data[1:]

                    if len(data) == 0:
                        continue

                    # Decode audio according to codec
                    if codec == "g711_ulaw":
                        pcm16 = decode_ulaw(data)
                        f32_samples = int16_to_float32(pcm16)
                    else:
                        pcm16 = np.frombuffer(data, dtype=np.int16)
                        f32_samples = int16_to_float32(pcm16)

                    # Speech energy detection for silence reset & barge-in
                    rms = compute_rms(f32_samples)
                    if rms > 0.012:
                        detector.on_user_speech(time.time())
                        # If user interrupts assistant speech, send pause control
                        if current_agent_turn_text:
                            await worker_client.send_control(ControlAction.PAUSE)

                    # Resample to 24 kHz model rate in exact 480-sample frames
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
                            await websocket.send_json({"type": "pong", "time": time.time()})
                        elif mtype == "interrupt":
                            await worker_client.send_control(ControlAction.PAUSE)
                    except json.JSONDecodeError:
                        pass
        except WebSocketDisconnect:
            disconnect_reason = "client_disconnected"
            stop_event.set()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"client_to_worker_loop exception: {e}")
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

                    # Stream text token to client
                    await websocket.send_json({
                        "type": "transcript",
                        "role": "assistant",
                        "text": token,
                    })

                    # Feed to EndOfCallDetector
                    matched = detector.on_agent_token(token, time.time())
                    if matched:
                        detector.on_agent_speaking_stopped(time.time())

                elif isinstance(msg, AudioMessage):
                    raw_audio = msg.data
                    if len(raw_audio) == 0:
                        continue

                    # Resample 24k -> client rate (16k or 8k)
                    resampled = out_resampler.resample_chunk(raw_audio, last=False)
                    if len(resampled) == 0:
                        continue

                    # Soft clip limiter to prevent speaker distortion
                    clipped_f32 = soft_clip(resampled, threshold=0.92)

                    if codec == "g711_ulaw":
                        pcm_bytes = encode_ulaw(clipped_f32)
                    else:
                        pcm_bytes = float32_to_int16(clipped_f32).tobytes()

                    await websocket.send_bytes(pcm_bytes)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"worker_to_client_loop exception: {e}")
            stop_event.set()

    async def lifecycle_heartbeat():
        nonlocal disconnect_reason
        try:
            while not stop_event.is_set():
                await asyncio.sleep(0.1)

                term_reason = detector.check_termination(time.time())
                if term_reason:
                    disconnect_reason = term_reason
                    await websocket.send_json({
                        "type": "call_ended",
                        "reason": term_reason,
                    })
                    stop_event.set()
                    break

        except asyncio.CancelledError:
            pass

    # Run tasks concurrently
    tasks = [
        asyncio.create_task(client_to_worker_loop()),
        asyncio.create_task(worker_to_client_loop()),
        asyncio.create_task(lifecycle_heartbeat()),
    ]

    stop_task = asyncio.create_task(stop_event.wait())
    all_tasks = [stop_task, *tasks]

    # Wait until stop_event or any task failure
    done, pending = await asyncio.wait(
        all_tasks,
        return_when=asyncio.FIRST_COMPLETED,
    )

    for d in done:
        if d.exception():
            logger.error(f"Voice task exception: {d.exception()}", exc_info=d.exception())
        else:
            logger.info(f"Voice task finished normally: {d}")

    for p in pending:
        p.cancel()
    await asyncio.gather(*pending, return_exceptions=True)

    # 9. Finalize Call Session in DB
    duration_sec = max(0.0, time.time() - call_start_time)
    try:
        async with session_factory() as db:
            call_svc = CallSessionService(db)
            if current_agent_turn_text:
                full_turn_text = "".join(current_agent_turn_text).strip()
                if full_turn_text:
                    await call_svc.add_turn(
                        session_id=call_session_id,
                        idx=1,
                        role="agent",
                        text=full_turn_text,
                        started_ms=current_agent_turn_start,
                    )

            await call_svc.end_session(
                session_id=call_session_id,
                end_reason=disconnect_reason,
                duration_sec=duration_sec,
            )
            await db.commit()
    except Exception as db_err:
        logger.error(f"Error finalizing call session {call_session_id}: {db_err}")

    # 10. Clean up worker & connection
    if worker_client:
        await worker_pool.release_worker(worker_client.worker_id)

    try:
        await websocket.close()
    except Exception:
        pass
