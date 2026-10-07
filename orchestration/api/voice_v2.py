"""
Lean S2S Voice Session Runtime Router (/v2/voice).
Handles full-duplex WebSocket streaming for 16 kHz web clients and 8 kHz telephony callers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect, status

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

# In-memory debug logs storage (ring buffer of recent sessions)
SESSION_DEBUG_LOGS: dict[str, dict[str, Any]] = {}
MAX_DEBUG_SESSIONS = 100


@router.get("/v2/debug/session/{session_id}")
@router.get("/debug/session/{session_id}")
async def get_session_debug(session_id: str):
    """Retrieve detailed per-call debug diagnostics and audio metrics."""
    if session_id not in SESSION_DEBUG_LOGS:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found in debug logs")
    return SESSION_DEBUG_LOGS[session_id]


@router.get("/v2/debug/sessions")
@router.get("/debug/sessions")
async def list_session_debugs():
    """List recent debug sessions."""
    return list(SESSION_DEBUG_LOGS.values())


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
    voice_prompt: str | None = Query(None, description="Override voice prompt (.pt)"),
    audio_temperature: float | None = Query(None, description="Audio generation temperature (0.0 - 2.0)"),
    text_temperature: float | None = Query(None, description="Text generation temperature (0.0 - 2.0)"),
    audio_topk: int | None = Query(None, description="Audio top-k sampling"),
    text_topk: int | None = Query(None, description="Text top-k sampling"),
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
            target_ver = agent.versions[-1]

    # Signal connection status to the client
    await websocket.send_json({
        "type": "status",
        "status": "connecting",
        "message": "Connecting to voice engine...",
    })

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

    # Parse pipeline settings for generation parameters
    pipeline_cfg: dict[str, Any] = {}
    raw_pipe = getattr(target_ver, "pipeline_json", None) or getattr(agent, "draft_pipeline_json", "{}")
    if raw_pipe:
        try:
            pipeline_cfg = json.loads(raw_pipe)
        except Exception:
            pass

    resolved_audio_temp = audio_temperature if audio_temperature is not None else pipeline_cfg.get("audio_temperature", 0.8)
    resolved_text_temp = text_temperature if text_temperature is not None else pipeline_cfg.get("text_temperature", 0.7)
    resolved_audio_topk = audio_topk if audio_topk is not None else pipeline_cfg.get("audio_topk", 250)
    resolved_text_topk = text_topk if text_topk is not None else pipeline_cfg.get("text_topk", 25)
    resolved_voice = voice_prompt or pipeline_cfg.get("voice_id") or voice_id

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
        strict=True,
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

    # Telemetry tracking state for diagnosis & /v2/debug/session/{id}
    telemetry: dict[str, Any] = {
        "session_id": call_session_id,
        "agent_id": agent.id,
        "agent_name": agent.name,
        "sample_rates": {
            "browser_gateway": sample_rate,
            "gateway_worker": MODEL_SAMPLE_RATE,
            "worker_gateway": MODEL_SAMPLE_RATE,
            "gateway_browser": sample_rate,
        },
        "traffic": {
            "browser_to_gateway": {"bytes": 0, "frames": 0},
            "gateway_to_worker": {"bytes": 0, "frames": 0},
            "worker_to_gateway": {"bytes": 0, "frames": 0},
            "gateway_to_browser": {"bytes": 0, "frames": 0},
        },
        "audio_levels": {
            "gateway_to_worker_rms": 0.0,
            "gateway_to_worker_peak": 0.0,
            "worker_to_gateway_rms": 0.0,
            "worker_to_gateway_peak": 0.0,
        },
        "tokens_received": 0,
        "priming_time_ms": 0.0,
        "closed_by": "unknown",
        "close_code": 1000,
        "close_reason": "active",
        "client_audio_context_state": "unknown",
        "created_at": call_start_time,
        "duration_sec": 0.0,
    }
    if len(SESSION_DEBUG_LOGS) >= MAX_DEBUG_SESSIONS:
        # Drop oldest
        oldest_k = next(iter(SESSION_DEBUG_LOGS))
        del SESSION_DEBUG_LOGS[oldest_k]
    SESSION_DEBUG_LOGS[call_session_id] = telemetry

    # 4. Initialize EndOfCallDetector
    detector = EndOfCallDetector(
        ending_text=ending_text,
        silence_timeout_sec=end_silence_sec,
        max_duration_sec=max_duration_sec,
    )

    # 5. Initialize Resamplers
    # Moshi PersonaPlex operates on exact 1920-sample frames (80ms at 24kHz = 12.5Hz frame rate)
    in_buffer = StreamingResampleBuffer(
        in_rate=sample_rate,
        out_rate=MODEL_SAMPLE_RATE,
        out_frame_samples=1920,
        quality="QQ",
    )
    out_resampler = AudioResampler(in_rate=MODEL_SAMPLE_RATE, out_rate=sample_rate, quality="QQ")

    # 6. Priming with Keepalive Heartbeats & Early Disconnect Protection
    priming_start = time.perf_counter()
    priming_stop = asyncio.Event()
    connect_task: asyncio.Task | None = None

    async def priming_heartbeat():
        while not priming_stop.is_set():
            await asyncio.sleep(1.0)
            if priming_stop.is_set():
                break
            elapsed_ms = round((time.perf_counter() - priming_start) * 1000, 1)
            try:
                await websocket.send_json({
                    "type": "status",
                    "status": "priming",
                    "elapsed_ms": elapsed_ms,
                    "message": f"Priming voice model ({round(elapsed_ms / 1000, 1)}s)...",
                })
            except Exception:
                priming_stop.set()
                if connect_task and not connect_task.done():
                    connect_task.cancel()
                break

    priming_ticker = asyncio.create_task(priming_heartbeat())

    # 7. Acquire Worker from Pool
    worker_client: PersonaPlexWorkerClient | None = None
    try:
        worker_client = await worker_pool.acquire_worker(session_id=call_session_id, timeout=15.0)
    except Exception as e:
        priming_stop.set()
        priming_ticker.cancel()
        logger.error(f"Failed to acquire worker for session {call_session_id}: {e}")
        telemetry["closed_by"] = "gateway"
        telemetry["close_reason"] = f"Worker acquire failed: {e}"
        await websocket.send_json({"type": "error", "message": "All inference workers are busy. Please retry."})
        await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER)
        return

    persona_cfg = PersonaConfig(
        id=agent.id,
        name=agent.name,
        voice_prompt=resolved_voice,
        text_prompt=compiled.formatted_prompt,
        system_prompt=compiled.formatted_prompt,
        audio_temperature=float(resolved_audio_temp),
        text_temperature=float(resolved_text_temp),
        top_k_audio=int(resolved_audio_topk),
        top_k_text=int(resolved_text_topk),
    )

    try:
        connect_task = asyncio.create_task(worker_client.connect(session_id=call_session_id, persona=persona_cfg))
        await connect_task
    except asyncio.CancelledError:
        logger.info("Worker connect cancelled due to early client disconnect.")
        telemetry["closed_by"] = "client"
        telemetry["close_reason"] = "client_disconnected_during_priming"
        await worker_pool.release_worker(worker_client.worker_id)
        return
    except Exception as e:
        priming_stop.set()
        priming_ticker.cancel()
        logger.error(f"Failed to connect worker client: {e}")
        telemetry["closed_by"] = "gateway"
        telemetry["close_reason"] = f"Worker connection failed: {e}"
        await worker_pool.release_worker(worker_client.worker_id)
        try:
            await websocket.send_json({"type": "error", "message": f"Worker connection failed: {e}"})
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception:
            pass
        return
    finally:
        priming_stop.set()
        priming_ticker.cancel()

    priming_ms = round((time.perf_counter() - priming_start) * 1000, 1)
    telemetry["priming_time_ms"] = priming_ms

    # Notify client of model ready state
    await websocket.send_json({
        "type": "status",
        "status": "ready",
        "priming_time_ms": priming_ms,
        "message": "Voice model primed and ready.",
    })

    # Notify client of ready session state
    await websocket.send_json({
        "type": "session_started",
        "session_id": call_session_id,
        "agent_id": agent.id,
        "agent_name": agent.name,
        "voice_id": resolved_voice,
        "timezone": tz_name,
        "sample_rate": sample_rate,
        "codec": codec,
        "token_count": compiled.token_count,
        "greeting_mode": greeting_mode,
        "audio_temperature": float(resolved_audio_temp),
        "text_temperature": float(resolved_text_temp),
    })

    # Greeting delivery if agent speaks first (strictly rendered variables)
    if greeting_mode == "agent_first" and greeting_text.strip():
        g_rendered = greeting_text
        for k, v in {
            "company": "BrightNet",
            "customer_name": customer_name or "the caller",
            "caller_name": caller_name or "the caller",
            "agent_name": agent.name,
            "day_part": compiled.day_part,
        }.items():
            g_rendered = g_rendered.replace(f"{{{{{k}}}}}", str(v))
        await websocket.send_json({
            "type": "transcript",
            "role": "assistant",
            "text": g_rendered,
            "is_greeting": True,
        })

    # 8. Full-Duplex Bi-Directional Streaming Engine
    call_stream_start = time.time()
    detector.start_session(call_stream_start)
    stop_event = asyncio.Event()
    disconnect_reason = "normal"

    def trigger_stop(reason: str):
        nonlocal disconnect_reason
        logger.debug(f"Triggering voice session stop: {reason}")
        disconnect_reason = reason
        stop_event.set()

    turn_counter = 0
    current_agent_turn_text = []
    current_agent_turn_start = 0.0

    async def client_to_worker_loop():
        try:
            while not stop_event.is_set():
                raw = await websocket.receive()
                if raw.get("type") == "websocket.disconnect":
                    trigger_stop("ws_disconnect_message")
                    break

                if "bytes" in raw and raw["bytes"]:
                    data = raw["bytes"]
                    # Handle optional 0x01 prefix
                    if len(data) > 0 and data[0] == 0x01:
                        data = data[1:]

                    if len(data) == 0:
                        continue

                    telemetry["traffic"]["browser_to_gateway"]["bytes"] += len(data)
                    telemetry["traffic"]["browser_to_gateway"]["frames"] += 1

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

                    # Resample to 24 kHz model rate in exact 1920-sample frames
                    frames_24k = in_buffer.push_chunk(f32_samples)
                    for frame_24k in frames_24k:
                        frms = compute_rms(frame_24k)
                        fpeak = float(np.max(np.abs(frame_24k)))
                        telemetry["audio_levels"]["gateway_to_worker_rms"] = round(telemetry["audio_levels"]["gateway_to_worker_rms"] * 0.9 + frms * 0.1, 4)
                        telemetry["audio_levels"]["gateway_to_worker_peak"] = round(max(telemetry["audio_levels"]["gateway_to_worker_peak"], fpeak), 4)
                        telemetry["traffic"]["gateway_to_worker"]["bytes"] += frame_24k.nbytes
                        telemetry["traffic"]["gateway_to_worker"]["frames"] += 1
                        await worker_client.send_audio(frame_24k)

                elif "text" in raw and raw["text"]:
                    try:
                        msg_json = json.loads(raw["text"])
                        mtype = msg_json.get("type", "")
                        if mtype == "hangup":
                            trigger_stop("user_hangup")
                            break
                        elif mtype == "ping":
                            await websocket.send_json({"type": "pong", "time": time.time()})
                        elif mtype == "interrupt":
                            await worker_client.send_control(ControlAction.PAUSE)
                        elif mtype == "client_info":
                            telemetry["client_audio_context_state"] = msg_json.get("audio_context_state", "unknown")
                    except json.JSONDecodeError:
                        pass
        except WebSocketDisconnect:
            trigger_stop("client_disconnected_exception")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"DEBUG_CLIENT_LOOP_EXC: {e}", flush=True)
            logger.debug(f"client_to_worker_loop exception: {e}")
            trigger_stop(f"client_loop_exc_{e}")

    async def worker_to_client_loop():
        nonlocal disconnect_reason, turn_counter, current_agent_turn_text, current_agent_turn_start
        try:
            async for msg in worker_client.recv_messages():
                if stop_event.is_set():
                    break

                if isinstance(msg, TextMessage):
                    token = msg.text
                    telemetry["tokens_received"] += 1
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

                    f32_worker = (
                        np.frombuffer(raw_audio, dtype=np.float32)
                        if isinstance(raw_audio, bytes)
                        else (raw_audio if raw_audio.dtype == np.float32 else raw_audio.astype(np.float32))
                    )
                    if len(f32_worker) == 0:
                        continue

                    telemetry["traffic"]["worker_to_gateway"]["bytes"] += len(raw_audio)
                    telemetry["traffic"]["worker_to_gateway"]["frames"] += 1
                    mrms = compute_rms(f32_worker)
                    mpeak = float(np.max(np.abs(f32_worker)))
                    telemetry["audio_levels"]["worker_to_gateway_rms"] = round(telemetry["audio_levels"]["worker_to_gateway_rms"] * 0.9 + mrms * 0.1, 4)
                    telemetry["audio_levels"]["worker_to_gateway_peak"] = round(max(telemetry["audio_levels"]["worker_to_gateway_peak"], mpeak), 4)

                    # Resample 24k -> client rate (16k or 8k)
                    resampled = out_resampler.resample_chunk(f32_worker, last=False)
                    if len(resampled) == 0:
                        continue

                    # Soft clip limiter to prevent speaker distortion
                    clipped_f32 = soft_clip(resampled, threshold=0.92)

                    if codec == "g711_ulaw":
                        pcm_bytes = encode_ulaw(clipped_f32)
                    else:
                        pcm_bytes = float32_to_int16(clipped_f32).tobytes()

                    telemetry["traffic"]["gateway_to_browser"]["bytes"] += len(pcm_bytes)
                    telemetry["traffic"]["gateway_to_browser"]["frames"] += 1
                    await websocket.send_bytes(pcm_bytes)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"worker_to_client_loop exception: {e}")
            trigger_stop(f"worker_loop_exc_{e}")

    async def lifecycle_heartbeat():
        try:
            while not stop_event.is_set():
                await asyncio.sleep(0.1)

                term_reason = detector.check_termination(time.time())
                if term_reason:
                    try:
                        await websocket.send_json({
                            "type": "call_ended",
                            "reason": term_reason,
                        })
                    except Exception:
                        pass
                    trigger_stop(term_reason)
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
            logger.debug(f"Voice task finished normally: {d}")

    for p in pending:
        p.cancel()
    await asyncio.gather(*pending, return_exceptions=True)

    # 9. Finalize Call Session in DB and Telemetry
    duration_sec = max(0.0, time.time() - call_start_time)
    telemetry["duration_sec"] = round(duration_sec, 2)
    telemetry["close_reason"] = disconnect_reason
    telemetry["closed_by"] = "client" if ("client" in disconnect_reason or "user" in disconnect_reason) else "server"
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
