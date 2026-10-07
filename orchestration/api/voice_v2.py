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
from fastapi import APIRouter, Header, HTTPException, Query, WebSocket, WebSocketDisconnect, status

from ..audio.codecs import decode_ulaw, encode_ulaw
from ..audio.dsp import compute_rms, float32_to_int16, int16_to_float32, soft_clip
from ..audio.framing import InboundAudioFrameProcessor
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
from ..persona.registry import OFFICIAL_VOICE_PRESETS, PersonaConfig
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


from ..settings import app_settings

@router.get("/debug/sessions/{session_id}")
@router.get("/v2/debug/session/{session_id}")
@router.get("/debug/session/{session_id}")
async def get_session_debug(
    session_id: str,
    token: str | None = Query(None),
    authorization: str | None = Header(None),
):
    """Retrieve detailed per-call debug diagnostics and audio metrics (auth-protected)."""
    if app_settings.AUTH_TOKEN:
        auth_header = authorization.replace("Bearer ", "").strip() if authorization else None
        provided = token or auth_header
        if provided != app_settings.AUTH_TOKEN:
            raise HTTPException(status_code=401, detail="Unauthorized: invalid or missing auth token")

    if session_id not in SESSION_DEBUG_LOGS:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found in debug logs")
    return SESSION_DEBUG_LOGS[session_id]


@router.get("/debug/sessions")
@router.get("/v2/debug/sessions")
async def list_session_debugs(
    token: str | None = Query(None),
    authorization: str | None = Header(None),
):
    """List recent debug sessions (auth-protected)."""
    if app_settings.AUTH_TOKEN:
        auth_header = authorization.replace("Bearer ", "").strip() if authorization else None
        provided = token or auth_header
        if provided != app_settings.AUTH_TOKEN:
            raise HTTPException(status_code=401, detail="Unauthorized: invalid or missing auth token")
    return list(SESSION_DEBUG_LOGS.values())


from pathlib import Path
import datetime

CONVERSATIONS_DIR = Path("conversations")
CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_CONVERSATIONS_DIR = Path("logs/conversations")
LOGS_CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/conversations")
@router.get("/v2/conversations")
@router.get("/api/v2/conversations")
async def list_conversations(
    token: str | None = Query(None),
    authorization: str | None = Header(None),
):
    """List recent conversation JSON recordings with turn counts and audio stats."""
    if app_settings.AUTH_TOKEN:
        auth_header = authorization.replace("Bearer ", "").strip() if authorization else None
        provided = token or auth_header
        if provided != app_settings.AUTH_TOKEN:
            raise HTTPException(status_code=401, detail="Unauthorized: invalid or missing auth token")

    all_files = sorted(
        list(CONVERSATIONS_DIR.glob("*.json")) + list(LOGS_CONVERSATIONS_DIR.glob("*.json")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    seen_ids = set()
    summaries = []
    for f in all_files:
        if f.stem in seen_ids:
            continue
        seen_ids.add(f.stem)
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            summaries.append({
                "session_id": data.get("session_id", f.stem),
                "agent_name": data.get("agent_name"),
                "voice_id": data.get("voice_id"),
                "duration_sec": data.get("duration_sec"),
                "turns_count": len(data.get("turns", [])),
                "disconnect_reason": data.get("disconnect_reason"),
                "started_at": data.get("created_at_iso") or data.get("started_at"),
                "file_path": str(f),
            })
        except Exception:
            pass
        if len(summaries) >= 50:
            break
    return summaries


@router.get("/conversations/{session_id}")
@router.get("/v2/conversations/{session_id}")
@router.get("/api/v2/conversations/{session_id}")
async def get_conversation(
    session_id: str,
    token: str | None = Query(None),
    authorization: str | None = Header(None),
):
    """Retrieve full conversation JSON recording for a specific call session."""
    if app_settings.AUTH_TOKEN:
        auth_header = authorization.replace("Bearer ", "").strip() if authorization else None
        provided = token or auth_header
        if provided != app_settings.AUTH_TOKEN:
            raise HTTPException(status_code=401, detail="Unauthorized: invalid or missing auth token")

    target_file = CONVERSATIONS_DIR / f"{session_id}.json"
    if not target_file.exists():
        target_file = LOGS_CONVERSATIONS_DIR / f"{session_id}.json"
    if not target_file.exists():
        raise HTTPException(status_code=404, detail=f"Conversation recording for session '{session_id}' not found.")
    try:
        with open(target_file, "r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read conversation recording: {e}")



@router.websocket("/v2/voice")
async def voice_v2_endpoint(
    websocket: WebSocket,
    agent_id: str = Query("default", description="ID of the voice agent to talk with"),
    version: int | None = Query(None, description="Specific version number (defaults to published version)"),
    sample_rate: int = Query(CLIENT_SAMPLE_RATE, description="Client audio sample rate (16000 or 8000)"),
    codec: str = Query("pcm16", description="Client audio codec ('pcm16' or 'g711_ulaw')"),
    token: str | None = Query(None, description="Optional authentication token"),
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
    # 0. Authentication Verification
    if app_settings.AUTH_TOKEN:
        header_auth = websocket.headers.get("Authorization", "").replace("Bearer ", "").strip()
        auth_val = token or header_auth
        if auth_val != app_settings.AUTH_TOKEN:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Unauthorized: Invalid auth token")
            return

    await websocket.accept()
    session_factory = get_session_factory()

    # 1. Resolve Agent and Version snapshot from Database
    async with session_factory() as db:
        agent_svc = AgentService(db)
        agent = None
        if agent_id and agent_id != "default":
            agent = await agent_svc.get_agent(agent_id)
        else:
            agents_list, _ = await agent_svc.list_agents()
            if agents_list:
                agent = agents_list[0]

        if not agent:
            await websocket.send_json({"type": "error", "message": f"Agent '{agent_id}' not found"})
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        # Resolve version snapshot
        target_ver: AgentVersion | None = None
        if version is not None:
            target_ver = await agent_svc.get_version(agent.id, version)
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

    from ..persona.registry import OFFICIAL_VOICE_PRESETS, get_existing_voice_files
    available_voices = get_existing_voice_files()
    norm_voice = str(resolved_voice) if resolved_voice else "NATF0.pt"
    if not norm_voice.endswith(".pt") and not norm_voice.endswith(".wav"):
        norm_voice = f"{norm_voice}.pt"

    if norm_voice not in available_voices and not norm_voice.endswith(".wav"):
        logger.error(f"Requested voice '{resolved_voice}' not found on disk. Available: {available_voices}")
        await websocket.send_json({
            "type": "error",
            "message": f"voice '{resolved_voice}' not found, available: {', '.join(available_voices)}",
        })
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason=f"voice {resolved_voice} not found")
        return
    resolved_voice = norm_voice

    # 2. Compile prompt - pure system prompt with variables removed
    try:
        compiled = compile_prompt(
            system_prompt=system_prompt,
            agent_name=agent.name,
            strict=False,
        )
    except Exception as compile_err:
        logger.warning(f"compile_prompt note: {compile_err}, using direct system prompt")
        from ..protocol.prompt import wrap_system_prompt
        clean_text = system_prompt.replace("{{", "").replace("}}", "")
        compiled = type("CompiledPromptStub", (), {"text": wrap_system_prompt(clean_text)})()

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
    effective_silence_sec = float(end_silence_sec) if (end_silence_sec and float(end_silence_sec) > 0) else 1800.0
    effective_max_duration = float(max_duration_sec) if (max_duration_sec and float(max_duration_sec) > 0) else 1800.0
    detector = EndOfCallDetector(
        ending_text=ending_text,
        silence_timeout_sec=effective_silence_sec,
        max_duration_sec=effective_max_duration,
    )

    # 5. Initialize Inbound Framing Processor & Outbound Resampler
    # Moshi PersonaPlex operates on exact 1920-sample frames (80ms at 24kHz = 12.5Hz frame rate)
    inbound_processor = InboundAudioFrameProcessor(
        codec=codec,
        sample_rate=sample_rate,
        model_sample_rate=MODEL_SAMPLE_RATE,
        out_frame_samples=1920,
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
                    "message": f"Getting ready, about 10 seconds ({round(elapsed_ms / 1000, 1)}s)...",
                })
            except Exception:
                priming_stop.set()
                if connect_task and not connect_task.done():
                    connect_task.cancel()
                break

    priming_ticker = asyncio.create_task(priming_heartbeat())

    # 7. Acquire Worker from Pool (with health probe)
    worker_client: PersonaPlexWorkerClient | None = None
    try:
        worker_client = await worker_pool.acquire_worker(session_id=call_session_id, timeout=15.0)
    except Exception as e:
        priming_stop.set()
        priming_ticker.cancel()
        logger.error(f"Failed to acquire worker for session {call_session_id}: {e}")
        telemetry["closed_by"] = "gateway"
        telemetry["close_reason"] = f"Worker acquire failed: {e}"
        await websocket.send_json({"type": "error", "message": f"All inference workers are busy or unhealthy. Please retry: {e}"})
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
        await worker_pool.release_worker(worker_client.worker_id, success=True)
        return
    except Exception as e:
        priming_stop.set()
        priming_ticker.cancel()
        logger.error(f"Failed to connect worker client: {e}")
        telemetry["closed_by"] = "gateway"
        telemetry["close_reason"] = f"Worker connection failed: {e}"
        await worker_pool.release_worker(worker_client.worker_id, success=False)
        try:
            await websocket.send_json({"type": "error", "message": f"Worker connection failed: {e}"})
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR, reason=f"Worker connection failed: {e}")
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

    # 8. Full-Duplex Bi-Directional Streaming Engine
    # Note: PersonaPlex S2S vocalizes its own opening turn / greeting naturally from prompt conditioning.
    # We DO NOT send synthetic duplicate text transcripts prior to actual model vocalization.
    call_stream_start = time.time()
    detector.start_session(call_stream_start)
    stop_event = asyncio.Event()
    disconnect_reason = "normal"
    first_agent_audio_emitted = False
    ttfa_greeting_ms: float | None = None

    def trigger_stop(reason: str):
        nonlocal disconnect_reason
        logger.warning(f"[VoiceSession {call_session_id}] Triggering voice session stop: {reason}")
        disconnect_reason = reason
        stop_event.set()

    turn_counter = 0
    current_agent_turn_text = []
    current_agent_turn_start = 0.0

    conversation_turns: list[dict[str, Any]] = []
    current_agent_buffer: list[str] = []
    current_agent_start_ms: float = 0.0

    def commit_agent_turn():
        nonlocal current_agent_buffer, current_agent_start_ms
        if current_agent_buffer:
            text = "".join(current_agent_buffer).strip()
            if text:
                conversation_turns.append({
                    "role": "assistant",
                    "text": text,
                    "started_ms": round(current_agent_start_ms, 1),
                    "time_offset_sec": round(current_agent_start_ms / 1000.0, 2),
                    "timestamp": time.time(),
                })
            current_agent_buffer.clear()

    is_pcm_client = (codec == "pcm16" or codec == "g711_ulaw")

    # Inbound Opus encoder and outbound Opus reader for PCM clients (tests / telephony)
    server_opus_writer = None
    server_opus_reader = None
    if is_pcm_client:
        try:
            import sphn
            server_opus_writer = sphn.OpusStreamWriter(MODEL_SAMPLE_RATE)
            server_opus_reader = sphn.OpusStreamReader(MODEL_SAMPLE_RATE)
        except Exception as e:
            logger.warning(f"sphn library unavailable on gateway: {e}")

    # Decoupled audio queue for PCM clients: chunks -> resampled 24kHz frames -> worker
    audio_frame_queue: asyncio.Queue[np.ndarray] = asyncio.Queue()

    # Worker streaming & Rollover State
    active_worker_holder: list[PersonaPlexWorkerClient | None] = [worker_client]
    worker_message_queue: asyncio.Queue[bytes] = asyncio.Queue()
    active_worker_generation = 0
    current_worker_start_time = time.time()
    rollover_in_progress = False
    session_rollover_count = 0
    last_user_speech_time = time.time()
    last_agent_token_time = time.time()

    async def stream_worker_frames(client: PersonaPlexWorkerClient):
        """Streams raw binary frames from upstream worker into unified worker_message_queue."""
        nonlocal active_worker_generation
        active_worker_generation += 1
        my_gen = active_worker_generation
        try:
            async for raw in client.recv_raw_frames():
                if stop_event.is_set() or my_gen != active_worker_generation:
                    break
                await worker_message_queue.put(raw)
        except asyncio.CancelledError:
            pass
        except Exception as err:
            logger.debug(f"[VoiceSession {call_session_id}] stream_worker_frames notice: {err}")

    # Launch initial worker receiver task
    worker_stream_task = asyncio.create_task(stream_worker_frames(worker_client))

    async def client_to_queue_loop():
        """
        Receives incoming audio & control frames from client WebSocket.
        Uses InboundAudioFrameProcessor to handle 1-byte framing, odd-length fragments,
        and binary control frames without ValueError or premature disconnects.
        """
        nonlocal last_user_speech_time
        try:
            while not stop_event.is_set():
                raw = await websocket.receive()
                if raw.get("type") == "websocket.disconnect":
                    logger.warning(f"[VoiceSession {call_session_id}] Client WebSocket disconnected: code={raw.get('code')}")
                    trigger_stop(f"ws_disconnect_{raw.get('code', 1000)}")
                    break

                if "bytes" in raw and raw["bytes"]:
                    data = raw["bytes"]
                    if len(data) == 0:
                        continue

                    telemetry["traffic"]["browser_to_gateway"]["bytes"] += len(data)
                    telemetry["traffic"]["browser_to_gateway"]["frames"] += 1

                    if not is_pcm_client:
                        # 100% Transparent Binary Relay path for browser and official client
                        telemetry["traffic"]["gateway_to_worker"]["bytes"] += len(data)
                        telemetry["traffic"]["gateway_to_worker"]["frames"] += 1
                        cur_w = active_worker_holder[0]
                        if cur_w is not None and cur_w.is_connected:
                            if data[0] in (0x01, 0x02):
                                await cur_w.send_raw(data)
                            elif data.startswith(b"OggS"):
                                await cur_w.send_raw(b"\x01" + data)
                            elif data[0] not in (0x03, 0x06):
                                await cur_w.send_raw(b"\x01" + data)
                    else:
                        # Robust PCM16 / G.711 processing with carryover & control separation
                        event_type, frames_24k, f32_samples = inbound_processor.process_frame(data)
                        if event_type == "audio" and f32_samples is not None:
                            rms = compute_rms(f32_samples)
                            if rms > 0.012:
                                last_user_speech_time = time.time()
                                detector.on_user_speech(last_user_speech_time)
                                # Neural S2S model handles turn-taking naturally.
                                # DO NOT send b"\x03\x02" to worker (upstream Moshi only accepts kind 1).
                            for frame_24k in frames_24k:
                                await audio_frame_queue.put(frame_24k)
                        elif event_type == "interrupt":
                            logger.info(f"[VoiceSession {call_session_id}] User binary interrupt noted (handled locally)")

                elif "text" in raw and raw["text"]:
                    try:
                        msg_json = json.loads(raw["text"])
                        mtype = msg_json.get("type", "")
                        if mtype == "hangup":
                            logger.info(f"[VoiceSession {call_session_id}] User clicked hangup in client")
                            trigger_stop("user_hangup")
                            break
                        elif mtype == "ping":
                            await websocket.send_json({"type": "pong", "time": time.time()})
                        elif mtype == "interrupt":
                            logger.info(f"[VoiceSession {call_session_id}] User JSON interrupt noted (handled locally)")
                        elif mtype in ("user_transcript", "transcript"):
                            user_text = msg_json.get("text", "").strip()
                            if user_text:
                                commit_agent_turn()
                                conversation_turns.append({
                                    "role": "user",
                                    "text": user_text,
                                    "time_offset_sec": round(time.time() - call_start_time, 2),
                                    "timestamp": time.time(),
                                })
                                last_user_speech_time = time.time()
                                detector.on_user_speech(last_user_speech_time)
                        elif mtype == "client_info":
                            telemetry["client_audio_context_state"] = msg_json.get("audio_context_state", "unknown")
                    except json.JSONDecodeError:
                        pass
        except WebSocketDisconnect as wsd:
            logger.warning(f"[VoiceSession {call_session_id}] WebSocketDisconnect: code={wsd.code}")
            trigger_stop(f"client_disconnected_{wsd.code}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[VoiceSession {call_session_id}] client_to_queue_loop error: {e}", exc_info=True)
            trigger_stop(f"client_loop_exc_{e}")

    async def queue_to_worker_pacer():
        """
        Clocks the PersonaPlex worker at continuous 12.5 Hz (every 80ms) cadence for PCM clients.
        Sends proper stream headers and adapts to both Opus-encoded and raw PCM workers.
        """
        frame_interval = 0.080  # 80ms = 12.5 Hz
        try:
            while not stop_event.is_set():
                frame_to_send: np.ndarray | None = None
                try:
                    frame_to_send = audio_frame_queue.get_nowait()
                except asyncio.QueueEmpty:
                    frame_to_send = np.zeros(1920, dtype=np.float32)

                frms = compute_rms(frame_to_send)
                fpeak = float(np.max(np.abs(frame_to_send)))
                telemetry["audio_levels"]["gateway_to_worker_rms"] = round(
                    telemetry["audio_levels"]["gateway_to_worker_rms"] * 0.9 + frms * 0.1, 4
                )
                telemetry["audio_levels"]["gateway_to_worker_peak"] = round(
                    max(telemetry["audio_levels"]["gateway_to_worker_peak"], fpeak), 4
                )

                cur_worker = active_worker_holder[0]
                if cur_worker is not None and cur_worker.is_connected:
                    if getattr(cur_worker, "use_opus", True) and server_opus_writer is not None:
                        server_opus_writer.append_pcm(frame_to_send)
                        payload = server_opus_writer.read_bytes()
                        if payload and len(payload) > 0:
                            telemetry["traffic"]["gateway_to_worker"]["bytes"] += len(payload)
                            telemetry["traffic"]["gateway_to_worker"]["frames"] += 1
                            await cur_worker.send_raw(b"\x01" + payload)
                    else:
                        telemetry["traffic"]["gateway_to_worker"]["bytes"] += frame_to_send.nbytes
                        telemetry["traffic"]["gateway_to_worker"]["frames"] += 1
                        await cur_worker.send_audio(frame_to_send)

                # Dynamically pace: drain quickly if frames accumulated, else 80ms
                q_size = audio_frame_queue.qsize()
                sleep_interval = 0.040 if q_size > 3 else frame_interval
                await asyncio.sleep(sleep_interval)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[VoiceSession {call_session_id}] queue_to_worker_pacer error: {e}", exc_info=True)
            trigger_stop(f"pacer_exc_{e}")

    async def worker_to_client_loop():
        nonlocal disconnect_reason, turn_counter, current_agent_turn_text, current_agent_turn_start
        nonlocal first_agent_audio_emitted, ttfa_greeting_ms, last_agent_token_time
        try:
            while not stop_event.is_set():
                try:
                    raw = await asyncio.wait_for(worker_message_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                if len(raw) == 0:
                    continue

                opcode = raw[0]
                if opcode == 0x00:
                    # Handshake byte
                    continue

                elif opcode == 0x01:
                    # Audio payload from worker
                    last_agent_token_time = time.time()
                    if not first_agent_audio_emitted:
                        first_agent_audio_emitted = True
                        ttfa_greeting_ms = round((time.time() - call_start_time) * 1000.0, 1)
                        telemetry["ttfa_greeting_ms"] = ttfa_greeting_ms

                    telemetry["traffic"]["worker_to_gateway"]["bytes"] += len(raw)
                    telemetry["traffic"]["worker_to_gateway"]["frames"] += 1

                    if not is_pcm_client:
                        # Transparent binary forward directly to client
                        telemetry["traffic"]["gateway_to_browser"]["bytes"] += len(raw)
                        telemetry["traffic"]["gateway_to_browser"]["frames"] += 1
                        await websocket.send_bytes(raw)
                    else:
                        # Decode for PCM test / browser clients
                        pcm = None
                        payload = raw[1:]
                        if len(payload) == 1920 * 4 and not payload.startswith(b"OggS"):
                            # Raw PCM from mock worker in unit tests
                            try:
                                pcm = np.frombuffer(payload, dtype=np.float32)
                            except Exception as pcm_err:
                                logger.debug(f"Raw PCM decode note: {pcm_err}")
                        elif server_opus_reader is not None:
                            try:
                                server_opus_reader.append_bytes(payload)
                                pcm = server_opus_reader.read_pcm()
                            except Exception as opus_err:
                                logger.debug(f"Opus decode note: {opus_err}")
                        elif len(payload) > 1:
                            try:
                                pcm = np.frombuffer(payload, dtype=np.float32)
                            except Exception as pcm_err:
                                logger.debug(f"Raw PCM decode note: {pcm_err}")

                        if pcm is not None and len(pcm) > 0:
                            resampled = out_resampler.resample_chunk(pcm, last=False)
                            clipped = soft_clip(resampled, threshold=0.92)
                            if codec == "g711_ulaw":
                                pcm_bytes = encode_ulaw(clipped)
                            else:
                                pcm_bytes = float32_to_int16(clipped).tobytes()
                            telemetry["traffic"]["gateway_to_browser"]["bytes"] += len(pcm_bytes)
                            telemetry["traffic"]["gateway_to_browser"]["frames"] += 1
                            await websocket.send_bytes(pcm_bytes)

                elif opcode == 0x02:
                    # Text token
                    last_agent_token_time = time.time()
                    token = raw[1:].decode("utf-8", errors="replace")
                    telemetry["tokens_received"] += 1
                    if not current_agent_turn_text:
                        current_agent_turn_start = (time.time() - call_start_time) * 1000.0
                    current_agent_turn_text.append(token)
                    if not current_agent_buffer:
                        current_agent_start_ms = (time.time() - call_start_time) * 1000.0
                    current_agent_buffer.append(token)

                    # Stream text token to client as JSON transcript and forward raw frame
                    await websocket.send_json({
                        "type": "transcript",
                        "role": "assistant",
                        "text": token,
                    })
                    if not is_pcm_client:
                        await websocket.send_bytes(raw)

                    # Feed to EndOfCallDetector
                    matched = detector.on_agent_token(token, time.time())
                    if matched:
                        detector.on_agent_speaking_stopped(time.time())

                elif opcode in (0x03, 0x04):
                    if not is_pcm_client:
                        await websocket.send_bytes(raw)

                elif opcode == 0x05:
                    err_text = raw[1:].decode("utf-8", errors="replace")
                    logger.warning(f"Worker emitted error: {err_text}")
                    await websocket.send_json({"type": "error", "message": f"Worker error: {err_text}"})
                    if not is_pcm_client:
                        await websocket.send_bytes(raw)
                    break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[VoiceSession {call_session_id}] worker_to_client_loop error: {e}", exc_info=True)
            trigger_stop(f"worker_loop_exc_{e}")
        else:
            if not stop_event.is_set():
                logger.warning(f"[VoiceSession {call_session_id}] Upstream worker_client message stream ended")
                trigger_stop("worker_stream_ended")
                try:
                    await websocket.send_json({
                        "type": "error",
                        "message": "PersonaPlex worker stream closed unexpectedly. Ensure worker on 127.0.0.1:8998 is running.",
                    })
                except Exception:
                    pass

    async def perform_session_rollover():
        """
        Seamless Session Rollover (5+ Minute Calls):
        Primes a fresh worker session before the 240s LM RingKVCache context limit is reached,
        carries forward conversation context in a compact rolling summary, and performs
        a seamless switch during a conversational pause without dropping the browser WebSocket.
        """
        nonlocal rollover_in_progress, session_rollover_count, current_worker_start_time
        if rollover_in_progress or stop_event.is_set():
            return
        rollover_in_progress = True
        session_rollover_count += 1
        elapsed_sec = time.time() - current_worker_start_time
        logger.info(
            f"[VoiceSession {call_session_id}] Initiating seamless session rollover #{session_rollover_count} "
            f"at elapsed {elapsed_sec:.1f}s (budget: {app_settings.ROLLOVER_BUDGET_SEC}s, threshold: {app_settings.ROLLOVER_THRESHOLD})"
        )

        try:
            # 1. Build rolling summary from recent turns
            recent_tokens = current_agent_turn_text[-app_settings.ROLLOVER_SUMMARY_MAX_WORDS:] if current_agent_turn_text else []
            summary_snippet = "".join(recent_tokens).strip()
            if not summary_snippet:
                summary_snippet = "Friendly, relaxed phone conversation in progress."

            # 2. Form continuation prompt for the fresh worker session
            continuation_prompt = (
                f"{compiled.formatted_prompt}\n"
                f"[System Note: Ongoing phone call continuation with the caller. Recent discussion: {summary_snippet}. "
                f"Continue speaking naturally without re-introducing yourself or saying goodbye.]"
            )
            continuation_persona = PersonaConfig(
                id=agent.id,
                name=agent.name,
                voice_prompt=resolved_voice,
                text_prompt=continuation_prompt,
                system_prompt=continuation_prompt,
                audio_temperature=float(resolved_audio_temp),
                text_temperature=float(resolved_text_temp),
                top_k_audio=int(resolved_audio_topk),
                top_k_text=int(resolved_text_topk),
            )

            # 3. Try to lease secondary worker from pool
            new_worker: PersonaPlexWorkerClient | None = None
            try:
                new_worker = await worker_pool.acquire_worker(
                    session_id=f"{call_session_id}_roll_{session_rollover_count}",
                    timeout=2.0,
                    probe_health=True,
                )
            except Exception as pool_err:
                logger.info(f"[VoiceSession {call_session_id}] Pool has no idle second worker: {pool_err}")
                new_worker = None

            if new_worker is not None:
                # Multi-worker background priming path
                logger.info(f"[VoiceSession {call_session_id}] Priming standby worker {new_worker.worker_id} in background...")
                await new_worker.connect(session_id=f"{call_session_id}_roll_{session_rollover_count}", persona=continuation_persona)
                logger.info(f"[VoiceSession {call_session_id}] Standby worker {new_worker.worker_id} primed and ready.")

                # Wait for conversational pause (both user and agent quiet for >= 300ms)
                for _ in range(40):
                    if (time.time() - last_user_speech_time > 0.3) and (time.time() - last_agent_token_time > 0.3):
                        break
                    await asyncio.sleep(0.1)

                # Swap workers smoothly
                old_worker = active_worker_holder[0]
                active_worker_holder[0] = new_worker

                # Start reader for new worker
                asyncio.create_task(stream_worker_frames(new_worker))

                # Release old worker
                if old_worker:
                    await old_worker.close(mark_idle=True)
                    await worker_pool.release_worker(old_worker.worker_id, success=True)
                    logger.info(f"[VoiceSession {call_session_id}] Released previous worker {old_worker.worker_id}")

            else:
                # Single-worker refresh path: wait for pause, reconnect worker with fresh RingKVCache
                logger.info(f"[VoiceSession {call_session_id}] Single-worker refresh path: awaiting conversational pause...")
                for _ in range(30):
                    if (time.time() - last_user_speech_time > 0.3) and (time.time() - last_agent_token_time > 0.3):
                        break
                    await asyncio.sleep(0.1)

                cur_worker = active_worker_holder[0]
                if cur_worker is not None:
                    await cur_worker.close(mark_idle=True)
                    await cur_worker.connect(
                        session_id=f"{call_session_id}_refresh_{session_rollover_count}",
                        persona=continuation_persona,
                    )
                    asyncio.create_task(stream_worker_frames(cur_worker))

            if is_pcm_client:
                try:
                    import sphn
                    server_opus_reader = sphn.OpusStreamReader(MODEL_SAMPLE_RATE)
                except Exception:
                    pass

            current_worker_start_time = time.time()
            logger.info(f"[VoiceSession {call_session_id}] Seamless rollover #{session_rollover_count} completed successfully.")

        except Exception as e:
            logger.error(f"[VoiceSession {call_session_id}] Rollover exception: {e}", exc_info=True)
        finally:
            rollover_in_progress = False

    async def lifecycle_heartbeat():
        try:
            while not stop_event.is_set():
                await asyncio.sleep(0.1)

                term_reason = detector.check_termination(time.time())
                if term_reason:
                    logger.warning(f"[VoiceSession {call_session_id}] EndOfCallDetector termination triggered: {term_reason}")
                    try:
                        await websocket.send_json({
                            "type": "call_ended",
                            "reason": term_reason,
                        })
                    except Exception:
                        pass
                    trigger_stop(term_reason)
                    break

                # Seamless Session Rollover check (trigger background rollover before 240s LM context limit)
                if (
                    app_settings.ROLLOVER_ENABLED
                    and not rollover_in_progress
                    and (time.time() - current_worker_start_time) >= (app_settings.ROLLOVER_BUDGET_SEC * app_settings.ROLLOVER_THRESHOLD)
                ):
                    asyncio.create_task(perform_session_rollover())

        except asyncio.CancelledError:
            pass

    # Flush any stale frames queued prior to session start
    while not audio_frame_queue.empty():
        try:
            audio_frame_queue.get_nowait()
        except asyncio.QueueEmpty:
            break

    # Run tasks concurrently
    tasks = [
        asyncio.create_task(client_to_queue_loop()),
        asyncio.create_task(worker_to_client_loop()),
        asyncio.create_task(lifecycle_heartbeat()),
    ]
    if is_pcm_client:
        tasks.append(asyncio.create_task(queue_to_worker_pacer()))

    stop_task = asyncio.create_task(stop_event.wait())
    all_tasks = [stop_task, *tasks]

    # Wait until stop_event or any task failure
    done, pending = await asyncio.wait(
        all_tasks,
        return_when=asyncio.FIRST_COMPLETED,
    )

    for d in done:
        if d.exception():
            logger.error(f"[VoiceSession {call_session_id}] Voice task exception: {d.exception()}", exc_info=d.exception())
        else:
            logger.info(f"[VoiceSession {call_session_id}] Voice task finished: {d}, disconnect_reason: {disconnect_reason}")

    for p in pending:
        p.cancel()
    await asyncio.gather(*pending, return_exceptions=True)

    # 9. Finalize Call Session in DB, Telemetry, and Conversation JSON file
    commit_agent_turn()
    duration_sec = max(0.0, time.time() - call_start_time)
    telemetry["duration_sec"] = round(duration_sec, 2)
    telemetry["close_reason"] = disconnect_reason
    telemetry["closed_by"] = "client" if ("client" in disconnect_reason or "user" in disconnect_reason) else "server"

    # Save complete conversation recording to JSON file
    convo_payload = {
        "session_id": call_session_id,
        "agent_id": agent.id,
        "agent_name": agent.name,
        "voice_id": resolved_voice,
        "status": "completed",
        "started_at": call_start_time,
        "ended_at": time.time(),
        "created_at_iso": datetime.datetime.fromtimestamp(call_start_time, tz=datetime.timezone.utc).isoformat(),
        "duration_sec": round(duration_sec, 2),
        "disconnect_reason": disconnect_reason,
        "closed_by": telemetry["closed_by"],
        "system_prompt": system_prompt,
        "formatted_prompt": getattr(compiled, "formatted_prompt", getattr(compiled, "text", "")),
        "turns": conversation_turns,
        "telemetry": telemetry,
    }
    for dest_dir in (CONVERSATIONS_DIR, LOGS_CONVERSATIONS_DIR):
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / f"{call_session_id}.json"
            with open(dest_file, "w", encoding="utf-8") as fp:
                json.dump(convo_payload, fp, indent=2)
            logger.info(f"[VoiceSession {call_session_id}] Saved conversation JSON to {dest_file}")
        except Exception as file_err:
            logger.warning(f"Failed to write conversation JSON to {dest_dir}: {file_err}")

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
    cur_w = active_worker_holder[0]
    if cur_w:
        call_success = disconnect_reason in ("normal", "user_hangup", "duration_limit", "silence_timeout", "agent_closed")
        await worker_pool.release_worker(cur_w.worker_id, success=call_success)

    try:
        close_code = status.WS_1000_NORMAL_CLOSURE
        if "exc" in disconnect_reason or "fail" in disconnect_reason or "worker_stream_ended" in disconnect_reason:
            close_code = status.WS_1011_INTERNAL_ERROR
        await websocket.close(code=close_code)
    except Exception:
        pass

