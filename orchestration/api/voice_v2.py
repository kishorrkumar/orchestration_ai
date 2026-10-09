"""
Lean S2S Voice Session Runtime Router (/v2/voice).
Handles full-duplex WebSocket streaming for 16 kHz web clients and 8 kHz telephony callers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any

import numpy as np
from fastapi import APIRouter, Header, HTTPException, Query, WebSocket, WebSocketDisconnect, status

from ..audio.codecs import decode_ulaw, encode_ulaw
from ..audio.detokenizer import detokenize_sentencepiece_stream, stitch_token
from ..audio.dsp import compute_rms, float32_to_int16, int16_to_float32, normalize_speech_loudness, soft_clip
from ..audio.framing import InboundAudioFrameProcessor
from ..audio.recorder import SessionAudioRecorder
from ..audio.resample import AudioResampler, StreamingResampleBuffer
from ..db.models import AgentVersion
from ..db.service import AgentService, CallSessionService
from ..db.session import get_session_factory
from ..persona.registry import OFFICIAL_VOICE_PRESETS, PersonaConfig
from ..pipeline.end_detector import EndOfCallDetector
from ..prompts.compiler import compile_prompt
from ..protocol.audio import (
    CLIENT_SAMPLE_RATE,
    MODEL_SAMPLE_RATE,
)
from ..protocol.messages import AudioMessage, ControlAction, TextMessage
from ..tts.voice_clone import default_voice_cloner
from ..worker.client import PersonaPlexWorkerClient
from ..worker.pool import WorkerPool

logger = logging.getLogger("orchestration.voice_v2")

router = APIRouter(tags=["Voice V2"])

# In-memory debug logs storage (ring buffer of recent sessions)
SESSION_DEBUG_LOGS: dict[str, dict[str, Any]] = {}
MAX_DEBUG_SESSIONS = 100


from ..settings import MODEL_FRAME_SAMPLES, MODEL_SAMPLE_RATE, app_settings

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
    record_session: bool = Query(True, description="Enable dual-channel 24kHz blackbox audio recording"),
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

    # Early Inbound Audio Buffer (Zero dropped frames during handshake / lease - Defect 10.4)
    inbound_prebuffer: asyncio.Queue = asyncio.Queue(maxsize=1000)
    early_disconnect = asyncio.Event()

    async def inbound_reader_task():
        try:
            while not early_disconnect.is_set():
                msg = await websocket.receive()
                if msg.get("type") == "websocket.disconnect":
                    early_disconnect.set()
                    await inbound_prebuffer.put(msg)
                    break
                await inbound_prebuffer.put(msg)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"[VoiceSession] Inbound reader notice: {e}")

    inbound_reader = asyncio.create_task(inbound_reader_task())
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
            inbound_reader.cancel()
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

    # PersonaPlex S2S worker pool
    worker_pool: WorkerPool = getattr(websocket.app.state, "pool", None)
    if not worker_pool:
        inbound_reader.cancel()
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
    agent_voice = target_ver.voice_id if target_ver else (getattr(agent, "draft_voice_id", None) or getattr(agent, "voice_id", None))
    resolved_voice = voice_prompt or voice_id or agent_voice or pipeline_cfg.get("voice_id") or pipeline_cfg.get("voice") or "NATM1.pt"

    from ..persona.registry import OFFICIAL_VOICE_PRESETS, get_existing_voice_files
    from ..tts.voice_clone import default_voice_cloner
    available_voices = get_existing_voice_files()
    raw_v = str(resolved_voice).strip() if resolved_voice else "NATM1.pt"
    clean_stem = raw_v.replace(".pt", "").replace(".wav", "")

    matched_voice = None
    # 1. Direct check against VoiceCloner artifacts
    if default_voice_cloner.has_voice(clean_stem) or default_voice_cloner.get_voice_path(clean_stem):
        matched_voice = f"{clean_stem}.wav"
    elif raw_v in available_voices:
        matched_voice = raw_v
    elif f"{raw_v}.wav" in available_voices:
        matched_voice = f"{raw_v}.wav"
    elif f"{raw_v}.pt" in available_voices:
        matched_voice = f"{raw_v}.pt"
    elif raw_v.endswith(".pt") and raw_v.replace(".pt", ".wav") in available_voices:
        matched_voice = raw_v.replace(".pt", ".wav")
    elif raw_v.endswith(".wav") and raw_v.replace(".wav", ".pt") in available_voices:
        matched_voice = raw_v.replace(".wav", ".pt")

    if matched_voice is None:
        # Determine appropriate gender-matching fallback instead of blindly picking available_voices[0] (which is NATF0.pt)
        v_meta = default_voice_cloner.get_voice_metadata(clean_stem)
        v_gender = (v_meta.get("gender") or v_meta.get("preferred_gender") or "").lower() if v_meta else ""

        is_female = "female" in v_gender
        if not is_female and agent and hasattr(agent, "name"):
            name_l = agent.name.lower()
            if any(fn in name_l for fn in ("ananya", "priya", "sarah", "emma", "maria", "elena")):
                is_female = True

        target_fallback = "NATF0.pt" if is_female else "NATM1.pt"
        if target_fallback in available_voices:
            matched_voice = target_fallback
        elif available_voices:
            matched_voice = available_voices[0]
        else:
            matched_voice = target_fallback
        logger.info(
            f"[VOICE ROUTING] Voice '{resolved_voice}' resolved to '{matched_voice}' "
            f"(gender: {'female' if is_female else 'male'})."
        )

    resolved_voice = matched_voice

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
    call_session_id = f"call_{int(call_start_time)}_{uuid.uuid4().hex[:6]}_{agent.id[:8]}"
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
        "timing": {
            "connect_ms": 0.0,
            "priming_time_ms": 0.0,
            "ttfa_greeting_ms": None,
            "first_token_ms": None,
            "turn_latencies_ms": [],
            "p50_turn_latency_ms": None,
            "p95_turn_latency_ms": None,
        },
        "performance": {
            "last_gpu_step_ms": 0.0,
            "gpu_real_time_factor": 0.0,
            "dropped_frames": 0,
            "max_queue_depth": 0,
        },
        "recording": {
            "enabled": record_session,
            "file_path": None,
            "duration_sec": 0.0,
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

    # Dual-channel blackbox audio recorder (Channel 0: Caller, Channel 1: Assistant)
    audio_recorder = SessionAudioRecorder(
        session_id=call_session_id,
        sample_rate=MODEL_SAMPLE_RATE,
        enabled=record_session,
    )

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

    # 5b. Conversation Turns & Instant Branded Greeting Playout (<150ms TTFA)
    conversation_turns: list[dict[str, Any]] = []
    initial_greeting_delivered = False
    user_has_responded = False


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

    # 7. Acquire Worker from Pool (with warm standby claim & fast capacity rejection)
    session_manager = getattr(websocket.app.state, "session_manager", None)
    standby_claimed_session = None
    claimed_from_standby = False
    worker_client: PersonaPlexWorkerClient | None = None

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

    if session_manager and session_manager.has_standby():
        try:
            candidate = await session_manager.create_session(
                persona=persona_cfg,
                session_id=call_session_id,
                timeout=0.1,
                use_standby=True,
            )
            if candidate.is_claimed_from_standby:
                standby_claimed_session = candidate
                claimed_from_standby = True
                worker_client = candidate.worker
                logger.info(f"[STANDBY] Fast-claimed warm standby worker {worker_client.worker_id} for session {call_session_id} (<10ms lease)!")
        except Exception as e:
            logger.debug(f"Standby claim bypass note: {e}")

    if worker_client is None:
        try:
            worker_client = await worker_pool.acquire_worker(session_id=call_session_id, timeout=2.0)
        except Exception as e:
            priming_stop.set()
            priming_ticker.cancel()
            inbound_reader.cancel()
            logger.warning(f"Fast capacity rejection for session {call_session_id}: {e}")
            telemetry["closed_by"] = "gateway"
            telemetry["close_reason"] = f"Worker acquire failed: {e}"
            await websocket.send_json({"type": "error", "message": f"Inference capacity exceeded. All workers busy: {e}"})
            await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER, reason="Inference capacity exceeded")
            return

        try:
            connect_task = asyncio.create_task(worker_client.connect(session_id=call_session_id, persona=persona_cfg))
            await connect_task
        except asyncio.CancelledError:
            logger.info("Worker connect cancelled due to early client disconnect.")
            telemetry["closed_by"] = "client"
            telemetry["close_reason"] = "client_disconnected_during_priming"
            inbound_reader.cancel()
            await worker_pool.release_worker(worker_client.worker_id, success=True)
            return
        except Exception as e:
            priming_stop.set()
            priming_ticker.cancel()
            inbound_reader.cancel()
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
    else:
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
    current_agent_turn_text: list[str] = []
    current_agent_turn_start = 0.0

    current_agent_buffer: list[str] = []
    current_agent_start_ms: float = 0.0
    current_agent_turn_start_time: float = 0.0
    agent_turn_yielded: bool = False
    last_user_speech_end_time: float = 0.0
    turn_latencies: list[float] = []

    def commit_agent_turn():
        nonlocal current_agent_buffer, current_agent_start_ms, current_agent_turn_start_time
        if current_agent_buffer:
            text = detokenize_sentencepiece_stream(current_agent_buffer, agent_name=agent.name).strip()
            now_ms = (time.time() - call_start_time) * 1000.0
            if text:
                conversation_turns.append({
                    "role": "assistant",
                    "text": text,
                    "started_ms": round(current_agent_start_ms, 1),
                    "ended_ms": round(now_ms, 1),
                    "duration_ms": round(now_ms - current_agent_start_ms, 1),
                    "time_offset_sec": round(current_agent_start_ms / 1000.0, 2),
                    "timestamp": time.time(),
                    "words": len(text.split()),
                })
            current_agent_buffer.clear()
            current_agent_turn_start_time = 0.0

    is_pcm_client = (codec == "pcm16" or codec == "g711_ulaw")

    # Inbound / Outbound Opus Transcoding:
    # WorkerClient is the ONE codec owner for outbound worker streaming (PCM -> Opus).
    # server_opus_reader is used here ONLY to decode incoming worker Opus into PCM for browser/telephony clients.
    server_opus_reader = None
    sphn_installed = False
    sphn_version = "not_found"
    try:
        import sphn
        sphn_installed = True
        sphn_version = getattr(sphn, "__version__", "installed")
        if is_pcm_client:
            server_opus_reader = sphn.OpusStreamReader(MODEL_SAMPLE_RATE)
            logger.info(f"[VoiceSession {call_session_id}] Initialized server_opus_reader @ {MODEL_SAMPLE_RATE} Hz")
    except Exception as e:
        logger.warning(f"[VoiceSession {call_session_id}] sphn OpusStreamReader unavailable on gateway: {e}")

    logger.info(
        f"[VoiceSession {call_session_id}] Audio runtime configured: client_codec={codec}, "
        f"is_pcm_client={is_pcm_client}, sphn_available={sphn_installed} ({sphn_version}), "
        f"worker_use_opus={getattr(worker_client, 'use_opus', False)}"
    )

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
        nonlocal last_user_speech_time, last_user_speech_end_time, agent_turn_yielded, user_has_responded
        try:
            while not stop_event.is_set():
                raw = await inbound_prebuffer.get()
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
                                user_has_responded = True
                                last_user_speech_time = time.time()
                                detector.on_user_speech(last_user_speech_time)
                                if agent_turn_yielded:
                                    agent_turn_yielded = False
                                if current_agent_buffer:
                                    commit_agent_turn()
                                    agent_turn_yielded = False
                            for frame_24k in frames_24k:
                                audio_recorder.record_inbound(frame_24k)
                                await audio_frame_queue.put(frame_24k)
                        elif event_type == "interrupt":
                            logger.info(f"[VoiceSession {call_session_id}] User binary interrupt noted (handled locally)")
                            commit_agent_turn()
                            agent_turn_yielded = False

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
                            commit_agent_turn()
                            agent_turn_yielded = False
                        elif mtype in ("user_transcript", "transcript"):
                            user_text = msg_json.get("text", "").strip()
                            is_final = bool(msg_json.get("is_final", True))
                            if user_text:
                                user_has_responded = True
                                last_user_speech_time = time.time()
                                last_user_speech_end_time = time.time()
                                detector.on_user_speech(last_user_speech_time)
                                if agent_turn_yielded:
                                    agent_turn_yielded = False
                                commit_agent_turn()

                                # Deduplicate partial turns: if preceding turn was user and unfinalized, update in place
                                if conversation_turns and conversation_turns[-1].get("role") == "user" and not conversation_turns[-1].get("is_final", True):
                                    conversation_turns[-1]["text"] = user_text
                                    conversation_turns[-1]["is_final"] = is_final
                                    conversation_turns[-1]["ended_ms"] = round((time.time() - call_start_time) * 1000.0, 1)
                                else:
                                    now_ms = (time.time() - call_start_time) * 1000.0
                                    conversation_turns.append({
                                        "role": "user",
                                        "text": user_text,
                                        "is_final": is_final,
                                        "started_ms": round(now_ms, 1),
                                        "ended_ms": round(now_ms, 1),
                                        "time_offset_sec": round(time.time() - call_start_time, 2),
                                        "timestamp": time.time(),
                                    })
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
        Uses a monotonic deadline clock: next_deadline += 0.080; sleep(max(0, next_deadline - now)).
        Sends EXACTLY one 1920-sample float32 frame per tick (real frame from queue if available, else zeros).
        WorkerClient is the single codec owner for Opus transcoding via send_audio().
        """
        frame_interval = 0.080  # 80ms = 12.5 Hz
        next_deadline = time.monotonic()
        pacer_ticks = 0
        pacer_start_time = time.monotonic()
        pacer_frames_sent = 0
        pacer_bytes_sent = 0

        logger.info(f"[VoiceSession {call_session_id}] queue_to_worker_pacer started with monotonic deadline clock (12.5 Hz)")
        try:
            while not stop_event.is_set():
                next_deadline += frame_interval

                # Bound check: allow up to 20 frames (1.6s) jitter buffer before dropping oldest frames
                q_size = audio_frame_queue.qsize()
                if q_size > telemetry["performance"]["max_queue_depth"]:
                    telemetry["performance"]["max_queue_depth"] = q_size

                if q_size > 20:
                    dropped = 0
                    while audio_frame_queue.qsize() > 10:
                        try:
                            audio_frame_queue.get_nowait()
                            dropped += 1
                        except asyncio.QueueEmpty:
                            break
                    if dropped > 0:
                        telemetry["performance"]["dropped_frames"] += dropped
                        logger.warning(
                            f"[VoiceSession {call_session_id}] Audio queue backlog ({q_size} frames): "
                            f"dropped {dropped} oldest frames to preserve real-time lockstep"
                        )

                frame_to_send: np.ndarray | None = None
                try:
                    frame_to_send = audio_frame_queue.get_nowait()
                except asyncio.QueueEmpty:
                    frame_to_send = np.zeros(MODEL_FRAME_SAMPLES, dtype=np.float32)

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
                    payload_bytes = await cur_worker.send_audio(frame_to_send)
                    if payload_bytes > 0:
                        pacer_bytes_sent += payload_bytes
                        pacer_frames_sent += 1
                        telemetry["traffic"]["gateway_to_worker"]["bytes"] += payload_bytes
                        telemetry["traffic"]["gateway_to_worker"]["frames"] += 1
                    if cur_worker.last_frame_step_ms > 0:
                        telemetry["performance"]["last_gpu_step_ms"] = cur_worker.last_frame_step_ms
                        telemetry["performance"]["gpu_real_time_factor"] = round(cur_worker.last_frame_step_ms / 80.0, 3)

                pacer_ticks += 1
                if pacer_ticks % 50 == 0:  # Every ~4s
                    elapsed = time.monotonic() - pacer_start_time
                    fps = pacer_ticks / elapsed if elapsed > 0 else 0.0
                    logger.info(
                        f"[VoiceSession {call_session_id}] Pacer tick #{pacer_ticks}: {fps:.2f} ticks/s (target 12.5), "
                        f"frames_sent={pacer_frames_sent}, bytes_sent={pacer_bytes_sent}, queue={audio_frame_queue.qsize()}"
                    )

                now = time.monotonic()
                if next_deadline < now - frame_interval:
                    # Prevent catch-up bursts if loop execution fell behind
                    next_deadline = now
                sleep_time = max(0.0, next_deadline - now)
                await asyncio.sleep(sleep_time)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[VoiceSession {call_session_id}] queue_to_worker_pacer error: {e}", exc_info=True)
            trigger_stop(f"pacer_exc_{e}")

    async def worker_to_client_loop():
        nonlocal disconnect_reason, turn_counter, current_agent_turn_text, current_agent_turn_start
        nonlocal first_agent_audio_emitted, ttfa_greeting_ms, last_agent_token_time, user_has_responded
        nonlocal last_user_speech_end_time, agent_turn_yielded, turn_latencies
        worker_audio_frame_count = 0
        decode_err_count = 0
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
                    if agent_turn_yielded:
                        # Turn yielded to caller; suppress further assistant audio until user responds
                        continue

                    last_agent_token_time = time.time()
                    worker_audio_frame_count += 1
                    payload = raw[1:]
                    telemetry["worker_opus_bytes_in"] = telemetry.get("worker_opus_bytes_in", 0) + len(payload)
                    telemetry["traffic"]["worker_to_gateway"]["bytes"] += len(raw)
                    telemetry["traffic"]["worker_to_gateway"]["frames"] += 1

                    # Log the first 5 payload headers for immediate wire diagnostics
                    if worker_audio_frame_count <= 5:
                        header_hex = payload[:16].hex() if len(payload) >= 16 else payload.hex()
                        logger.info(
                            f"[VoiceSession {call_session_id}] Worker frame #{worker_audio_frame_count}: "
                            f"len={len(payload)}B, header={header_hex}"
                        )

                    if not is_pcm_client:
                        # Transparent binary forward directly to client
                        telemetry["traffic"]["gateway_to_browser"]["bytes"] += len(raw)
                        telemetry["traffic"]["gateway_to_browser"]["frames"] += 1
                        await websocket.send_bytes(raw)
                    else:
                        # Decode Opus / raw PCM for PCM browser / test clients
                        pcm = None
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
                                decode_err_count += 1
                                telemetry["decode_errors"] = telemetry.get("decode_errors", 0) + 1
                                if decode_err_count <= 10:
                                    logger.warning(
                                        f"[VoiceSession {call_session_id}] Opus decode exception #{decode_err_count}: {opus_err}",
                                        exc_info=True,
                                    )
                        elif len(payload) > 1:
                            try:
                                pcm = np.frombuffer(payload, dtype=np.float32)
                            except Exception as pcm_err:
                                logger.debug(f"Raw PCM decode note: {pcm_err}")

                        # Only proceed if we received actual decoded audio samples
                        if pcm is not None and len(pcm) > 0:
                            telemetry["decoded_pcm_frames"] = telemetry.get("decoded_pcm_frames", 0) + 1
                            rms_val = compute_rms(pcm)
                            peak_val = float(np.max(np.abs(pcm)))
                            telemetry["audio_levels"]["worker_to_gateway_rms"] = round(
                                telemetry["audio_levels"]["worker_to_gateway_rms"] * 0.9 + rms_val * 0.1, 4
                            )
                            telemetry["audio_levels"]["worker_to_gateway_peak"] = round(
                                max(telemetry["audio_levels"]["worker_to_gateway_peak"], peak_val), 4
                            )

                            if not first_agent_audio_emitted:
                                first_agent_audio_emitted = True
                                first_audio_ms = round((time.time() - call_start_time) * 1000.0, 1)
                                ttfa_greeting_ms = first_audio_ms
                                telemetry["ttfa_greeting_ms"] = ttfa_greeting_ms
                                telemetry["timing"]["ttfa_greeting_ms"] = ttfa_greeting_ms
                                telemetry["first_audio_to_browser_ms"] = first_audio_ms
                                logger.info(
                                    f"[VoiceSession {call_session_id}] FIRST AUDIBLE AUDIO EMITTED: "
                                    f"samples={len(pcm)}, rms={rms_val:.4f}, ttfa={ttfa_greeting_ms}ms"
                                )

                            # Turn latency measurement: user speech end -> first agent audio of reply
                            if last_user_speech_end_time > 0:
                                t_lat = round((time.time() - last_user_speech_end_time) * 1000.0, 1)
                                if 0 < t_lat < 10000.0:
                                    turn_latencies.append(t_lat)
                                    telemetry["timing"]["turn_latencies_ms"] = turn_latencies[-20:]
                                    if len(turn_latencies) >= 2:
                                        telemetry["timing"]["p50_turn_latency_ms"] = round(float(np.percentile(turn_latencies, 50)), 1)
                                        telemetry["timing"]["p95_turn_latency_ms"] = round(float(np.percentile(turn_latencies, 95)), 1)
                                last_user_speech_end_time = 0.0

                            # Loudness normalization to -16 LUFS (speech RMS ~0.12) with AGC & soft limiting
                            normalized_pcm = normalize_speech_loudness(pcm, target_rms=0.12)

                            # Record assistant speech into dual-channel blackbox recorder (Channel 1)
                            audio_recorder.record_outbound(normalized_pcm)

                            resampled = out_resampler.resample_chunk(normalized_pcm, last=False)
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
                    if initial_greeting_delivered and not user_has_responded:
                        if (time.time() - call_stream_start) < 6.0:
                            # Branded greeting is active; suppress early pretraining text tokens from worker
                            continue
                        else:
                            user_has_responded = True

                    if agent_turn_yielded:
                        # Suppress additional tokens if turn already yielded
                        continue

                    last_agent_token_time = time.time()
                    token = raw[1:].decode("utf-8", errors="replace")
                    telemetry["tokens_received"] += 1
                    if "first_token_ms" not in telemetry or telemetry["timing"]["first_token_ms"] is None:
                        t_tok = round((time.time() - call_start_time) * 1000.0, 1)
                        telemetry["first_token_ms"] = t_tok
                        telemetry["timing"]["first_token_ms"] = t_tok
                        logger.info(f"[VoiceSession {call_session_id}] First text token received at {t_tok}ms: '{token}'")

                    if not current_agent_turn_text:
                        current_agent_turn_start = (time.time() - call_start_time) * 1000.0
                    current_agent_turn_text.append(token)
                    if not current_agent_buffer:
                        current_agent_start_ms = (time.time() - call_start_time) * 1000.0
                        current_agent_turn_start_time = time.time()
                    current_agent_buffer.append(token)

                    # Stream text token to client as JSON transcript and forward raw frame
                    await websocket.send_json({
                        "type": "transcript",
                        "role": "assistant",
                        "text": token,
                    })
                    if not is_pcm_client:
                        await websocket.send_bytes(raw)

                    # Monologue Guard & Question Turn Yield
                    current_stitched = detokenize_sentencepiece_stream(current_agent_buffer, agent_name=agent.name)
                    current_words = current_stitched.split()
                    turn_elapsed_sec = time.time() - current_agent_turn_start_time

                    should_yield = False
                    # A) If agent finishes a question ("?" with >= 4 words), yield immediately to caller
                    if "?" in current_stitched and len(current_words) >= 4:
                        should_yield = True
                    # B) If agent exceeds 25 words or 11 seconds and reaches sentence boundary (. / ! / ?)
                    elif (len(current_words) >= 25 or turn_elapsed_sec >= 11.0) and current_stitched.endswith((".", "!", "?")):
                        should_yield = True
                    # C) Hard cutoff at 12 seconds continuous speech
                    elif turn_elapsed_sec >= 12.0 and len(current_words) >= 8:
                        should_yield = True

                    if should_yield:
                        logger.info(
                            f"[VoiceSession {call_session_id}] Monologue guard triggered: yielding turn to caller. "
                            f"words={len(current_words)}, duration={turn_elapsed_sec:.1f}s, text='{current_stitched[-60:]}'"
                        )
                        commit_agent_turn()
                        agent_turn_yielded = True

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

    def make_task_logger(task_name: str):
        def _done_cb(t: asyncio.Task):
            try:
                if not t.cancelled():
                    exc = t.exception()
                    if exc:
                        logger.error(
                            f"[VoiceSession {call_session_id}] Background task '{task_name}' failed: {exc}",
                            exc_info=exc,
                        )
                        trigger_stop(f"task_{task_name}_failed_{exc}")
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"[VoiceSession {call_session_id}] Task inspector error ({task_name}): {e}")
        return _done_cb

    # Run tasks concurrently with error tracking
    t_client = asyncio.create_task(client_to_queue_loop())
    t_client.add_done_callback(make_task_logger("client_to_queue_loop"))

    t_worker = asyncio.create_task(worker_to_client_loop())
    t_worker.add_done_callback(make_task_logger("worker_to_client_loop"))

    t_heartbeat = asyncio.create_task(lifecycle_heartbeat())
    t_heartbeat.add_done_callback(make_task_logger("lifecycle_heartbeat"))

    worker_stream_task.add_done_callback(make_task_logger("stream_worker_frames"))

    tasks = [worker_stream_task, t_client, t_worker, t_heartbeat]
    if is_pcm_client:
        t_pacer = asyncio.create_task(queue_to_worker_pacer())
        t_pacer.add_done_callback(make_task_logger("queue_to_worker_pacer"))
        tasks.append(t_pacer)

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

    # Finalize dual-channel session audio recording
    rec_summary = audio_recorder.close()
    telemetry["recording"] = rec_summary

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
        "recording_file": rec_summary.get("file_path"),
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
                full_turn_text = detokenize_sentencepiece_stream(current_agent_turn_text).strip()
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

    # 10. Clean up worker, inbound reader & connection
    inbound_reader.cancel()
    if standby_claimed_session and session_manager:
        await session_manager.end_session(call_session_id)
    else:
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

