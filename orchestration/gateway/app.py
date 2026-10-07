"""
FastAPI Orchestration Gateway for NVIDIA PersonaPlex.

Provides:
- REST API for Agent Persona management (/v1/agents)
- REST API for Worker Pool management (/v1/workers)
- REST API for Session observability & history (/v1/sessions)
- Real-time Full-Duplex WebSocket gateway (/v1/realtime)
- Operational Health & Prometheus Metrics (/healthz, /metrics)
- Built-in Developer Console (/console)
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ..config import settings
from ..persona.prompts import build_system_prompt
from ..persona.registry import (
    OFFICIAL_VOICE_PRESETS,
    PersonaConfig,
    PersonaRegistry,
    default_registry,
)
from ..protocol.messages import (
    ErrorMessage,
    HandshakeMessage,
    MetadataMessage,
    encode_message,
)
from ..rag.engine import default_rag_engine
from ..session.manager import SessionManager, SessionState
from ..tts.voice_clone import default_voice_cloner
from ..worker.pool import PoolCapacityExceededError, WorkerNodeConfig, WorkerPool
from .lean_studio_ui import LEAN_STUDIO_HTML
from .security import default_rate_limiter
from .studio_ui import STUDIO_HTML

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("orchestration.gateway")

VOICE_METADATA = [
    # Natural Female
    {"id": "NATF0.pt", "name": "NATF0", "gender": "Female", "category": "Natural", "tag": "Warm & Calm", "description": "Soft-spoken and gentle teacher demeanor"},
    {"id": "NATF1.pt", "name": "NATF1", "gender": "Female", "category": "Natural", "tag": "Crisp & Professional", "description": "Customer service, clear diction"},
    {"id": "NATF2.pt", "name": "NATF2", "gender": "Female", "category": "Natural", "tag": "Expressive & Friendly", "description": "Default assistant voice, engaging tone"},
    {"id": "NATF3.pt", "name": "NATF3", "gender": "Female", "category": "Natural", "tag": "Bright & Articulate", "description": "Academic explanations and coaching"},
    # Natural Male
    {"id": "NATM0.pt", "name": "NATM0", "gender": "Male", "category": "Natural", "tag": "Deep & Authoritative", "description": "Commanding tone, corporate advisor"},
    {"id": "NATM1.pt", "name": "NATM1", "gender": "Male", "category": "Natural", "tag": "Warm & Narrative", "description": "Drive-through host, conversational guide"},
    {"id": "NATM2.pt", "name": "NATM2", "gender": "Male", "category": "Natural", "tag": "Technical & Energetic", "description": "Sales engineer, drone specialist"},
    {"id": "NATM3.pt", "name": "NATM3", "gender": "Male", "category": "Natural", "tag": "Casual & Direct", "description": "Informal companion, quick answers"},
    # Variety Female
    {"id": "VARF0.pt", "name": "VARF0", "gender": "Female", "category": "Variety", "tag": "Dynamic Storyteller", "description": "High expressive dynamic range"},
    {"id": "VARF1.pt", "name": "VARF1", "gender": "Female", "category": "Variety", "tag": "Upbeat Presenter", "description": "Podcast host, cheerful guide"},
    {"id": "VARF2.pt", "name": "VARF2", "gender": "Female", "category": "Variety", "tag": "Thoughtful Analyst", "description": "Deliberate and reflective pacing"},
    {"id": "VARF3.pt", "name": "VARF3", "gender": "Female", "category": "Variety", "tag": "Melodic & Serene", "description": "Mindfulness, calming meditation"},
    {"id": "VARF4.pt", "name": "VARF4", "gender": "Female", "category": "Variety", "tag": "Vibrant Actor", "description": "Roleplaying scenarios, distinct inflections"},
    # Variety Male
    {"id": "VARM0.pt", "name": "VARM0", "gender": "Male", "category": "Variety", "tag": "Radio Announcer", "description": "Broadcaster resonance and clarity"},
    {"id": "VARM1.pt", "name": "VARM1", "gender": "Male", "category": "Variety", "tag": "Emergency / Space", "description": "Mars mission astronaut, urgent dramatic tone"},
    {"id": "VARM2.pt", "name": "VARM2", "gender": "Male", "category": "Variety", "tag": "Empathetic Listener", "description": "Supportive and comforting timbre"},
    {"id": "VARM3.pt", "name": "VARM3", "gender": "Male", "category": "Variety", "tag": "Rapid Tech Host", "description": "Fast-paced, conversational enthusiasm"},
    {"id": "VARM4.pt", "name": "VARM4", "gender": "Male", "category": "Variety", "tag": "Smooth Baritone", "description": "Late-night radio, deep soothing tone"},
]


class RAGTextUpload(BaseModel):
    title: str = Field(..., description="Document title or filename")
    text: str = Field(..., description="Raw text content")


class RAGQueryRequest(BaseModel):
    query: str = Field(..., description="Text query to search knowledge base")
    top_k: int = Field(default=3, ge=1, le=10)


def create_app(
    pool: WorkerPool | None = None,
    registry: PersonaRegistry | None = None,
    session_manager: SessionManager | None = None,
    worker_type: str = "auto",
    active_server = None,
) -> FastAPI:
    import os

    from fastapi import Response
    worker_pool = pool or WorkerPool()
    persona_registry = registry or default_registry
    mgr = session_manager or SessionManager(pool=worker_pool)

    # Resolve active server / worker type
    resolved_server = active_server

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal resolved_server
        logger.info("PersonaPlex Orchestration Gateway starting up")
        server_to_stop: Any = None

        # Only auto-spawn a local worker if pool has no workers registered
        if len(worker_pool._workers) == 0:
            target_type = os.environ.get("PERSONAPLEX_WORKER_TYPE", worker_type if worker_type != "auto" else "mock").lower()
            if target_type == "cascaded":
                from ..worker.cascaded_worker import CascadedLocalWorkerServer
                try:
                    cascaded_server = CascadedLocalWorkerServer(host="127.0.0.1", port=8998)
                    await cascaded_server.start()
                    server_to_stop = cascaded_server
                    resolved_server = cascaded_server
                    logger.info("Cascaded local worker auto-started on ws://127.0.0.1:8998")
                except OSError as e:
                    logger.info("Cascaded worker port 8998 already active (%s); reusing running worker", e)
                mock_cfg = WorkerNodeConfig(id="cascaded-worker-1", host="127.0.0.1", port=8998)
                worker_pool.register_worker(mock_cfg)
            else:
                from ..worker.mock_worker import PersonaPlexMockServer
                try:
                    mock_server = PersonaPlexMockServer(host="127.0.0.1", port=8998)
                    await mock_server.start()
                    server_to_stop = mock_server
                    resolved_server = mock_server
                    logger.info("Mock worker auto-started on ws://127.0.0.1:8998")
                except OSError as e:
                    logger.info("Mock worker port 8998 already active (%s); reusing running worker", e)
                mock_cfg = WorkerNodeConfig(id="mock-worker-1", host="127.0.0.1", port=8998)
                worker_pool.register_worker(mock_cfg)

        # Database initialization & seeding for Voice Agent Platform
        try:
            from ..db.seed import seed_database
            from ..db.session import get_session_factory, init_db
            await init_db()
            session_factory = get_session_factory()
            async with session_factory() as session:
                await seed_database(session)
            logger.info("Database initialized and seeded successfully")
        except Exception as e:
            logger.warning("Database init notice: %s", e)

        yield
        logger.info("PersonaPlex Orchestration Gateway shutting down")
        if server_to_stop:
            await server_to_stop.stop()

    app = FastAPI(
        title="PersonaPlex Realtime Voice Orchestration Gateway",
        description="Open-source full-duplex orchestration layer for NVIDIA PersonaPlex speech-to-speech agents",
        version="0.1.0",
        lifespan=lifespan,
    )

    from .security import SecurityMiddleware
    app.add_middleware(SecurityMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount S2S Voice Agent Platform API routers
    from ..api.agents import router as agents_router
    from ..api.prompts import router as prompts_router
    from ..api.voice_v2 import router as voice_v2_router
    app.include_router(agents_router)
    app.include_router(prompts_router)
    app.include_router(voice_v2_router)

    # Mount V2 Clean Architecture routers & RFC 9457 Problem Details error handlers
    from ..interfaces.http.error_handlers import register_error_handlers
    from ..interfaces.http.routers.agents import router as agents_v2_router
    from ..interfaces.http.routers.calls import router as calls_v2_router
    from ..interfaces.http.routers.health import router as health_v2_router
    from ..interfaces.http.routers.prompts import router as prompts_v2_router
    from ..interfaces.http.routers.providers import router as providers_v2_router

    register_error_handlers(app)
    app.include_router(agents_v2_router)
    app.include_router(calls_v2_router)
    app.include_router(prompts_v2_router)
    app.include_router(health_v2_router)
    app.include_router(providers_v2_router)

    # Attach instances to app state for test inspection
    app.state.pool = worker_pool
    app.state.registry = persona_registry
    app.state.session_manager = mgr

    # ==========================================================
    # Health & Operational Metrics
    # ==========================================================

    @app.get("/healthz", tags=["System"])
    async def health_check():
        pool_stats = worker_pool.get_stats()
        active_sessions = len(mgr.list_active_sessions())
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "active_sessions": active_sessions,
            "pool": pool_stats,
        }

    @app.get("/metrics", tags=["System"])
    async def get_metrics():
        try:
            from ..telemetry.gpu import get_live_gpu_telemetry
            gpu = get_live_gpu_telemetry()
            active = mgr.list_active_sessions()
            pool_stats = worker_pool.get_stats()
            total_user_frames = sum(s.get("user_frames_in", 0) for s in active)
            total_agent_frames = sum(s.get("agent_frames_out", 0) for s in active)
            total_barge_ins = sum(s.get("barge_in_events", 0) for s in active)

            stt_ms = getattr(resolved_server, "last_stt_ms", 0.0) if resolved_server else 0.0
            llm_ttft_ms = getattr(resolved_server, "last_llm_ttft_ms", 0.0) if resolved_server else 0.0
            tts_ttfa_ms = getattr(resolved_server, "last_tts_ttfa_ms", 0.0) if resolved_server else 0.0
            total_ttfa_ms = getattr(resolved_server, "last_total_ttfa_ms", 0.0) if resolved_server else 0.0
            worker_underruns = getattr(resolved_server, "total_underruns", 0) if resolved_server else 0
            if resolved_server:
                wtype = getattr(resolved_server, "__class__", type).__name__
            elif pool_stats["total_workers"] > 0:
                wtype = "PersonaPlex 7B (GPU)"
            else:
                wtype = "Mock"

            latest_session = active[-1] if active else (mgr.list_recent_history(1)[-1] if mgr.list_recent_history(1) else {})
            priming_time_ms = round(latest_session.get("handshake_time_sec", 0.0) * 1000, 1)
            ttfa_ms = round(latest_session.get("ttfa_sec", 0.0) * 1000, 1) if latest_session.get("ttfa_sec") else total_ttfa_ms

            # Collect live per-frame step time across registered workers
            frame_step_ms = 0.0
            for w in worker_pool._workers.values():
                if hasattr(w, "last_frame_step_ms") and w.last_frame_step_ms > 0:
                    frame_step_ms = w.last_frame_step_ms
                    break

            return {
                "timestamp": time.time(),
                "worker_type": wtype,
                "workers_total": pool_stats["total_workers"],
                "workers_idle": pool_stats["idle_workers"],
                "workers_busy": pool_stats["busy_workers"],
                "active_sessions_count": len(active),
                "total_user_frames_in": total_user_frames,
                "total_agent_frames_out": total_agent_frames,
                "total_barge_in_events": total_barge_ins,
                "priming_time_ms": priming_time_ms,
                "ttfa_ms": round(ttfa_ms, 1),
                "frame_step_ms": round(frame_step_ms, 1),
                "standby_ready": mgr.has_standby(),
                "stt_ms": round(stt_ms, 1),
                "llm_ttft_ms": round(llm_ttft_ms, 1),
                "tts_ttfa_ms": round(tts_ttfa_ms, 1),
                "total_ttfa_ms": round(total_ttfa_ms, 1),
                "worker_underruns": worker_underruns,
                "gpu": gpu,
            }
        except Exception as e:
            logger.warning(f"Error serving /metrics: {e}")
            return {
                "timestamp": time.time(),
                "worker_type": "PersonaPlex",
                "workers_total": 1,
                "workers_idle": 1,
                "workers_busy": 0,
                "active_sessions_count": 0,
                "total_user_frames_in": 0,
                "total_agent_frames_out": 0,
                "total_barge_in_events": 0,
                "priming_time_ms": 0.0,
                "ttfa_ms": 0.0,
                "frame_step_ms": 0.0,
                "standby_ready": False,
                "stt_ms": 0.0,
                "llm_ttft_ms": 0.0,
                "tts_ttfa_ms": 0.0,
                "total_ttfa_ms": 0.0,
                "worker_underruns": 0,
                "gpu": {},
            }

    @app.post("/v1/sessions/preprime", tags=["Sessions"])
    @app.get("/v1/sessions/preprime", tags=["Sessions"])
    async def preprime_standby_session(persona_id: str = "casual_friend"):
        """Pre-primes a standby session in the background so incoming calls connect in 0ms (<1.5s TTFA)."""
        persona = persona_registry.get(persona_id) or list(persona_registry.list_all())[0]
        standby = await mgr.preprime_standby(persona)
        return {
            "status": "ready" if (standby and standby.state == SessionState.ACTIVE) else "preparing",
            "is_standby_ready": mgr.has_standby(),
            "persona_id": persona.id,
            "voice": persona.get_normalized_voice_prompt(),
        }

    # ==========================================================
    # Audio Cleaner & A/B Recording Endpoints
    # ==========================================================

    @app.get("/v1/audio/raw/{session_id}", tags=["Audio"])
    async def get_raw_audio(session_id: str):
        session = mgr.get_session(session_id)
        if not session or not hasattr(session, "cleaner"):
            raise HTTPException(status_code=404, detail="Session or audio not found")
        data = session.cleaner.get_raw_wav_bytes()
        if not data:
            raise HTTPException(status_code=404, detail="No audio recorded yet")
        return Response(content=data, media_type="audio/wav", headers={"Content-Disposition": f"attachment; filename=raw_{session_id}.wav"})

    @app.get("/v1/audio/clean/{session_id}", tags=["Audio"])
    async def get_clean_audio(session_id: str):
        session = mgr.get_session(session_id)
        if not session or not hasattr(session, "cleaner"):
            raise HTTPException(status_code=404, detail="Session or audio not found")
        data = session.cleaner.get_clean_wav_bytes()
        if not data:
            raise HTTPException(status_code=404, detail="No audio recorded yet")
        return Response(content=data, media_type="audio/wav", headers={"Content-Disposition": f"attachment; filename=clean_{session_id}.wav"})

    @app.post("/v1/audio/toggle-bypass/{session_id}", tags=["Audio"])
    async def toggle_bypass(session_id: str):
        session = mgr.get_session(session_id)
        if not session or not hasattr(session, "cleaner"):
            raise HTTPException(status_code=404, detail="Session not found")
        session.cleaner.bypass = not session.cleaner.bypass
        return {"session_id": session_id, "bypass": session.cleaner.bypass}

    @app.post("/v1/audio/mode/{session_id}", tags=["Audio"])
    async def set_audio_mode(session_id: str, mode: str = "isolation"):
        session = mgr.get_session(session_id)
        if not session or not hasattr(session, "cleaner"):
            raise HTTPException(status_code=404, detail="Session not found")
        session.cleaner.set_mode(mode)
        return {"session_id": session_id, "mode": session.cleaner.mode}

    # ==========================================================
    # Persona & Agent Catalog
    # ==========================================================

    def _validate_persona(p: PersonaConfig) -> None:
        voice = p.get_normalized_voice_prompt()
        if voice not in OFFICIAL_VOICE_PRESETS and not default_voice_cloner.has_voice(voice):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid voice '{voice}'. Must be an official preset ({', '.join(OFFICIAL_VOICE_PRESETS)}) or a registered cloned voice."
            )
        try:
            from ..persona.registry import validate_system_prompt
            validate_system_prompt(p.system_prompt or p.text_prompt or "")
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    @app.get("/v1/personas", tags=["Personas"])
    async def list_personas():
        return {
            "personas": [p.model_dump() for p in persona_registry.list_all()],
            "voice_presets": OFFICIAL_VOICE_PRESETS,
        }

    @app.get("/v1/personas/{persona_id}", tags=["Personas"])
    async def get_persona(persona_id: str):
        agent = persona_registry.get(persona_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Persona '{persona_id}' not found")
        return agent.model_dump()

    @app.post("/v1/personas", tags=["Personas"], status_code=status.HTTP_201_CREATED)
    async def register_persona(persona: PersonaConfig):
        _validate_persona(persona)
        persona_registry.register(persona)
        return {"status": "created", "persona": persona.model_dump()}

    @app.put("/v1/personas/{persona_id}", tags=["Personas"])
    async def update_persona(persona: PersonaConfig, persona_id: str):
        persona.id = persona_id
        _validate_persona(persona)
        persona_registry.register(persona)
        return {"status": "updated", "persona": persona.model_dump()}

    @app.delete("/v1/personas/{persona_id}", tags=["Personas"])
    async def delete_persona(persona_id: str):
        deleted = persona_registry.delete(persona_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Persona '{persona_id}' not found")
        return {"status": "deleted", "persona_id": persona_id}

    # ==========================================================
    # Worker Pool Management
    # ==========================================================

    @app.get("/v1/workers", tags=["Workers"])
    async def list_workers():
        return {
            "workers": worker_pool.list_workers(),
            "stats": worker_pool.get_stats(),
        }

    @app.post("/v1/workers", tags=["Workers"], status_code=status.HTTP_201_CREATED)
    async def add_worker(node: WorkerNodeConfig):
        worker_pool.register_worker(node)
        return {"status": "registered", "worker": node.model_dump()}

    @app.delete("/v1/workers/{worker_id}", tags=["Workers"])
    async def remove_worker(worker_id: str):
        worker_pool.unregister_worker(worker_id)
        return {"status": "unregistered", "worker_id": worker_id}

    # ==========================================================
    # Session Observability
    # ==========================================================

    @app.get("/v1/sessions", tags=["Sessions"])
    async def list_active_sessions():
        return {
            "active_sessions": mgr.list_active_sessions(),
            "count": len(mgr.list_active_sessions()),
        }

    @app.get("/v1/sessions/history", tags=["Sessions"])
    async def list_session_history(limit: int = 50):
        return {
            "history": mgr.list_recent_history(limit=limit),
        }

    @app.post("/v1/sessions/{session_id}/end", tags=["Sessions"])
    async def terminate_session(session_id: str):
        metrics = await mgr.end_session(session_id)
        if not metrics:
            raise HTTPException(status_code=404, detail=f"Active session '{session_id}' not found")
        return {"status": "terminated", "metrics": metrics}

    # ==========================================================
    # Voice Catalog, Presets & Voice Cloning API
    # ==========================================================

    @app.get("/v1/voices", tags=["Voices"])
    async def list_voices():
        # Built-in presets with preview URLs
        presets = []
        for vm in VOICE_METADATA:
            entry = dict(vm)
            entry["preview_url"] = f"/v1/voices/{vm['id']}/preview"
            entry["is_cloned"] = False
            presets.append(entry)

        # Append registered cloned voices
        cloned = default_voice_cloner.list_cloned_voices()
        for cv in cloned:
            presets.append({
                "id": cv["id"],
                "name": cv.get("name", cv["id"]),
                "gender": cv.get("gender", "Neutral"),
                "category": "Cloned",
                "tag": cv.get("tag", "Custom Cloned Voice"),
                "description": cv.get("description", "User-cloned voice profile"),
                "preview_url": f"/v1/voices/{cv['id']}/preview",
                "is_cloned": True,
                "duration_sec": cv.get("duration_sec", 0.0),
            })

        return {
            "voices": presets,
            "total": len(presets),
            "categories": ["All", "Female", "Male", "Natural", "Variety", "Cloned"],
        }

    @app.get("/v1/voices/cloned", tags=["Voices"])
    async def list_cloned_voices():
        return {
            "cloned_voices": default_voice_cloner.list_cloned_voices(),
        }

    @app.get("/v1/voices/{voice_id}", tags=["Voices"])
    async def get_voice_details(voice_id: str):
        # 1. Check presets
        for vm in VOICE_METADATA:
            if vm["id"].lower() == voice_id.lower() or vm["name"].lower() == voice_id.lower():
                entry: dict[str, Any] = dict(vm)
                entry["preview_url"] = f"/v1/voices/{vm['id']}/preview"
                entry["is_cloned"] = False
                return entry

        # 2. Check cloned
        if default_voice_cloner.has_voice(voice_id):
            meta = default_voice_cloner.get_voice_metadata(voice_id)
            if meta:
                return meta

        raise HTTPException(
            status_code=404,
            detail=f"Voice '{voice_id}' not found. Must be one of the 18 official presets or a registered cloned voice."
        )

    @app.get("/v1/voices/{voice_id}/preview", tags=["Voices"])
    async def get_voice_preview(voice_id: str):
        # 1. Cloned voice: serve stored artifact wav
        if default_voice_cloner.has_voice(voice_id):
            wav_path = default_voice_cloner.get_voice_path(voice_id)
            if wav_path and wav_path.exists() and wav_path.suffix == ".wav":
                def _read_wav_file():
                    with open(wav_path, "rb") as f:
                        return f.read()
                content = await asyncio.to_thread(_read_wav_file)
                return Response(content=content, media_type="audio/wav")

        # 2. Official preset: check if file or sample exists, else synthesize preview tone
        clean_id = voice_id.upper()
        if not clean_id.endswith(".PT") and not clean_id.endswith(".WAV"):
            clean_id = f"{clean_id}.pt"

        valid_preset = any(vm["id"].upper() == clean_id for vm in VOICE_METADATA)
        if not valid_preset:
            raise HTTPException(status_code=404, detail=f"Voice preset '{voice_id}' not found.")

        # Synthesize harmonic preview chime at 24kHz off the event loop
        def _synth_preview_tone():
            import numpy as np
            import soundfile as sf
            sr = 24000
            dur = 1.8
            t = np.linspace(0, dur, int(sr * dur), endpoint=False)
            f0 = 220.0 if "F" in clean_id else 140.0
            sig = 0.4 * np.sin(2 * np.pi * f0 * t) * np.exp(-1.5 * t) + 0.2 * np.sin(2 * np.pi * f0 * 1.5 * t) * np.exp(-2.0 * t)
            bio = io.BytesIO()
            sf.write(bio, sig.astype(np.float32), sr, format="WAV")
            return bio.getvalue()

        wav_bytes = await asyncio.to_thread(_synth_preview_tone)
        return Response(content=wav_bytes, media_type="audio/wav")

    @app.post("/v1/voices/clone", tags=["Voices"], status_code=status.HTTP_201_CREATED)
    async def clone_voice_endpoint(
        audio: UploadFile = File(...),
        voice_name: str = Form(default="My Cloned Voice"),
        owner: str = Form(default="default_user"),
        consent: bool = Form(default=False),
        gender: str | None = Form(default=None),
    ):
        if not consent:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Voice cloning requires explicit user consent. Please check the consent box."
            )

        content = await audio.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

        try:
            meta = await asyncio.to_thread(
                default_voice_cloner.clone_voice,
                audio_bytes=content,
                voice_name=voice_name,
                owner=owner,
                consent=consent,
                preferred_gender=gender,
            )
            return {"status": "cloned", "voice": meta}
        except Exception as e:
            logger.warning(f"Voice cloning validation error: {e}")
            raise HTTPException(status_code=400, detail=f"Voice cloning rejected: {e!s}")

    @app.delete("/v1/voices/{voice_id}", tags=["Voices"])
    @app.delete("/v1/voices/cloned/{voice_id}", tags=["Voices"])
    async def delete_cloned_voice_endpoint(voice_id: str):
        # Prevent deletion of official presets
        if any(vm["id"].lower() == voice_id.lower() for vm in VOICE_METADATA):
            raise HTTPException(status_code=400, detail=f"Cannot delete official preset voice '{voice_id}'.")

        success = default_voice_cloner.delete_voice(voice_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Cloned voice '{voice_id}' not found.")
        return {"status": "deleted", "voice_id": voice_id}

    @app.get("/v1/system-prompt", tags=["Persona"])
    async def get_system_prompt(
        accent: str = Query(default="indian"),
        character: str = Query(default="professional"),
    ):
        return {
            "principles": 12,
            "prompt": build_system_prompt(accent, character),
            "accent": accent,
            "character": character,
        }

    # ==========================================================
    # RAG Knowledge Engine Endpoints
    # ==========================================================

    @app.get("/v1/rag/documents", tags=["RAG"])
    async def list_rag_documents():
        return {
            "documents": default_rag_engine.list_documents(),
            "count": len(default_rag_engine.list_documents()),
        }

    @app.post("/v1/rag/upload", tags=["RAG"])
    async def upload_rag_document(file: UploadFile = File(...)):
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        doc = await asyncio.to_thread(
            default_rag_engine.add_document,
            filename=file.filename or "uploaded_doc",
            content=content,
        )
        return {
            "status": "indexed",
            "document": {
                "doc_id": doc.doc_id,
                "filename": doc.filename,
                "file_type": doc.file_type,
                "num_pages": doc.num_pages,
                "chunks_count": len(doc.chunks),
            },
        }

    @app.post("/v1/rag/text", tags=["RAG"])
    async def add_rag_text(data: RAGTextUpload):
        if not data.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty.")
        doc = default_rag_engine.add_text(title=data.title, text=data.text)
        return {
            "status": "indexed",
            "document": {
                "doc_id": doc.doc_id,
                "filename": doc.filename,
                "chunks_count": len(doc.chunks),
            },
        }

    @app.delete("/v1/rag/documents/{doc_id}", tags=["RAG"])
    async def delete_rag_document(doc_id: str):
        success = default_rag_engine.delete_document(doc_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found.")
        return {"status": "deleted", "doc_id": doc_id}

    @app.post("/v1/rag/clear", tags=["RAG"])
    async def clear_rag_documents():
        default_rag_engine.clear()
        return {"status": "cleared"}

    @app.post("/v1/rag/query", tags=["RAG"])
    async def query_rag(req: RAGQueryRequest):
        results = default_rag_engine.query(req.query, top_k=req.top_k)
        grounded = default_rag_engine.generate_grounded_response(req.query)
        return {
            "query": req.query,
            "matches": [
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_name": chunk.doc_name,
                    "text": chunk.text,
                    "score": round(score, 4),
                }
                for chunk, score in results
            ],
            "grounded_response": grounded,
        }

    # ==========================================================
    # Real-Time WebSocket Streaming Endpoint
    # ==========================================================

    @app.websocket("/v1/realtime")
    async def realtime_endpoint(
        websocket: WebSocket,
        persona_id: str = Query(default="indian_pro", description="Registered persona ID"),
        voice_prompt: str | None = Query(default=None, description="Optional override for voice prompt"),
        neural_voice: str | None = Query(default=None, description="Optional override for neural TTS voice"),
        text_prompt: str | None = Query(default=None, description="Optional override for text prompt"),
        session_id: str | None = Query(default=None, description="Optional custom session ID"),
        accent: str | None = Query(default=None, description="Optional accent override (Indian English)"),
        character: str | None = Query(default=None, description="Optional character override (Professional, Friendly & Funny)"),
        call_flow: str | None = Query(default=None, description="Optional call flow role"),
        cleaner_mode: str = Query(default="isolation", description="Audio cleaning mode: isolation, rnnoise, bypass"),
    ):
        # 0. Check Rate Limit
        client_ip = websocket.client.host if websocket.client else "127.0.0.1"
        if not default_rate_limiter.is_allowed(client_ip):
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Rate limit exceeded")
            return

        # 0b. Check API Key Authentication if configured
        if settings.gateway.api_key:
            auth_header = websocket.headers.get("authorization", "")
            bearer_token = auth_header[7:].strip() if auth_header.startswith("Bearer ") else None
            api_key = (
                websocket.headers.get("x-api-key")
                or bearer_token
                or websocket.query_params.get("api_key")
            )
            if not api_key or api_key != settings.gateway.api_key:
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Unauthorized: invalid or missing API key")
                return

        await websocket.accept()

        # 1. Resolve Persona
        base_persona = persona_registry.get(persona_id)
        if not base_persona:
            # Fall back to indian_pro if requested ID not found
            base_persona = persona_registry.get("indian_pro") or list(persona_registry.list_all())[0]

        # Apply overrides if provided
        active_persona = base_persona.model_copy()
        if voice_prompt:
            active_persona.voice_prompt = voice_prompt
            active_persona.voice_ref = voice_prompt
        if neural_voice:
            active_persona.neural_voice = neural_voice
        if text_prompt:
            active_persona.text_prompt = text_prompt
            active_persona.system_prompt = text_prompt
        if accent:
            active_persona.accent = accent
        if character:
            active_persona.character = character
        if call_flow:
            active_persona.call_flow = call_flow

        # 2. Validate Voice Selection (must be one of 18 presets or a cloned voice)
        req_voice = active_persona.get_normalized_voice_prompt()
        if req_voice not in OFFICIAL_VOICE_PRESETS and not default_voice_cloner.has_voice(req_voice):
            logger.warning(f"Rejected invalid voice selection: '{req_voice}'")
            err = encode_message(ErrorMessage(error=f"400 Bad Request: Voice '{req_voice}' not found."))
            await websocket.send_bytes(err)
            await websocket.close(code=1008, reason="Invalid voice selection")
            return

        # 3. Acquire Worker and Create Session
        try:
            session = await mgr.create_session(
                persona=active_persona,
                session_id=session_id,
                timeout=30.0,
                barge_in_warmup_sec=1.5,
            )
            if hasattr(session, "cleaner"):
                session.cleaner.set_mode(cleaner_mode)
        except PoolCapacityExceededError as e:
            logger.warning(f"Capacity exceeded for session request: {e}")
            err = encode_message(ErrorMessage(error=f"503 Service Unavailable: All workers busy. {e!s}"))
            await websocket.send_bytes(err)
            await websocket.close(code=1013, reason="Pool saturated")
            return

        # 3. Connect to Worker & Run Prompting Phase
        try:
            await session.start()
            # Send handshake ack to client signaling interactive audio streaming is live
            await websocket.send_bytes(encode_message(HandshakeMessage(version=0, model=0)))
            # Send session metadata
            await websocket.send_bytes(
                encode_message(MetadataMessage(data={
                    "event": "session_started",
                    "session_id": session.session_id,
                    "claimed_from_standby": getattr(session, "is_claimed_from_standby", False),
                    "handshake_time_ms": round(session.metrics.handshake_time_sec * 1000, 1),
                    "persona": active_persona.id,
                    "voice": active_persona.get_normalized_voice_prompt(),
                    "accent": active_persona.accent,
                    "character": active_persona.character,
                    "sample_rate": 24000,
                    "frame_size": 1920,
                    "frame_rate": 12.5,
                }))
            )
        except Exception as e:
            logger.error(f"Failed to start session: {e}")
            err = encode_message(ErrorMessage(error=f"Worker initialization failed: {e!s}"))
            await websocket.send_bytes(err)
            await mgr.end_session(session.session_id)
            await websocket.close(code=1011, reason="Worker initialization failed")
            return

        # 4. Bi-directional Streaming
        async def send_to_client(data: bytes):
            try:
                await websocket.send_bytes(data)
            except Exception:
                pass

        async def client_receive_loop():
            try:
                while True:
                    msg_bytes = await websocket.receive_bytes()
                    res = await session.ingest_client_message(msg_bytes)
                    if isinstance(res, ErrorMessage):
                        try:
                            await websocket.send_bytes(encode_message(res))
                        except Exception:
                            break
            except WebSocketDisconnect:
                logger.info(f"Client disconnected from session {session.session_id}")
            except Exception as e:
                logger.warning(f"WebSocket error in session {session.session_id}: {e}")

        forwarder_task = asyncio.create_task(session.run_worker_forwarder(send_to_client))
        receiver_task = asyncio.create_task(client_receive_loop())

        try:
            done, pending = await asyncio.wait(
                [forwarder_task, receiver_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
        finally:
            await mgr.end_session(session.session_id)

    # ==========================================================
    # Interactive Lean Voice Agent Studio & Modern React SPA
    # ==========================================================
    from pathlib import Path

    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    frontend_dist = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
    if (frontend_dist / "index.html").exists():
        assets_dir = frontend_dist / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        @app.get("/audio-capture-worklet.js", include_in_schema=False)
        async def audio_worklet_js():
            worklet = frontend_dist / "audio-capture-worklet.js"
            if worklet.exists():
                return FileResponse(str(worklet), media_type="application/javascript")
            return HTMLResponse(content="", status_code=404)

        @app.get("/favicon.svg", include_in_schema=False)
        async def favicon_svg():
            fav = frontend_dist / "favicon.svg"
            if fav.exists():
                return FileResponse(str(fav), media_type="image/svg+xml")
            return HTMLResponse(content="", status_code=404)

        @app.get("/icons.svg", include_in_schema=False)
        async def icons_svg():
            ico = frontend_dist / "icons.svg"
            if ico.exists():
                return FileResponse(str(ico), media_type="image/svg+xml")
            return HTMLResponse(content="", status_code=404)

        @app.get("/", response_class=FileResponse, tags=["UI"])
        @app.get("/app", response_class=FileResponse, tags=["UI"])
        @app.get("/agents", response_class=FileResponse, tags=["UI"])
        @app.get("/calls", response_class=FileResponse, tags=["UI"])
        @app.get("/design-system", response_class=FileResponse, tags=["UI"])
        @app.get("/providers", response_class=FileResponse, tags=["UI"])
        async def react_app():
            return FileResponse(str(frontend_dist / "index.html"))
    else:
        @app.get("/", response_class=HTMLResponse, tags=["UI"])
        async def root_redirect():
            return HTMLResponse(content=LEAN_STUDIO_HTML)

    @app.get("/studio", response_class=HTMLResponse, tags=["UI"])
    async def lean_studio_view():
        return HTMLResponse(content=LEAN_STUDIO_HTML)

    @app.get("/console", response_class=HTMLResponse, tags=["UI"])
    @app.get("/console/legacy", response_class=HTMLResponse, tags=["UI"])
    async def legacy_developer_console():
        return HTMLResponse(content=STUDIO_HTML)

    return app
