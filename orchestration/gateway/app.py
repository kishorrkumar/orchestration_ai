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
from contextlib import asynccontextmanager
import logging
import time
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException, status, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..persona.registry import (
    PersonaConfig,
    PersonaRegistry,
    default_registry,
    OFFICIAL_VOICE_PRESETS,
)
from ..worker.pool import WorkerPool, WorkerNodeConfig, PoolCapacityExceededError
from ..session.manager import SessionManager, SessionState
from ..protocol.messages import (
    MessageType,
    HandshakeMessage,
    ErrorMessage,
    MetadataMessage,
    encode_message,
    decode_message,
)
from ..rag.engine import default_rag_engine
from .studio_ui import STUDIO_HTML

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
    pool: Optional[WorkerPool] = None,
    registry: Optional[PersonaRegistry] = None,
    session_manager: Optional[SessionManager] = None,
) -> FastAPI:
    worker_pool = pool or WorkerPool()
    persona_registry = registry or default_registry
    mgr = session_manager or SessionManager(pool=worker_pool)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("PersonaPlex Orchestration Gateway starting up")
        yield
        logger.info("PersonaPlex Orchestration Gateway shutting down")

    app = FastAPI(
        title="PersonaPlex Realtime Voice Orchestration Gateway",
        description="Open-source full-duplex orchestration layer for NVIDIA PersonaPlex speech-to-speech agents",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
        active = mgr.list_active_sessions()
        pool_stats = worker_pool.get_stats()
        total_user_frames = sum(s.get("user_frames_in", 0) for s in active)
        total_agent_frames = sum(s.get("agent_frames_out", 0) for s in active)
        total_barge_ins = sum(s.get("barge_in_events", 0) for s in active)
        return {
            "timestamp": time.time(),
            "workers_total": pool_stats["total_workers"],
            "workers_idle": pool_stats["idle_workers"],
            "workers_busy": pool_stats["busy_workers"],
            "active_sessions_count": len(active),
            "total_user_frames_in": total_user_frames,
            "total_agent_frames_out": total_agent_frames,
            "total_barge_in_events": total_barge_ins,
        }

    # ==========================================================
    # Persona & Agent Catalog
    # ==========================================================

    @app.get("/v1/agents", tags=["Agents"])
    async def list_agents():
        return {
            "agents": [p.model_dump() for p in persona_registry.list_all()],
            "voice_presets": OFFICIAL_VOICE_PRESETS,
        }

    @app.get("/v1/agents/{agent_id}", tags=["Agents"])
    async def get_agent(agent_id: str):
        agent = persona_registry.get(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
        return agent.model_dump()

    @app.post("/v1/agents", tags=["Agents"], status_code=status.HTTP_201_CREATED)
    async def register_agent(persona: PersonaConfig):
        persona_registry.register(persona)
        return {"status": "created", "agent": persona.model_dump()}

    @app.delete("/v1/agents/{agent_id}", tags=["Agents"])
    async def delete_agent(agent_id: str):
        deleted = persona_registry.delete(agent_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
        return {"status": "deleted", "agent_id": agent_id}

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
    # Voice Catalog & Presets
    # ==========================================================

    @app.get("/v1/voices", tags=["Voices"])
    async def list_voices():
        return {
            "voices": VOICE_METADATA,
            "total": len(VOICE_METADATA),
            "categories": ["All", "Female", "Male", "Natural", "Variety"],
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
        doc = default_rag_engine.add_document(filename=file.filename or "uploaded_doc", content=content)
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
        persona_id: str = Query(default="wise_teacher", description="Registered persona ID"),
        voice_prompt: Optional[str] = Query(default=None, description="Optional override for voice prompt"),
        text_prompt: Optional[str] = Query(default=None, description="Optional override for text prompt"),
        session_id: Optional[str] = Query(default=None, description="Optional custom session ID"),
        accent: Optional[str] = Query(default=None, description="Optional accent override (indian, american, british)"),
        character: Optional[str] = Query(default=None, description="Optional character override (professional, funny, warm)"),
    ):
        await websocket.accept()

        # 1. Resolve Persona
        base_persona = persona_registry.get(persona_id)
        if not base_persona:
            err = encode_message(ErrorMessage(error=f"Unknown persona_id '{persona_id}'"))
            await websocket.send_bytes(err)
            await websocket.close(code=1008, reason="Invalid persona")
            return

        # Apply overrides if provided
        active_persona = base_persona.model_copy()
        if voice_prompt:
            active_persona.voice_prompt = voice_prompt
        if text_prompt:
            active_persona.text_prompt = text_prompt
        if accent:
            active_persona.accent = accent
        if character:
            active_persona.character = character

        # 2. Acquire Worker and Create Session
        try:
            session = await mgr.create_session(persona=active_persona, session_id=session_id, timeout=3.0)
        except PoolCapacityExceededError as e:
            logger.warning(f"Capacity exceeded for session request: {e}")
            err = encode_message(ErrorMessage(error=f"503 Service Unavailable: All workers busy. {str(e)}"))
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
            err = encode_message(ErrorMessage(error=f"Worker initialization failed: {str(e)}"))
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

        forwarder_task = asyncio.create_task(session.run_worker_forwarder(send_to_client))

        try:
            while True:
                msg_bytes = await websocket.receive_bytes()
                await session.ingest_client_message(msg_bytes)
        except WebSocketDisconnect:
            logger.info(f"Client disconnected from session {session.session_id}")
        except Exception as e:
            logger.warning(f"WebSocket error in session {session.session_id}: {e}")
        finally:
            forwarder_task.cancel()
            try:
                await forwarder_task
            except asyncio.CancelledError:
                pass
            await mgr.end_session(session.session_id)

    # ==========================================================
    # Interactive Developer Web Console
    # ==========================================================

    @app.get("/console", response_class=HTMLResponse, tags=["UI"])
    async def developer_console():
        return HTMLResponse(content=STUDIO_HTML)

    @app.get("/", response_class=HTMLResponse, tags=["UI"])
    async def root_redirect():
        return HTMLResponse(content=STUDIO_HTML)

    return app
