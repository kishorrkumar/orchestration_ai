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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException, status
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

logger = logging.getLogger("orchestration.gateway")


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
    # Real-Time WebSocket Streaming Endpoint
    # ==========================================================

    @app.websocket("/v1/realtime")
    async def realtime_endpoint(
        websocket: WebSocket,
        persona_id: str = Query(default="wise_teacher", description="Registered persona ID"),
        voice_prompt: Optional[str] = Query(default=None, description="Optional override for voice prompt"),
        text_prompt: Optional[str] = Query(default=None, description="Optional override for text prompt"),
        session_id: Optional[str] = Query(default=None, description="Optional custom session ID"),
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
        return HTMLResponse(content=_CONSOLE_HTML)

    @app.get("/", response_class=HTMLResponse, tags=["UI"])
    async def root_redirect():
        return HTMLResponse(content=_CONSOLE_HTML)

    return app


_CONSOLE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PersonaPlex Voice Orchestration Console</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Outfit:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-dark: #0a0d14;
      --bg-panel: rgba(18, 24, 38, 0.7);
      --bg-card: rgba(26, 34, 52, 0.6);
      --border: rgba(255, 255, 255, 0.08);
      --accent-nvidia: #76b900;
      --accent-cyan: #00e5ff;
      --accent-purple: #9d4edd;
      --accent-rose: #ff4d6d;
      --text-main: #f0f4fc;
      --text-muted: #8e9bb0;
      --font-ui: 'Outfit', sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: radial-gradient(circle at top right, #1a233a 0%, var(--bg-dark) 60%);
      color: var(--text-main);
      font-family: var(--font-ui);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      border-bottom: 1px solid var(--border);
      padding: 1.2rem 2.5rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: rgba(10, 13, 20, 0.85);
      backdrop-filter: blur(12px);
    }
    .logo-area { display: flex; align-items: center; gap: 0.8rem; }
    .badge-nvidia {
      background: rgba(118, 185, 0, 0.15);
      color: var(--accent-nvidia);
      border: 1px solid var(--accent-nvidia);
      padding: 0.2rem 0.6rem;
      border-radius: 6px;
      font-size: 0.8rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .grid-container {
      display: grid;
      grid-template-columns: 320px 1fr 340px;
      gap: 1.5rem;
      padding: 2rem 2.5rem;
      flex: 1;
    }
    .panel {
      background: var(--bg-panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 1.5rem;
      backdrop-filter: blur(16px);
      display: flex;
      flex-direction: column;
      gap: 1.2rem;
    }
    h2 { font-size: 1.1rem; font-weight: 600; display: flex; align-items: center; gap: 0.5rem; color: #fff; }
    .stat-badge {
      display: inline-block;
      padding: 0.25rem 0.6rem;
      border-radius: 999px;
      font-size: 0.75rem;
      font-weight: 600;
      background: rgba(0, 229, 255, 0.15);
      color: var(--accent-cyan);
    }
    select, input, textarea {
      width: 100%;
      background: var(--bg-card);
      border: 1px solid var(--border);
      color: #fff;
      padding: 0.65rem 0.9rem;
      border-radius: 8px;
      font-family: inherit;
      outline: none;
      transition: all 0.2s;
    }
    select:focus, input:focus, textarea:focus { border-color: var(--accent-cyan); }
    button.btn-primary {
      background: linear-gradient(135deg, #76b900 0%, #4c8200 100%);
      color: #fff;
      border: none;
      padding: 0.8rem;
      border-radius: 8px;
      font-weight: 600;
      cursor: pointer;
      box-shadow: 0 4px 15px rgba(118, 185, 0, 0.3);
      transition: all 0.2s;
    }
    button.btn-primary:hover { opacity: 0.9; transform: translateY(-1px); }
    button.btn-danger {
      background: rgba(255, 77, 109, 0.2);
      border: 1px solid var(--accent-rose);
      color: var(--accent-rose);
      padding: 0.8rem;
      border-radius: 8px;
      font-weight: 600;
      cursor: pointer;
    }
    .transcript-box {
      flex: 1;
      background: rgba(0, 0, 0, 0.4);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1.2rem;
      font-family: var(--font-mono);
      font-size: 0.9rem;
      overflow-y: auto;
      line-height: 1.6;
      white-space: pre-wrap;
      color: #e2e8f0;
      min-height: 250px;
    }
    .metric-row {
      display: flex;
      justify-content: space-between;
      padding: 0.6rem 0;
      border-bottom: 1px solid rgba(255,255,255,0.04);
      font-size: 0.9rem;
    }
    .metric-val { font-family: var(--font-mono); font-weight: 600; color: var(--accent-cyan); }
    .status-indicator {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      display: inline-block;
      margin-right: 6px;
    }
    .status-idle { background: #10b981; }
    .status-busy { background: #f59e0b; }
    .status-unhealthy { background: #ef4444; }
  </style>
</head>
<body>
  <header>
    <div class="logo-area">
      <span class="badge-nvidia">NVIDIA PersonaPlex</span>
      <h1 style="font-size: 1.25rem; font-weight: 700;">Realtime Voice Orchestration Layer</h1>
    </div>
    <div style="display: flex; gap: 1rem; align-items: center;">
      <span id="gateway-status" class="stat-badge">Gateway Online</span>
    </div>
  </header>

  <div class="grid-container">
    <!-- Left: Agent Persona Selector & Config -->
    <div class="panel">
      <h2>Agent Persona</h2>
      <div>
        <label style="font-size: 0.8rem; color: var(--text-muted); display: block; margin-bottom: 0.4rem;">Select Persona</label>
        <select id="persona-select" onchange="onPersonaChanged()">
          <option value="wise_teacher">Sophia (Wise Teacher)</option>
          <option value="citysan_service">Ayelen Lucero (CitySan)</option>
          <option value="jerusalem_shakshuka">Owen Foster (Shakshuka)</option>
          <option value="aerorentals_pro">Tomaz Novak (AeroRentals)</option>
          <option value="mars_astronaut">Alex (Mars Mission)</option>
        </select>
      </div>

      <div>
        <label style="font-size: 0.8rem; color: var(--text-muted); display: block; margin-bottom: 0.4rem;">Voice Preset (.pt)</label>
        <input type="text" id="voice-input" value="NATF2.pt">
      </div>

      <div style="flex: 1; display: flex; flex-direction: column;">
        <label style="font-size: 0.8rem; color: var(--text-muted); display: block; margin-bottom: 0.4rem;">Text Prompt</label>
        <textarea id="prompt-input" style="flex: 1; resize: none;" rows="6"></textarea>
      </div>

      <button id="btn-session" class="btn-primary" onclick="toggleSession()">Start Voice Session</button>
    </div>

    <!-- Center: Live Audio Streaming & Streaming Transcript -->
    <div class="panel">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <h2>Full-Duplex Transcript & Audio</h2>
        <span id="session-badge" class="stat-badge" style="background: rgba(255,255,255,0.06); color: var(--text-muted);">DISCONNECTED</span>
      </div>

      <div class="transcript-box" id="transcript-content">Connecting... Click "Start Voice Session" to begin.</div>

      <div style="display: flex; gap: 0.6rem; align-items: center;">
        <input type="text" id="chat-input" placeholder="Type here or speak naturally into your microphone..." onkeydown="if(event.key==='Enter') sendTextMessage()" style="flex: 1;">
        <button class="btn-primary" onclick="sendTextMessage()" style="padding: 0.65rem 1.2rem; white-space: nowrap;">Send</button>
      </div>

      <div style="display: flex; gap: 1rem; align-items: center;">
        <div style="flex: 1; height: 35px; background: rgba(0,0,0,0.5); border-radius: 6px; border: 1px solid var(--border); position: relative; overflow: hidden;">
          <div id="audio-energy-bar" style="height: 100%; width: 0%; background: linear-gradient(90deg, #76b900, #00e5ff); transition: width 0.08s;"></div>
        </div>
        <span id="mic-status-label" style="font-size: 0.8rem; font-family: var(--font-mono); color: var(--text-muted);">24 kHz / 12.5 Hz (80ms)</span>
      </div>
    </div>

    <!-- Right: Telemetry & Worker Pool -->
    <div class="panel">
      <h2>Cluster & Worker Pool</h2>
      <div id="worker-list" style="display: flex; flex-direction: column; gap: 0.6rem;">
        <div style="font-size: 0.85rem; color: var(--text-muted);">Loading workers...</div>
      </div>

      <h2 style="margin-top: 1rem;">Session Observability</h2>
      <div>
        <div class="metric-row"><span>Inbound Frames:</span><span class="metric-val" id="metric-frames-in">0</span></div>
        <div class="metric-row"><span>Agent Audio Frames:</span><span class="metric-val" id="metric-frames-out">0</span></div>
        <div class="metric-row"><span>Tokens Streamed:</span><span class="metric-val" id="metric-tokens">0</span></div>
        <div class="metric-row"><span>Barge-in Interruptions:</span><span class="metric-val" id="metric-barge-in">0</span></div>
        <div class="metric-row"><span>Frame Cadence:</span><span class="metric-val">80.0 ms</span></div>
      </div>
    </div>
  </div>

  <script>
    let ws = null;
    let audioCtx = null;
    let micStream = null;
    let micSource = null;
    let micProcessor = null;
    let silentGain = null;
    let recognition = null;
    let nextPlayTime = 0;
    let isConnected = false;
    let agentTurnActive = false;

    // Mic accumulator
    let micBuffer = [];
    const TARGET_SAMPLE_RATE = 24000;
    const FRAME_SIZE = 1920;

    let framesIn = 0;
    let framesOut = 0;
    let tokensCount = 0;
    let bargeIns = 0;

    async function loadAgents() {
      try {
        const res = await fetch('/v1/agents');
        const data = await res.json();
        const sel = document.getElementById('persona-select');
        sel.innerHTML = '';
        data.agents.forEach(a => {
          const opt = document.createElement('option');
          opt.value = a.id;
          opt.innerText = `${a.name} (${a.voice_prompt})`;
          sel.appendChild(opt);
        });
        window.agentsData = data.agents;
        onPersonaChanged();
      } catch (e) {
        console.error('Failed to load agents', e);
      }
    }

    function onPersonaChanged() {
      const selId = document.getElementById('persona-select').value;
      if (!window.agentsData) return;
      const agent = window.agentsData.find(a => a.id === selId);
      if (agent) {
        document.getElementById('voice-input').value = agent.voice_prompt;
        document.getElementById('prompt-input').value = agent.text_prompt;
      }
    }

    async function updateClusterStats() {
      try {
        const res = await fetch('/v1/workers');
        const data = await res.json();
        const container = document.getElementById('worker-list');
        if (data.workers.length === 0) {
          container.innerHTML = '<div style="font-size:0.85rem;color:var(--text-muted);">No workers registered. Add worker node.</div>';
          return;
        }
        container.innerHTML = data.workers.map(w => {
          let dotClass = 'status-idle';
          if (w.status === 'BUSY') dotClass = 'status-busy';
          if (w.status === 'UNHEALTHY') dotClass = 'status-unhealthy';
          return `
            <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:6px;padding:0.6rem;font-size:0.85rem;">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <div><span class="status-indicator ${dotClass}"></span><strong>${w.id}</strong></div>
                <span class="stat-badge">${w.status}</span>
              </div>
              <div style="font-size:0.75rem;color:var(--text-muted);margin-top:0.3rem;">
                Target: ${w.host}:${w.port} ${w.active_session ? '| Sess: ' + w.active_session : ''}
              </div>
            </div>
          `;
        }).join('');
      } catch (e) {}
    }

    setInterval(updateClusterStats, 2000);
    loadAgents();
    updateClusterStats();

    function appendTranscript(speaker, text) {
      const box = document.getElementById('transcript-content');
      if (speaker === 'Agent') {
        if (!agentTurnActive) {
          box.innerHTML += `\\n\\n<strong style="color: #76b900;">Agent:</strong> `;
          agentTurnActive = true;
        }
        box.innerHTML += text;
      } else if (speaker === 'User') {
        box.innerHTML += `\\n\\n<strong style="color: #00e5ff;">You:</strong> ${text}`;
        agentTurnActive = false;
      }
      box.scrollTop = box.scrollHeight;
    }

    function sendTextMessage() {
      const input = document.getElementById('chat-input');
      const text = input.value.trim();
      if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;

      appendTranscript('User', text);
      input.value = '';

      // Stop current playback (barge-in)
      if (audioCtx) {
        nextPlayTime = audioCtx.currentTime;
      }

      // Send as 0x02 Text message
      const textBytes = new TextEncoder().encode(text);
      const msg = new Uint8Array(1 + textBytes.byteLength);
      msg[0] = 0x02; // Text kind
      msg.set(textBytes, 1);
      ws.send(msg);
    }

    async function toggleSession() {
      if (isConnected) {
        disconnectSession();
      } else {
        await startSession();
      }
    }

    async function startSession() {
      const personaId = document.getElementById('persona-select').value;
      const voice = document.getElementById('voice-input').value;
      const prompt = document.getElementById('prompt-input').value;

      document.getElementById('session-badge').innerText = 'REQUESTING MIC...';

      // 1. Initialize Web Audio Context
      try {
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        audioCtx = new AudioContextClass({ sampleRate: TARGET_SAMPLE_RATE });
        if (audioCtx.state === 'suspended') {
          await audioCtx.resume();
        }
      } catch (e) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }

      nextPlayTime = audioCtx.currentTime;

      // 2. Request Microphone Access
      try {
        micStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          }
        });
      } catch (err) {
        alert('Microphone access denied: ' + err.message);
        disconnectSession();
        return;
      }

      // 3. Connect WebSocket
      const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const url = `${proto}//${window.location.host}/v1/realtime?persona_id=${encodeURIComponent(personaId)}&voice_prompt=${encodeURIComponent(voice)}&text_prompt=${encodeURIComponent(prompt)}`;

      document.getElementById('session-badge').innerText = 'CONNECTING GATEWAY...';
      document.getElementById('transcript-content').innerHTML = '<em>Connecting session...</em>';
      framesIn = 0;
      framesOut = 0;
      tokensCount = 0;
      bargeIns = 0;
      micBuffer = [];
      agentTurnActive = false;

      ws = new WebSocket(url);
      ws.binaryType = 'arraybuffer';

      ws.onopen = () => {
        isConnected = true;
        document.getElementById('session-badge').innerText = 'ACTIVE (FULL-DUPLEX)';
        document.getElementById('session-badge').style.background = 'rgba(118, 185, 0, 0.2)';
        document.getElementById('session-badge').style.color = '#76b900';
        document.getElementById('btn-session').innerText = 'End Session';
        document.getElementById('btn-session').className = 'btn-danger';
        document.getElementById('transcript-content').innerHTML = '<em>Connected. Listening to your voice...</em>';

        // Start streaming microphone audio and speech recognition
        startMicPipeline();
        startSpeechRecognition();
      };

      ws.onmessage = (event) => {
        const data = new Uint8Array(event.data);
        if (data.length === 0) return;
        const kind = data[0];
        const payload = data.slice(1);

        if (kind === 0x00) {
          console.log('[PersonaPlex] Handshake 0x00 received from gateway');
        } else if (kind === 0x01) {
          // Agent Audio Frame (PCM Float32, 1920 samples @ 24kHz)
          framesOut++;
          document.getElementById('metric-frames-out').innerText = framesOut;

          // Convert bytes to Float32Array
          const floatSamples = new Float32Array(payload.buffer, payload.byteOffset, payload.byteLength / 4);

          // Play through speakers using Web Audio API
          playAudioChunk(floatSamples);

          // Update energy visualizer
          document.getElementById('audio-energy-bar').style.width = `${Math.min(100, (framesOut % 10) * 10 + 20)}%`;
        } else if (kind === 0x02) {
          // Streaming Text Token
          tokensCount++;
          document.getElementById('metric-tokens').innerText = tokensCount;
          const text = new TextDecoder().decode(payload);
          appendTranscript('Agent', text);
        } else if (kind === 0x04) {
          // Metadata (Session started, barge_in event, etc.)
          const meta = JSON.parse(new TextDecoder().decode(payload));
          if (meta.event === 'barge_in') {
            bargeIns++;
            document.getElementById('metric-barge-in').innerText = bargeIns;
          }
        } else if (kind === 0x05) {
          const err = new TextDecoder().decode(payload);
          alert('Orchestration Error: ' + err);
        }
      };

      ws.onclose = () => {
        disconnectSession();
      };
      ws.onerror = (e) => {
        console.error('WebSocket error:', e);
      };
    }

    function startSpeechRecognition() {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SpeechRecognition) return;

      try {
        recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = 'en-US';

        recognition.onresult = (event) => {
          let interimText = '';
          for (let i = event.resultIndex; i < event.results.length; ++i) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
              const cleaned = transcript.trim();
              if (cleaned.length > 0) {
                appendTranscript('User', cleaned);
                // Cut off agent playback immediately (barge-in)
                if (audioCtx) {
                  nextPlayTime = audioCtx.currentTime;
                }
                // Send text over WebSocket (0x02)
                if (ws && ws.readyState === WebSocket.OPEN) {
                  const textBytes = new TextEncoder().encode(cleaned);
                  const msg = new Uint8Array(1 + textBytes.byteLength);
                  msg[0] = 0x02; // Text kind
                  msg.set(textBytes, 1);
                  ws.send(msg);
                }
              }
            } else {
              interimText += transcript;
            }
          }
        };

        recognition.onerror = (err) => {
          console.warn('SpeechRecognition error:', err);
        };

        recognition.onend = () => {
          if (isConnected) {
            try { recognition.start(); } catch (e) {}
          }
        };

        recognition.start();
      } catch (e) {
        console.warn('SpeechRecognition initialization error:', e);
      }
    }

    function startMicPipeline() {
      if (!micStream || !audioCtx) return;

      micSource = audioCtx.createMediaStreamSource(micStream);
      micProcessor = audioCtx.createScriptProcessor(2048, 1, 1);

      silentGain = audioCtx.createGain();
      silentGain.gain.value = 0.0;

      const inputSampleRate = audioCtx.sampleRate;

      micProcessor.onaudioprocess = (e) => {
        if (!isConnected || !ws || ws.readyState !== WebSocket.OPEN) return;

        const inputChannel = e.inputBuffer.getChannelData(0);

        let resampled;
        if (inputSampleRate === TARGET_SAMPLE_RATE) {
          resampled = inputChannel;
        } else {
          const ratio = TARGET_SAMPLE_RATE / inputSampleRate;
          const newLength = Math.round(inputChannel.length * ratio);
          resampled = new Float32Array(newLength);
          for (let i = 0; i < newLength; i++) {
            const srcIdx = i / ratio;
            const idx0 = Math.floor(srcIdx);
            const idx1 = Math.min(idx0 + 1, inputChannel.length - 1);
            const frac = srcIdx - idx0;
            resampled[i] = inputChannel[idx0] * (1 - frac) + inputChannel[idx1] * frac;
          }
        }

        for (let i = 0; i < resampled.length; i++) {
          micBuffer.push(resampled[i]);
        }

        while (micBuffer.length >= FRAME_SIZE) {
          const frame = new Float32Array(micBuffer.slice(0, FRAME_SIZE));
          micBuffer = micBuffer.slice(FRAME_SIZE);

          const byteBuffer = new Uint8Array(1 + frame.byteLength);
          byteBuffer[0] = 0x01; // Audio kind
          byteBuffer.set(new Uint8Array(frame.buffer), 1);

          ws.send(byteBuffer);
          framesIn++;
          document.getElementById('metric-frames-in').innerText = framesIn;
        }
      };

      micSource.connect(micProcessor);
      micProcessor.connect(silentGain);
      silentGain.connect(audioCtx.destination);
    }

    function playAudioChunk(floatSamples) {
      if (!audioCtx) return;

      const buffer = audioCtx.createBuffer(1, floatSamples.length, TARGET_SAMPLE_RATE);
      buffer.copyToChannel(floatSamples, 0);

      const source = audioCtx.createBufferSource();
      source.buffer = buffer;
      source.connect(audioCtx.destination);

      const now = audioCtx.currentTime;
      if (nextPlayTime < now) {
        nextPlayTime = now + 0.02;
      }

      source.start(nextPlayTime);
      nextPlayTime += buffer.duration;
    }

    function disconnectSession() {
      if (ws) {
        ws.close();
        ws = null;
      }
      if (recognition) {
        try { recognition.stop(); } catch (e) {}
        recognition = null;
      }
      if (micStream) {
        micStream.getTracks().forEach(t => t.stop());
        micStream = null;
      }
      if (micProcessor) {
        try { micProcessor.disconnect(); } catch (e) {}
        micProcessor = null;
      }
      if (micSource) {
        try { micSource.disconnect(); } catch (e) {}
        micSource = null;
      }
      if (silentGain) {
        try { silentGain.disconnect(); } catch (e) {}
        silentGain = null;
      }

      isConnected = false;
      agentTurnActive = false;
      document.getElementById('session-badge').innerText = 'DISCONNECTED';
      document.getElementById('session-badge').style.background = 'rgba(255,255,255,0.06)';
      document.getElementById('session-badge').style.color = 'var(--text-muted)';
      document.getElementById('btn-session').innerText = 'Start Voice Session';
      document.getElementById('btn-session').className = 'btn-primary';
      document.getElementById('audio-energy-bar').style.width = '0%';
    }
  </script>
</body>
</html>
"""
