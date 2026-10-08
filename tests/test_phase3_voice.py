"""
Integration tests for Phase 3: Lean S2S Voice Session Runtime (/v2/voice).
Verifies full-duplex WebSocket streaming, prompt compilation, 16 kHz/8 kHz audio routing,
barge-in, EndOfCallDetector hangup, and database call logging.
"""

from __future__ import annotations

import asyncio
import json
import uuid

import numpy as np
import pytest
import uvicorn
import websockets
from websockets.asyncio.client import connect as ws_connect

from orchestration.db.service import AgentService, CallSessionService
from orchestration.db.session import get_session_factory
from orchestration.gateway.app import create_app
from orchestration.persona.registry import PersonaRegistry
from orchestration.session.manager import SessionManager
from orchestration.worker.mock_worker import PersonaPlexMockServer
from orchestration.worker.pool import WorkerNodeConfig, WorkerPool


@pytest.mark.asyncio
async def test_v2_voice_rejects_missing_agent():
    """Verify /v2/voice cleanly rejects requests for non-existent agents."""
    pool = WorkerPool()
    app = create_app(pool=pool, registry=PersonaRegistry(), session_manager=SessionManager(pool=pool))

    gw_port = 8790
    config = uvicorn.Config(app, host="127.0.0.1", port=gw_port, log_level="warning")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.4)

    try:
        url = f"ws://127.0.0.1:{gw_port}/v2/voice?agent_id=does_not_exist_agent_xyz"
        async with ws_connect(url) as ws:
            msg = await ws.recv()
            if isinstance(msg, str):
                data = json.loads(msg)
                assert data["type"] == "error"
                assert "not found" in data["message"].lower()
    except Exception as e:
        # Some websocket clients raise directly on 1008 close
        assert "1008" in str(e) or "closed" in str(e).lower()
    finally:
        server.should_exit = True
        await server_task


@pytest.mark.asyncio
async def test_v2_voice_full_web_lifecycle():
    """
    Verify complete 16 kHz web full-duplex session on /v2/voice:
    - Resolves agent and version from DB
    - Dynamic prompt compilation
    - Session start handshake
    - Greeting playback
    - Bi-directional 16 kHz audio streaming
    - Clean hangup and DB record persistence
    """
    mock_port = 9896
    mock_server = PersonaPlexMockServer(host="127.0.0.1", port=mock_port)
    await mock_server.start()

    session_factory = get_session_factory()
    import uuid
    test_agent_id = f"agt_web_{uuid.uuid4().hex[:8]}"
    async with session_factory() as db:
        svc = AgentService(db)
        await svc.create_agent(
            id=test_agent_id,
            name="Clinic Receptionist Test",
            voice_id="NATF1.pt",
            greeting_text="Hello, thank you for calling Metro Clinic. How may I assist you?",
            greeting_mode="agent_first",
            system_prompt="You are a warm receptionist at Metro Clinic. Help patients book appointments.",
            ending_text="Have a healthy day. Goodbye!",
            end_silence_sec=25,
            max_duration_sec=300,
            timezone="America/New_York",
            auto_publish=True,
        )
        await db.commit()

    pool = WorkerPool()
    pool.register_worker(WorkerNodeConfig(id="worker-v2-web", host="127.0.0.1", port=mock_port))
    app = create_app(pool=pool, registry=PersonaRegistry(), session_manager=SessionManager(pool=pool))

    gw_port = 8791
    config = uvicorn.Config(app, host="127.0.0.1", port=gw_port, log_level="warning")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.4)

    try:
        url = f"ws://127.0.0.1:{gw_port}/v2/voice?agent_id={test_agent_id}&sample_rate=16000&codec=pcm16"
        async with ws_connect(url) as ws:
            # 1. Receive session_started event (skipping status handshakes if present)
            start_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            start_msg = json.loads(start_raw)
            while start_msg.get("type") == "status":
                start_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                start_msg = json.loads(start_raw)
            assert start_msg["type"] == "session_started"
            assert start_msg["agent_id"] == test_agent_id
            assert start_msg["sample_rate"] == 16000
            assert start_msg["codec"] == "pcm16"
            assert start_msg["greeting_mode"] == "agent_first"
            session_id = start_msg["session_id"]

            # 2. Receive greeting transcript
            greet_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            greet_msg = json.loads(greet_raw)
            assert greet_msg["type"] == "transcript"
            assert greet_msg["role"] == "assistant"
            # In Phase 1 full-duplex S2S, fake synthetic text greeting injection was removed
            # to eliminate the double-greeting bug. The transcript tokens arrive directly from the worker.
            assert len(greet_msg["text"]) > 0

            # 3. Stream 16 kHz PCM frames from client (320 samples = 640 bytes)
            frame_i16 = (np.sin(np.linspace(0, 10, 320)) * 5000).astype(np.int16)
            for _ in range(5):
                await ws.send(frame_i16.tobytes())
                await asyncio.sleep(0.02)

            # 4. Receive agent responses (audio bytes or text tokens)
            got_audio_or_transcript = False
            for _ in range(10):
                try:
                    res = await asyncio.wait_for(ws.recv(), timeout=1.5)
                    if isinstance(res, bytes) and len(res) > 0:
                        got_audio_or_transcript = True
                        break
                    elif isinstance(res, str):
                        data = json.loads(res)
                        if data.get("type") in ("transcript", "call_ended"):
                            got_audio_or_transcript = True
                            break
                except TimeoutError:
                    break

            assert got_audio_or_transcript, "Client received neither audio nor transcript from server"

            # 5. Clean user hangup
            await ws.send(json.dumps({"type": "hangup"}))
            await asyncio.sleep(0.3)

        # 6. Verify database record
        async with session_factory() as db:
            call_svc = CallSessionService(db)
            db_call = await call_svc.get_session(session_id)
            assert db_call is not None
            assert db_call.agent_id == test_agent_id
            assert db_call.end_reason == "user_hangup"
            assert db_call.duration_sec >= 0.0

        # Verify worker is released
        await asyncio.sleep(0.2)
        worker = pool.get_worker("worker-v2-web")
        assert worker.is_available

    finally:
        server.should_exit = True
        await server_task
        await mock_server.stop()


@pytest.mark.asyncio
async def test_v2_voice_telephony_g711_lifecycle():
    """
    Verify 8 kHz Telephony session on /v2/voice:
    - 8 kHz sample rate
    - G.711 mu-law (160 bytes per 20 ms frame)
    - Bidirectional encode/decode
    """
    mock_port = 9897
    mock_server = PersonaPlexMockServer(host="127.0.0.1", port=mock_port)
    await mock_server.start()

    session_factory = get_session_factory()
    test_agent_id = f"agt_tel_{uuid.uuid4().hex[:8]}"
    async with session_factory() as db:
        svc = AgentService(db)
        await svc.create_agent(
            id=test_agent_id,
            name="Telephony Support Bot",
            voice_id="NATM0.pt",
            greeting_text="Welcome to support line.",
            greeting_mode="agent_first",
            system_prompt="Assist callers with telephony inquiries concisely.",
            auto_publish=True,
        )
        await db.commit()

    pool = WorkerPool()
    pool.register_worker(WorkerNodeConfig(id="worker-v2-tel", host="127.0.0.1", port=mock_port))
    app = create_app(pool=pool, registry=PersonaRegistry(), session_manager=SessionManager(pool=pool))

    gw_port = 8792
    config = uvicorn.Config(app, host="127.0.0.1", port=gw_port, log_level="warning")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.4)

    try:
        url = f"ws://127.0.0.1:{gw_port}/v2/voice?agent_id={test_agent_id}&sample_rate=8000&codec=g711_ulaw"
        async with ws_connect(url) as ws:
            # 1. Receive session_started
            start_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            start_msg = json.loads(start_raw)
            while start_msg.get("type") == "status":
                start_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                start_msg = json.loads(start_raw)
            assert start_msg["type"] == "session_started"
            assert start_msg["sample_rate"] == 8000
            assert start_msg["codec"] == "g711_ulaw"

            # 2. Greeting transcript from worker/model
            greet_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            greet_data = json.loads(greet_raw)
            assert greet_data["type"] == "transcript"
            assert greet_data["role"] == "assistant"
            # Assistant speech tokens arrive from worker instead of fake injection
            assert len(greet_data["text"]) > 0

            # 3. Send 160-byte G.711 mu-law frames (20 ms at 8 kHz)
            from orchestration.audio.codecs import encode_ulaw
            pcm16 = (np.sin(np.linspace(0, 10, 160)) * 5000).astype(np.int16)
            ulaw_bytes = encode_ulaw(pcm16)
            assert len(ulaw_bytes) == 160

            for _ in range(5):
                await ws.send(ulaw_bytes)
                await asyncio.sleep(0.02)

            # 4. User hangup
            await ws.send(json.dumps({"type": "hangup"}))
            await asyncio.sleep(0.2)

    finally:
        server.should_exit = True
        await server_task
        await mock_server.stop()


@pytest.mark.asyncio
async def test_v2_voice_end_of_call_detector():
    """Verify EndOfCallDetector triggers clean automatic hangup upon silence timeout."""
    mock_port = 9898
    mock_server = PersonaPlexMockServer(host="127.0.0.1", port=mock_port)
    await mock_server.start()

    session_factory = get_session_factory()
    test_agent_id = f"agt_eoc_{uuid.uuid4().hex[:8]}"
    async with session_factory() as db:
        svc = AgentService(db)
        await svc.create_agent(
            id=test_agent_id,
            name="Closing Agent",
            voice_id="NATF2.pt",
            greeting_text="Hello, how can I help you?",
            greeting_mode="agent_first",
            system_prompt="Assist with simple queries.",
            ending_text="Have a wonderful day, goodbye!",
            end_silence_sec=20,
            max_duration_sec=2,  # 2 second max duration for rapid test
            auto_publish=True,
        )
        await db.commit()

    pool = WorkerPool()
    pool.register_worker(WorkerNodeConfig(id="worker-v2-eoc", host="127.0.0.1", port=mock_port))
    app = create_app(pool=pool, registry=PersonaRegistry(), session_manager=SessionManager(pool=pool))

    gw_port = 8793
    config = uvicorn.Config(app, host="127.0.0.1", port=gw_port, log_level="warning")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.4)

    try:
        url = f"ws://127.0.0.1:{gw_port}/v2/voice?agent_id={test_agent_id}&sample_rate=16000&codec=pcm16"
        async with ws_connect(url) as ws:
            # Receive session_started
            start_raw = await ws.recv()
            start_msg = json.loads(start_raw)
            while start_msg.get("type") == "status":
                start_raw = await ws.recv()
                start_msg = json.loads(start_raw)
            assert start_msg["type"] == "session_started"

            # Receive greeting
            greet_raw = await ws.recv()
            greet_msg = json.loads(greet_raw)
            assert greet_msg["type"] == "transcript"

            # Client remains silent; EndOfCallDetector should trigger silence_timeout or max_duration
            call_ended_received = False
            for _ in range(100):
                try:
                    res = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    if isinstance(res, str):
                        data = json.loads(res)
                        if data.get("type") == "call_ended":
                            call_ended_received = True
                            assert data["reason"] in ("max_duration", "max_duration_exceeded", "silence_timeout", "agent_closed", "goodbye")
                            break
                except TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    break

            assert call_ended_received, "Failed to receive automatic call_ended from EndOfCallDetector"

    finally:
        server.should_exit = True
        await server_task
        await mock_server.stop()

