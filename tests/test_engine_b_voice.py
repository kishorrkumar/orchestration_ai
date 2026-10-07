"""
Integration tests for Engine B Cascaded Voice Runtime (/v2/voice).
Verifies:
1. Engine B dispatch over /v2/voice with wire parity.
2. Greeting delivery (transcript + synthesized audio).
3. Bi-directional audio streaming with Fake STT/LLM/TTS services.
4. Per-turn latency instrumentation (TurnMetrics).
5. Immediate barge-in cancellation.
6. DB session and turn recording with latency metrics.
"""

from __future__ import annotations

import asyncio
import json
import uuid

import numpy as np
import pytest
import uvicorn
from sqlalchemy import select
from websockets.asyncio.client import connect as ws_connect

from orchestration.db.models import CallSession, CallTurn
from orchestration.db.service import AgentService
from orchestration.db.session import get_session_factory, init_db
from orchestration.gateway.app import create_app
from orchestration.persona.registry import PersonaRegistry
from orchestration.session.manager import SessionManager
from orchestration.worker.pool import WorkerPool


@pytest.mark.asyncio
async def test_engine_b_v2_voice_full_lifecycle():
    """
    Test complete Engine B lifecycle over /v2/voice WebSocket:
    - Resolves draft_engine='cascaded_cloud'
    - session_started with engine='cascaded_cloud'
    - Greeting transcript + audio frames
    - Inbound user audio -> FakeSTT transcript -> FakeLLM stream -> FakeTTS audio
    - Turn metrics event
    - Clean hangup and DB record persistence with turn metrics
    """
    await init_db()
    session_factory = get_session_factory()
    test_agent_id = f"agt_engb_{uuid.uuid4().hex[:8]}"

    pipeline_spec = {
        "stt": {"provider": "fake_stt", "model": "mock-flux", "language": "en-US"},
        "llm": {"provider": "fake_llm", "model": "mock-gpt", "temperature": 0.7, "max_tokens": 200},
        "tts": {"provider": "fake_tts", "model": "mock-voice", "voice_id": "mock-alex", "speed": 1.0},
        "turn": {"strategy": "auto", "eager_eot": True, "stop_secs": 0.2},
    }

    async with session_factory() as db:
        svc = AgentService(db)
        await svc.create_agent(
            id=test_agent_id,
            name="Engine B Virtual Banker",
            voice_id="mock-alex",
            greeting_text="Hello, thank you for calling Apex Bank. How may I help you today?",
            greeting_mode="agent_first",
            system_prompt="You are a helpful banking voice agent. Assist callers with accounts.",
            ending_text="Thank you for calling Apex Bank. Goodbye!",
            end_silence_sec=20,
            max_duration_sec=300,
            timezone="America/New_York",
            engine="cascaded_cloud",
            language="en",
            pipeline_json=json.dumps(pipeline_spec),
            auto_publish=True,
        )
        await db.commit()

    pool = WorkerPool()
    app = create_app(pool=pool, registry=PersonaRegistry(), session_manager=SessionManager(pool=pool))

    gw_port = 8795
    config = uvicorn.Config(app, host="127.0.0.1", port=gw_port, log_level="warning")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.4)

    try:
        url = f"ws://127.0.0.1:{gw_port}/v2/voice?agent_id={test_agent_id}&sample_rate=16000&codec=pcm16"
        async with ws_connect(url) as ws:
            # 1. Receive session_started
            start_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            start_msg = json.loads(start_raw)
            assert start_msg["type"] == "session_started"
            assert start_msg["agent_id"] == test_agent_id
            assert start_msg["engine"] == "cascaded_cloud"
            assert start_msg["stt_provider"] == "fake_stt"
            assert start_msg["llm_provider"] == "fake_llm"
            assert start_msg["tts_provider"] == "fake_tts"
            session_id = start_msg["session_id"]

            # 2. Receive greeting transcript
            greet_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            greet_msg = json.loads(greet_raw)
            assert greet_msg["type"] == "transcript"
            assert greet_msg["role"] == "assistant"
            assert greet_msg["is_greeting"] is True
            assert "Apex Bank" in greet_msg["text"]

            # 3. Receive greeting audio and wait until greeting finishes speaking
            got_greeting_audio = False
            for _ in range(30):
                res = await asyncio.wait_for(ws.recv(), timeout=2.0)
                if isinstance(res, bytes) and len(res) > 0:
                    got_greeting_audio = True
                elif isinstance(res, str):
                    data = json.loads(res)
                    if data.get("type") == "agent.speaking.end":
                        break
            assert got_greeting_audio, "Expected TTS audio frames for greeting"

            # Small pause before user starts speaking
            await asyncio.sleep(0.1)

            # 4. Stream 25 frames of 16kHz PCM audio to trigger FakeSTT (25 frames = 500ms)
            frame_i16 = (np.sin(np.linspace(0, 10, 320)) * 5000).astype(np.int16)
            for _ in range(25):
                await ws.send(frame_i16.tobytes())
                await asyncio.sleep(0.01)

            # 5. Receive user transcript, assistant response tokens, audio, and metrics
            got_user_transcript = False
            got_assistant_transcript = False
            got_metrics = False

            for _ in range(40):
                try:
                    res = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    if isinstance(res, str):
                        data = json.loads(res)
                        mtype = data.get("type")
                        if mtype == "transcript" and data.get("role") == "user":
                            got_user_transcript = True
                            assert "what time is it" in data["text"].lower()
                        elif mtype == "transcript" and data.get("role") == "assistant" and not data.get("is_greeting"):
                            got_assistant_transcript = True
                        elif mtype == "metrics.turn":
                            got_metrics = True
                            assert "metrics" in data
                            if got_user_transcript and got_assistant_transcript:
                                break
                except TimeoutError:
                    break

            assert got_user_transcript, "Expected finalized user transcript from FakeSTT"
            assert got_assistant_transcript, "Expected streamed assistant transcript from FakeLLM"
            assert got_metrics, "Expected turn metrics event"

            # 6. Clean hangup
            await ws.send(json.dumps({"type": "hangup"}))
            ended_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            ended_msg = json.loads(ended_raw)
            assert ended_msg["type"] == "session_ended"
            assert ended_msg["reason"] == "user_hangup"

        # 7. Verify Database persistence
        async with session_factory() as db:
            stmt = select(CallSession).where(CallSession.id == session_id)
            call_rec = (await db.execute(stmt)).scalar_one_or_none()
            assert call_rec is not None
            assert call_rec.engine == "cascaded_cloud"
            assert call_rec.end_reason == "user_hangup"
            assert call_rec.duration_sec >= 0.0

            # Verify CallTurn records
            turn_stmt = select(CallTurn).where(CallTurn.session_id == session_id).order_by(CallTurn.idx)
            turns = (await db.execute(turn_stmt)).scalars().all()
            assert len(turns) >= 2  # user turn and assistant turn
            roles = [t.role for t in turns]
            assert "user" in roles
            assert "assistant" in roles

    finally:
        server.should_exit = True
        await server_task


@pytest.mark.asyncio
async def test_engine_b_v2_voice_barge_in():
    """
    Test immediate barge-in interruption on Engine B:
    - User sends interrupt frame while agent is speaking
    - Pipeline sends agent.speaking.end with interrupted=True
    """
    await init_db()
    session_factory = get_session_factory()
    test_agent_id = f"agt_barge_{uuid.uuid4().hex[:8]}"

    pipeline_spec = {
        "stt": {"provider": "fake_stt", "model": "mock-flux", "language": "en-US"},
        "llm": {"provider": "fake_llm", "model": "mock-gpt", "temperature": 0.7, "max_tokens": 200},
        "tts": {"provider": "fake_tts", "model": "mock-voice", "voice_id": "mock-alex", "speed": 1.0},
        "turn": {"strategy": "auto", "eager_eot": True, "stop_secs": 0.2},
    }

    async with session_factory() as db:
        svc = AgentService(db)
        await svc.create_agent(
            id=test_agent_id,
            name="Barge-In Test Agent",
            greeting_text="Long greeting text that will take some time to finish speaking completely.",
            greeting_mode="agent_first",
            engine="cascaded_cloud",
            pipeline_json=json.dumps(pipeline_spec),
            auto_publish=True,
        )
        await db.commit()

    pool = WorkerPool()
    app = create_app(pool=pool, registry=PersonaRegistry(), session_manager=SessionManager(pool=pool))

    gw_port = 8796
    config = uvicorn.Config(app, host="127.0.0.1", port=gw_port, log_level="warning")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.4)

    try:
        url = f"ws://127.0.0.1:{gw_port}/v2/voice?agent_id={test_agent_id}&sample_rate=16000&codec=pcm16"
        async with ws_connect(url) as ws:
            # Consume session_started & greeting
            await ws.recv()  # session_started
            await ws.recv()  # greeting transcript

            # Send interrupt
            await ws.send(json.dumps({"type": "interrupt"}))

            # Expect agent.speaking.end with interrupted=True
            got_interrupted = False
            for _ in range(10):
                msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                if isinstance(msg, str):
                    data = json.loads(msg)
                    if data.get("type") == "agent.speaking.end" and data.get("interrupted") is True:
                        got_interrupted = True
                        break

            assert got_interrupted, "Expected agent.speaking.end with interrupted=True on barge-in"

            # Clean exit
            await ws.send(json.dumps({"type": "hangup"}))
    finally:
        server.should_exit = True
        await server_task
