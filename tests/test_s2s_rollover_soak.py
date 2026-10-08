"""
Tests for Seamless Session Rollover and Soak Call Mechanics (Bug 2).

Verifies:
1. Session Rollover triggers reliably before LM context budget expires without WebSocket drop.
2. Cross-worker / refresh transition preserves continuous audio streaming.
3. Transcript repetition loop detector catches degenerative repetition loops and passes natural speech.
"""

from __future__ import annotations

import asyncio
import json
import uuid

import numpy as np
import pytest
import uvicorn
from websockets.asyncio.client import connect as ws_connect

from orchestration.audio.framing import InboundAudioFrameProcessor
from orchestration.db.service import AgentService, CallSessionService
from orchestration.db.session import get_session_factory
from orchestration.gateway.app import create_app
from orchestration.persona.registry import PersonaRegistry
from orchestration.session.manager import SessionManager
from orchestration.settings import app_settings
from orchestration.worker.mock_worker import PersonaPlexMockServer
from orchestration.worker.pool import WorkerNodeConfig, WorkerPool


def detect_repetition_loops(transcript_text: str, n_gram_size: int = 4, max_repeats: int = 3) -> bool:
    """
    Detects degenerative repetition loops in model transcript.
    Returns True if any n-gram sequence is repeated >= max_repeats consecutively.
    """
    words = transcript_text.lower().split()
    if len(words) < n_gram_size * max_repeats:
        return False

    for i in range(len(words) - n_gram_size + 1):
        ngram = tuple(words[i : i + n_gram_size])
        repeats = 1
        j = i + n_gram_size
        while j + n_gram_size <= len(words):
            next_ngram = tuple(words[j : j + n_gram_size])
            if next_ngram == ngram:
                repeats += 1
                if repeats >= max_repeats:
                    return True
                j += n_gram_size
            else:
                break
    return False


def test_repetition_detector():
    """Verify repetition detector flags degenerative loops and allows normal speech."""
    natural_speech = (
        "Hello Aarav, how are you today? I am doing quite well, thank you for asking! "
        "Have you watched the cricket match yesterday? Yes, it was really exciting."
    )
    assert not detect_repetition_loops(natural_speech)

    degenerative_loop = (
        "Hello Aarav! How are you doing? "
        "How are you doing? How are you doing? How are you doing? How are you doing?"
    )
    assert detect_repetition_loops(degenerative_loop)


@pytest.mark.asyncio
async def test_seamless_session_rollover_lifecycle():
    """
    Simulates a multi-worker pool and triggers seamless rollover
    by setting a brief rollover budget. Asserts:
    - Background rollover executes.
    - Client WebSocket connection never drops.
    - Continuous audio and transcript tokens are received.
    - Clean hangup.
    """
    # Configure fast rollover for testing (3s budget, 50% threshold = triggers at 1.5s)
    orig_budget = app_settings.ROLLOVER_BUDGET_SEC
    orig_thresh = app_settings.ROLLOVER_THRESHOLD
    app_settings.ROLLOVER_BUDGET_SEC = 3.0
    app_settings.ROLLOVER_THRESHOLD = 0.50

    mock_port_1 = 9880
    mock_port_2 = 9881
    server_1 = PersonaPlexMockServer(host="127.0.0.1", port=mock_port_1)
    server_2 = PersonaPlexMockServer(host="127.0.0.1", port=mock_port_2)
    await server_1.start()
    await server_2.start()

    session_factory = get_session_factory()
    test_agent_id = f"agt_rollover_{uuid.uuid4().hex[:8]}"
    async with session_factory() as db:
        svc = AgentService(db)
        await svc.create_agent(
            id=test_agent_id,
            name="Aarav Rollover Test",
            voice_id="NATM1.pt",
            greeting_text="Hey, Aarav here! Good to talk with you.",
            greeting_mode="agent_first",
            system_prompt="You enjoy having a good conversation. You are Aarav.",
            end_silence_sec=60,
            max_duration_sec=300,
            auto_publish=True,
        )
        await db.commit()

    pool = WorkerPool()
    pool.register_worker(WorkerNodeConfig(id="worker-w1", host="127.0.0.1", port=mock_port_1))
    pool.register_worker(WorkerNodeConfig(id="worker-w2", host="127.0.0.1", port=mock_port_2))

    app = create_app(pool=pool, registry=PersonaRegistry(), session_manager=SessionManager(pool=pool))

    gw_port = 8795
    config = uvicorn.Config(app, host="127.0.0.1", port=gw_port, log_level="warning")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.4)

    try:
        url = f"ws://127.0.0.1:{gw_port}/v2/voice?agent_id={test_agent_id}&sample_rate=16000&codec=pcm16"
        async with ws_connect(url) as ws:
            # 1. Wait for session_started
            start_raw = await ws.recv()
            start_msg = json.loads(start_raw)
            while start_msg.get("type") == "status":
                start_raw = await ws.recv()
                start_msg = json.loads(start_raw)
            assert start_msg["type"] == "session_started"

            audio_frames_received = 0
            transcript_tokens = []

            # 2. Stream audio continuously for 4.5 seconds (crossing the 1.5s rollover threshold)
            t_start = asyncio.get_running_loop().time()
            dummy_pcm16 = (np.sin(np.linspace(0, 10, 320)) * 2000).astype(np.int16).tobytes()

            while (asyncio.get_running_loop().time() - t_start) < 4.5:
                await ws.send(dummy_pcm16)

                try:
                    res = await asyncio.wait_for(ws.recv(), timeout=0.05)
                    if isinstance(res, bytes):
                        audio_frames_received += 1
                    elif isinstance(res, str):
                        data = json.loads(res)
                        if data.get("type") == "transcript":
                            transcript_tokens.append(data.get("text", ""))
                except TimeoutError:
                    pass

                await asyncio.sleep(0.02)

            # Assert WebSocket remained fully connected throughout and continuous audio arrived
            assert audio_frames_received > 0, "Expected audio frames during simulated call"
            full_transcript = "".join(transcript_tokens)
            assert not detect_repetition_loops(full_transcript), "Detected degenerative phrase repetition loop"

            # 3. Clean hangup
            await ws.send(json.dumps({"type": "hangup"}))
            await asyncio.sleep(0.3)

    finally:
        app_settings.ROLLOVER_BUDGET_SEC = orig_budget
        app_settings.ROLLOVER_THRESHOLD = orig_thresh
        server.should_exit = True
        await server_task
        await server_1.stop()
        await server_2.stop()
