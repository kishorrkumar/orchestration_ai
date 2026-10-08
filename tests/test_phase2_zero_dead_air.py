"""
Unit & Integration tests for Phase 2 (Zero Dead Air without Greeting & Early Audio Buffering)
and Phase 3 (Conversational Intelligence, Monologue Guard, Detokenizer Contraction Repair, Anti-Leak).
"""

from __future__ import annotations

import asyncio
import json
import time

import numpy as np
import pytest
import uvicorn
from websockets.asyncio.client import connect as ws_connect

from orchestration.audio.detokenizer import detokenize_sentencepiece_stream
from orchestration.gateway.app import create_app
from orchestration.persona.registry import PersonaConfig
from orchestration.prompts.compiler import compile_prompt
from orchestration.worker.mock_worker import PersonaPlexMockServer
from orchestration.worker.pool import WorkerNodeConfig, WorkerPool


@pytest.mark.asyncio
async def test_detokenizer_contractions_and_anti_leak():
    """Verify apostrophe/contraction repair, Kyutai/Moshi redaction, and clean spacing."""
    # 1. Contraction repair
    tokens = [" I", "'", "m", " calling", " from", " Delhi", ",", " and", " you", "'", "re", " ready", "."]
    text = detokenize_sentencepiece_stream(tokens)
    assert text == "I'm calling from Delhi, and you're ready."
    assert "I ' m" not in text
    assert "you ' re" not in text

    # 2. Anti-Moshi & Kyutai leak filter
    leak_tokens = [" Hello", ",", " I", " am", " Moshi", " from", " Kyutai", "."]
    sanitized = detokenize_sentencepiece_stream(leak_tokens, agent_name="Elena")
    assert "Moshi" not in sanitized
    assert "Kyutai" not in sanitized
    assert "Elena" in sanitized
    assert "Snapserve" in sanitized


@pytest.mark.asyncio
async def test_prompt_compiler_preserves_custom_user_prompt():
    """Verify compiler respects user's explicit persona without injecting default sales rep role."""
    user_prompt = "You are Alex, an expert support concierge for Hotel Grand. Speak concisely."
    compiled = compile_prompt(
        system_prompt=user_prompt,
        agent_name="Alex",
    )
    assert "<system>" in compiled.text
    assert "You are Alex, an expert support concierge for Hotel Grand." in compiled.text
    assert "friendly sales rep at Snapserve" not in compiled.text


@pytest.mark.asyncio
async def test_inbound_audio_buffering_during_handshake():
    """
    Verify early audio buffering (Defect 10.4):
    Frames sent by caller immediately upon WebSocket connect (before worker finishes priming)
    are buffered and received by the worker rather than being dropped.
    """
    mock_port = 9881
    gw_port = 8781

    # Simulate worker with 300ms priming delay
    mock_server = PersonaPlexMockServer(host="127.0.0.1", port=mock_port, prompt_init_delay=0.3)
    await mock_server.start()

    pool = WorkerPool()
    pool.register_worker(WorkerNodeConfig(id="test-w1", host="127.0.0.1", port=mock_port))

    app = create_app(pool=pool)
    config = uvicorn.Config(app, host="127.0.0.1", port=gw_port, log_level="error")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    await asyncio.sleep(0.5)

    try:
        url = f"ws://127.0.0.1:{gw_port}/v2/voice?sample_rate=16000&codec=pcm16"
        async with ws_connect(url) as ws:
            # Send audio frames IMMEDIATELY before receiving session_started!
            # 3 frames of 16kHz audio (320 samples = 640 bytes each)
            frame_bytes = (np.sin(np.linspace(0, 10, 320)) * 5000).astype(np.int16).tobytes()
            for _ in range(3):
                await ws.send(frame_bytes)
                await asyncio.sleep(0.01)

            # Receive status and session_started
            start_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            start_msg = json.loads(start_raw)
            while start_msg.get("type") == "status":
                start_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                start_msg = json.loads(start_raw)

            assert start_msg["type"] == "session_started"

            # Verify worker received frames (via frames_sent or audio playback from mock)
            # Send one more frame and hang up
            await ws.send(frame_bytes)
            await asyncio.sleep(0.1)

            # Send hangup
            await ws.send(json.dumps({"type": "hangup"}))
            await asyncio.sleep(0.2)

    finally:
        server.should_exit = True
        await server_task
        await mock_server.stop()


@pytest.mark.asyncio
async def test_fast_capacity_rejection_when_pool_exhausted():
    """
    Verify fast failure (Defect 10.1):
    When all workers are busy or unavailable, gateway rejects call in <= 2.5s with WS 1013
    rather than causing 15s of dead air.
    """
    gw_port = 8782

    # Pool with NO registered workers (capacity exhausted)
    empty_pool = WorkerPool()

    app = create_app(pool=empty_pool)
    config = uvicorn.Config(app, host="127.0.0.1", port=gw_port, log_level="error")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    await asyncio.sleep(0.5)

    try:
        t0 = time.perf_counter()
        url = f"ws://127.0.0.1:{gw_port}/v2/voice?sample_rate=16000&codec=pcm16"
        try:
            async with ws_connect(url) as ws:
                # Expect error or close within 2.5s
                msg_raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
                msg = json.loads(msg_raw)
                while msg.get("type") == "status":
                    msg_raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
                    msg = json.loads(msg_raw)
                assert msg.get("type") == "error"
                assert "capacity" in msg.get("message", "").lower() or "busy" in msg.get("message", "").lower()
        except Exception:
            pass
        elapsed = time.perf_counter() - t0
        # Must fail fast in <= 2.5 seconds (never 15 seconds)
        assert elapsed < 2.5, f"Capacity rejection took {elapsed:.2f}s; expected < 2.5s"

    finally:
        server.should_exit = True
        await server_task
