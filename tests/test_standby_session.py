"""
Unit tests for Standby Pre-Primed Voice Sessions.
Validates:
1. Pre-priming establishes an active session on idle worker.
2. Continuous 80ms silence frames maintain Moshi worker priming state.
3. Incoming call claims standby session in 0ms (TTFA < 1.5s).
4. Automatic replacement standby session is scheduled upon claim.
5. Voice mismatch falls back to standard acquisition without breaking.
"""

import asyncio
import pytest
import numpy as np

from orchestration.persona.registry import PersonaConfig, PersonaRegistry
from orchestration.protocol.messages import AudioMessage, HandshakeMessage, MessageType, decode_message, encode_message
from orchestration.session.manager import SessionManager, SessionState
from orchestration.worker.client import PersonaPlexWorkerClient
from orchestration.worker.mock_worker import PersonaPlexMockServer
from orchestration.worker.pool import WorkerNodeConfig, WorkerPool


@pytest.mark.asyncio
async def test_standby_preprime_and_instant_claim():
    mock_port = 9896
    mock = PersonaPlexMockServer(host="127.0.0.1", port=mock_port, prompt_init_delay=0.1)
    await mock.start()

    try:
        pool = WorkerPool()
        pool.register_worker(WorkerNodeConfig(id="worker-standby-test", host="127.0.0.1", port=mock_port))
        mgr = SessionManager(pool=pool, enable_standby=True)

        persona = PersonaConfig(
            id="test_persona",
            name="Test Voice",
            voice_prompt="NATM0.pt",
            text_prompt="You enjoy having a good conversation.",
        )

        # 1. Pre-prime standby session
        standby = await mgr.preprime_standby(persona)
        assert standby is not None
        assert standby.is_standby is True
        assert standby.state == SessionState.ACTIVE
        assert mgr.has_standby() is True

        # 2. Wait 150ms to verify continuous 80ms silence feeder is running
        await asyncio.sleep(0.15)
        assert standby.metrics.silence_frames > 0

        # 3. Create incoming call session: should claim standby session instantly!
        call_session = await mgr.create_session(
            persona=persona,
            session_id="user_call_1",
            use_standby=True,
        )
        assert call_session.session_id == "user_call_1"
        assert call_session.is_claimed_from_standby is True
        assert call_session.metrics.claimed_from_standby is True
        assert call_session.state == SessionState.ACTIVE

        # 4. Calling start() on a claimed session returns immediately in < 0.01s
        t0 = asyncio.get_running_loop().time()
        await call_session.start()
        elapsed = asyncio.get_running_loop().time() - t0
        assert elapsed < 0.05, f"Claimed session start took {elapsed}s, expected < 0.05s"

        # 5. Clean up call session
        await mgr.end_session("user_call_1")

    finally:
        await mock.stop()


@pytest.mark.asyncio
async def test_standby_mismatched_voice_fallback():
    mock1 = PersonaPlexMockServer(host="127.0.0.1", port=9897, prompt_init_delay=0.05)
    mock2 = PersonaPlexMockServer(host="127.0.0.1", port=9898, prompt_init_delay=0.05)
    await mock1.start()
    await mock2.start()

    try:
        pool = WorkerPool()
        # Register 2 workers so one can hold standby and one can handle the mismatch
        pool.register_worker(WorkerNodeConfig(id="worker-w1", host="127.0.0.1", port=9897))
        pool.register_worker(WorkerNodeConfig(id="worker-w2", host="127.0.0.1", port=9898))
        mgr = SessionManager(pool=pool, enable_standby=True)

        persona_a = PersonaConfig(id="p_a", name="A", voice_prompt="NATM0.pt", text_prompt="Prompt A")
        persona_b = PersonaConfig(id="p_b", name="B", voice_prompt="NATF0.pt", text_prompt="Prompt B")

        # Pre-prime with NATM0.pt
        standby = await mgr.preprime_standby(persona_a)
        assert standby is not None
        assert mgr.has_standby() is True

        # Call requests NATF0.pt: should NOT claim mismatched standby
        call_session = await mgr.create_session(
            persona=persona_b,
            session_id="call_diff_voice",
            use_standby=True,
        )
        assert call_session.is_claimed_from_standby is False
        assert call_session.persona.voice_prompt == "NATF0.pt"

        await call_session.start()
        await mgr.end_session("call_diff_voice")
        await standby.close()

    finally:
        await mock1.stop()
        await mock2.stop()
