import asyncio

import numpy as np
import pytest

from orchestration.persona.registry import default_registry
from orchestration.protocol.audio import FRAME_SIZE
from orchestration.protocol.messages import (
    AudioMessage,
    MessageType,
    encode_message,
)
from orchestration.session.manager import SessionManager, SessionState
from orchestration.worker.mock_worker import PersonaPlexMockServer
from orchestration.worker.pool import WorkerNodeConfig, WorkerPool


@pytest.mark.asyncio
async def test_session_lifecycle_and_barge_in():
    port = 9888
    mock = PersonaPlexMockServer(host="127.0.0.1", port=port, prompt_init_delay=0.01, frame_interval_sec=0.03)
    await mock.start()

    try:
        pool = WorkerPool()
        pool.register_worker(WorkerNodeConfig(id="worker-sess-test", host="127.0.0.1", port=port))

        manager = SessionManager(pool=pool)
        persona = default_registry.get("indian_pro")
        assert persona is not None

        session = await manager.create_session(persona=persona, session_id="test-session-lifecycle")
        assert session.state == SessionState.INITIALIZING

        # Start session
        await session.start()
        assert session.state == SessionState.ACTIVE

        # Forwarder task to simulate client receiving
        received_client_messages = []
        async def send_to_client(payload: bytes):
            received_client_messages.append(payload)

        forwarder_task = asyncio.create_task(session.run_worker_forwarder(send_to_client))

        # Push silent audio frame
        silent_frame = np.zeros(FRAME_SIZE, dtype=np.float32)
        raw_audio_msg = encode_message(AudioMessage(data=silent_frame.tobytes()))
        await session.ingest_client_message(raw_audio_msg)
        assert session.metrics.user_frames_in == 1
        assert session.state == SessionState.ACTIVE

        # Wait a moment for initial agent frames while ACTIVE
        await asyncio.sleep(0.1)
        assert len(received_client_messages) > 0

        # Push high-energy frame (user interruption / barge-in)
        loud_frame = (np.ones(FRAME_SIZE, dtype=np.float32) * 0.5)
        raw_loud_msg = encode_message(AudioMessage(data=loud_frame.tobytes()))
        await session.ingest_client_message(raw_loud_msg)
        assert session.metrics.user_frames_in == 2
        assert session.metrics.barge_in_events == 1
        assert session.state == SessionState.INTERRUPTED

        # End session
        metrics = await manager.end_session("test-session-lifecycle")
        assert metrics is not None
        assert metrics["session_id"] == "test-session-lifecycle"
        assert metrics["user_frames_in"] == 2
        assert metrics["barge_in_events"] == 1
        assert metrics["duration_sec"] >= 0.0

        forwarder_task.cancel()
        try:
            await forwarder_task
        except asyncio.CancelledError:
            pass

        # Verify worker was released back to pool
        worker = pool.get_worker("worker-sess-test")
        assert worker.is_available

    finally:
        await mock.stop()


@pytest.mark.asyncio
async def test_audio_before_handshake_rejected():
    """AUDIT-009: Ingesting audio before handshake/active state must return ErrorMessage, not None."""
    pool = WorkerPool()
    port = 9889
    mock = PersonaPlexMockServer(host="127.0.0.1", port=port)
    await mock.start()
    try:
        pool.register_worker(WorkerNodeConfig(id="worker-sess-early", host="127.0.0.1", port=port))
        manager = SessionManager(pool=pool)
        persona = default_registry.get("indian_pro")
        session = await manager.create_session(persona=persona, session_id="test-early-audio")
        assert session.state == SessionState.INITIALIZING

        # Send audio while INITIALIZING
        raw_audio_msg = encode_message(AudioMessage(data=np.zeros(FRAME_SIZE, dtype=np.float32).tobytes()))
        res = await session.ingest_client_message(raw_audio_msg)
        assert res is not None
        assert res.type == MessageType.ERROR
        assert "not accepted before session is active" in res.error

        await manager.end_session("test-early-audio")
    finally:
        await mock.stop()


@pytest.mark.asyncio
async def test_ingest_malformed_message_returns_error_without_crash():
    """AUDIT-003: Malformed frames must return ErrorMessage and not crash or raise uncaught exceptions."""
    pool = WorkerPool()
    port = 9890
    mock = PersonaPlexMockServer(host="127.0.0.1", port=port)
    await mock.start()
    try:
        pool.register_worker(WorkerNodeConfig(id="worker-sess-malformed", host="127.0.0.1", port=port))
        manager = SessionManager(pool=pool)
        persona = default_registry.get("indian_pro")
        session = await manager.create_session(persona=persona, session_id="test-malformed")
        await session.start()
        assert session.state == SessionState.ACTIVE

        # Test empty frame
        res_empty = await session.ingest_client_message(b"")
        assert res_empty is not None
        assert res_empty.type == MessageType.ERROR
        assert "Malformed frame" in res_empty.error

        # Test invalid kind (0xFF)
        res_invalid = await session.ingest_client_message(b"\xFF\x01\x02")
        assert res_invalid is not None
        assert res_invalid.type == MessageType.ERROR
        assert "Malformed frame" in res_invalid.error

        # Session must remain ACTIVE
        assert session.state == SessionState.ACTIVE

        await manager.end_session("test-malformed")
    finally:
        await mock.stop()


@pytest.mark.asyncio
async def test_worker_release_on_start_failure():
    """AUDIT-002: If session.start() fails, worker lease must be released back to the pool."""
    pool = WorkerPool()
    # Port 19999 has no server running
    pool.register_worker(WorkerNodeConfig(id="worker-fail-start", host="127.0.0.1", port=19999))
    manager = SessionManager(pool=pool)
    persona = default_registry.get("indian_pro")

    session = await manager.create_session(persona=persona, session_id="test-start-fail")
    # Worker is leased
    worker = pool.get_worker("worker-fail-start")
    assert not worker.is_available

    with pytest.raises(Exception):
        await session.start()

    # Worker MUST be released so another session can acquire it
    assert worker.is_available, "Worker was leaked after session.start() failed!"


@pytest.mark.asyncio
async def test_barge_in_suppresses_agent_frames():
    """AUDIT-007: When barge-in occurs, agent audio frames must be suppressed until user finishes speaking."""
    port = 9891
    mock = PersonaPlexMockServer(host="127.0.0.1", port=port, frame_interval_sec=0.02)
    await mock.start()
    try:
        pool = WorkerPool()
        pool.register_worker(WorkerNodeConfig(id="worker-sess-bargein", host="127.0.0.1", port=port))
        manager = SessionManager(pool=pool)
        persona = default_registry.get("indian_pro")
        session = await manager.create_session(persona=persona, session_id="test-bargein-suppression")
        await session.start()

        received_audio: list[bytes] = []
        async def send_to_client(payload: bytes):
            if payload and payload[0] == 0x01:
                received_audio.append(payload)

        forwarder_task = asyncio.create_task(session.run_worker_forwarder(send_to_client))
        await asyncio.sleep(0.1)
        initial_count = len(received_audio)
        assert initial_count > 0, "Expected initial agent audio frames"

        # User interrupts with loud speech
        loud_frame = np.ones(FRAME_SIZE, dtype=np.float32) * 0.5
        await session.ingest_client_message(encode_message(AudioMessage(data=loud_frame.tobytes())))
        assert session.state == SessionState.INTERRUPTED

        # During interruption, no new agent audio frames should reach client
        count_at_interrupt = len(received_audio)
        await asyncio.sleep(0.1)
        assert len(received_audio) == count_at_interrupt, "Agent audio frames leaked during user interruption!"

        forwarder_task.cancel()
        try:
            await forwarder_task
        except asyncio.CancelledError:
            pass
        await manager.end_session("test-bargein-suppression")
    finally:
        await mock.stop()



