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
from orchestration.worker.pool import WorkerPool, WorkerNodeConfig
from orchestration.worker.mock_worker import PersonaPlexMockServer
from orchestration.session.manager import SessionManager, SessionState


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

        # Push high-energy frame (user interruption / barge-in)
        loud_frame = (np.ones(FRAME_SIZE, dtype=np.float32) * 0.5)
        raw_loud_msg = encode_message(AudioMessage(data=loud_frame.tobytes()))
        await session.ingest_client_message(raw_loud_msg)
        assert session.metrics.user_frames_in == 2
        assert session.metrics.barge_in_events == 1
        assert session.state == SessionState.INTERRUPTED

        # Wait a moment for agent frames to arrive
        await asyncio.sleep(0.15)
        assert len(received_client_messages) > 0

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

