import asyncio
import numpy as np
import pytest

from orchestration.persona.registry import default_registry
from orchestration.protocol.audio import FRAME_SIZE
from orchestration.protocol.messages import (
    AudioMessage,
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
        persona = default_registry.get("wise_teacher")
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
