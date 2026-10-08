import asyncio

import numpy as np
import pytest

from orchestration.persona.registry import default_registry
from orchestration.protocol.audio import FRAME_SIZE
from orchestration.protocol.messages import MessageType
from orchestration.worker.client import (
    PersonaPlexWorkerClient,
    WorkerConnectionError,
    WorkerStatus,
)
from orchestration.worker.mock_worker import PersonaPlexMockServer


@pytest.mark.asyncio
async def test_mock_worker_handshake_and_stream():
    port = 9876
    mock = PersonaPlexMockServer(host="127.0.0.1", port=port, prompt_init_delay=0.02, frame_interval_sec=0.02)
    await mock.start()

    try:
        persona = default_registry.get("indian_pro")
        assert persona is not None

        client = PersonaPlexWorkerClient(worker_id="test-worker-1", host="127.0.0.1", port=port)
        assert client.status == WorkerStatus.IDLE

        # Connect
        await client.connect(session_id="sess-100", persona=persona)
        assert client.status == WorkerStatus.BUSY

        # Stream user speech / utterance (pure S2S user-first)
        test_frame = np.zeros(FRAME_SIZE, dtype=np.float32)
        await client.send_audio(test_frame)
        await client.send_text("Hello, is anyone there?")

        # Receive a few messages
        received_audio = 0
        received_text = 0

        async def read_stream():
            nonlocal received_audio, received_text
            async for msg in client.recv_messages():
                if msg.type == MessageType.AUDIO:
                    received_audio += 1
                elif msg.type == MessageType.TEXT:
                    received_text += 1
                if received_audio >= 3 and received_text >= 1:
                    break

        await asyncio.wait_for(read_stream(), timeout=2.0)
        assert received_audio >= 3
        assert received_text >= 1

        await client.close()
        assert client.status == WorkerStatus.IDLE

    finally:
        await mock.stop()


@pytest.mark.asyncio
async def test_mock_worker_single_concurrency_lock():
    port = 9877
    mock = PersonaPlexMockServer(host="127.0.0.1", port=port, prompt_init_delay=0.02, frame_interval_sec=0.04)
    await mock.start()

    try:
        persona = default_registry.get("indian_pro")
        client1 = PersonaPlexWorkerClient(worker_id="client-1", host="127.0.0.1", port=port)
        await client1.connect(session_id="sess-1", persona=persona)
        assert client1.status == WorkerStatus.BUSY

        # Second client attempting to connect to the same worker instance must be rejected
        client2 = PersonaPlexWorkerClient(worker_id="client-2", host="127.0.0.1", port=port, connect_timeout=1.0)
        with pytest.raises(WorkerConnectionError):
            await client2.connect(session_id="sess-2", persona=persona)

        # Disconnect client 1
        await client1.close()
        assert client1.status == WorkerStatus.IDLE
        await asyncio.sleep(0.05)

        # Now client 2 can connect!
        await client2.connect(session_id="sess-2-retry", persona=persona)
        assert client2.status == WorkerStatus.BUSY
        await client2.close()

    finally:
        await mock.stop()
