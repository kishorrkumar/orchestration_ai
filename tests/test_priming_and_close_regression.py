import asyncio
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest

from orchestration.persona.registry import PersonaConfig, default_registry
from orchestration.protocol.audio import FRAME_SIZE
from orchestration.protocol.messages import AudioMessage, encode_message
from orchestration.session.manager import SessionManager, SessionState
from orchestration.worker.client import PersonaPlexWorkerClient
from orchestration.worker.mock_worker import PersonaPlexMockServer
from orchestration.worker.pool import WorkerNodeConfig, WorkerPool


@pytest.mark.asyncio
async def test_upstream_query_param_sanitization():
    """Regression test: Only upstream-recognized query params are forwarded in chat URL."""
    client = PersonaPlexWorkerClient(worker_id="test-w", host="127.0.0.1", port=8998)
    persona = PersonaConfig(
        id="test_persona",
        name="Test Persona",
        voice_prompt="NATM0.pt",
        neural_voice="NATF2.pt",
        text_prompt="You enjoy having a good conversation.",
        accent="Indian English",
        character="Conversational",
        call_flow="conversational_companion",
    )

    url = client.build_url(persona)
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)

    # Allowed upstream params
    assert "voice_prompt" in qs
    assert qs["voice_prompt"] == ["NATM0.pt"]
    assert "text_prompt" in qs
    assert qs["text_prompt"][0].startswith("<system>")
    assert qs["text_prompt"][0].endswith("<system>")
    # Exactly two <system> delimiters: start and end
    assert qs["text_prompt"][0].count("<system>") == 2

    # Disallowed upstream params (must be stripped)
    assert "accent" not in qs
    assert "character" not in qs
    assert "call_flow" not in qs
    assert "neural_voice" not in qs


@pytest.mark.asyncio
async def test_priming_delay_and_burst_suppression():
    """Regression test: Gateway gracefully handles multi-second priming delay and flushes queued priming frames."""
    port = 9899
    # Simulate a 0.8s model priming delay before handshake message is sent
    mock = PersonaPlexMockServer(host="127.0.0.1", port=port, prompt_init_delay=0.8, frame_interval_sec=0.04)
    await mock.start()

    try:
        pool = WorkerPool()
        pool.register_worker(WorkerNodeConfig(id="worker-priming-test", host="127.0.0.1", port=port))

        manager = SessionManager(pool=pool)
        persona = default_registry.get("indian_pro")

        session = await manager.create_session(persona=persona, session_id="test-priming-session")
        assert session.state == SessionState.INITIALIZING

        # Start session asynchronously
        start_task = asyncio.create_task(session.start())

        # While session is in PROMPTING (priming delay), simulate client prematurely sending frames
        await asyncio.sleep(0.1)
        assert session.state == SessionState.PROMPTING
        silent_frame = np.zeros(FRAME_SIZE, dtype=np.float32)
        err = await session.ingest_client_message(encode_message(AudioMessage(data=silent_frame.tobytes())))
        # Pre-handshake audio should be rejected or discarded
        assert err is not None

        # Wait for priming to complete
        await start_task
        assert session.state == SessionState.ACTIVE
        # Inbound buffer should be clean (0 burst frames)
        assert session.inbound_buffer.available_frames == 0

        await session.close()
    finally:
        await mock.stop()


@pytest.mark.asyncio
async def test_empty_error_message_prevention():
    """Regression test: Errors on connect never produce empty error messages (e.g. from TimeoutError)."""
    # Point client to a port that does not exist to trigger connection failure
    client = PersonaPlexWorkerClient(worker_id="test-offline-worker", host="127.0.0.1", port=59999, connect_timeout=0.2)
    persona = default_registry.get("indian_pro")

    with pytest.raises(Exception) as exc_info:
        await client.connect("sess-fail-test", persona)

    error_msg = str(exc_info.value)
    # The error message MUST be non-empty and specify the reason/type
    assert len(error_msg.strip()) > 0
    assert "test-offline-worker" in error_msg
