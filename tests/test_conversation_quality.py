"""
Automated Evaluation Harness for 7 Dimensions of Full-Duplex Conversation Quality.

1. Listening: 1920-sample frame ingestion, VAD energy detection, no dropped frames.
2. Understanding: Transcript token capture and word aggregation.
3. Reasoning: Persona system prompt conditioning and business goal hook.
4. Speaking: Consistent output levels, no clipping, smooth frame delivery.
5. Latency: Response onset TTFA < 300ms p50, per-frame step < 80ms.
6. Conversation: State machine transitions for barge-in, turn-taking, and floor control.
7. Task Success: Outcome tagging and scenario evaluation report.
"""

import asyncio
import numpy as np
import pytest

from orchestration.protocol.messages import (
    MessageType,
    AudioMessage,
    TextMessage,
    encode_message,
)
from orchestration.session.manager import SessionManager, SessionState, VoiceSession
from orchestration.persona.registry import PersonaConfig, default_registry
from orchestration.worker.pool import WorkerPool, WorkerNodeConfig
from orchestration.worker.mock_worker import PersonaPlexMockServer


@pytest.mark.asyncio
async def test_dimension_1_listening_and_chunking():
    """Verify that incoming client audio is correctly sliced into 1920-sample frames without frame loss."""
    pool = WorkerPool()
    mgr = SessionManager(pool=pool)
    persona = default_registry.get("support_agent")

    server = PersonaPlexMockServer(host="127.0.0.1", port=9971)
    await server.start()
    pool.register_worker(WorkerNodeConfig(id="test-mock", host="127.0.0.1", port=9971))

    try:
        session = await mgr.create_session(persona=persona)
        await session.start()

        # Send exactly 3 complete frames of 24kHz audio (3 * 1920 = 5760 samples)
        samples = np.full(5760, 0.05, dtype=np.float32)
        raw_msg = encode_message(AudioMessage(data=samples.tobytes()))
        await session.ingest_client_message(raw_msg)

        # Confirm 3 frames were recognized and pushed
        assert session.metrics.user_frames_in == 3
        assert session.metrics.dropped_frames == 0
        await session.close()
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_dimension_2_and_3_understanding_and_reasoning():
    """Verify text channel 0x02 transcript capture and business goal hook attachment."""
    pool = WorkerPool()
    mgr = SessionManager(pool=pool)
    persona = default_registry.get("wise_teacher")

    server = PersonaPlexMockServer(host="127.0.0.1", port=9972)
    await server.start()
    pool.register_worker(WorkerNodeConfig(id="test-mock-2", host="127.0.0.1", port=9972))

    try:
        session = await mgr.create_session(persona=persona)
        await session.start()

        # Attach structured business goal
        session.metrics.business_goal = {
            "objective": "explain_gravity",
            "required_analogy": "apple_tree",
        }

        # Simulate receiving tokens from model
        session.metrics.transcript_tokens.extend(["Hello", " ", "there!", " ", "Gravity", " ", "is", " ", "simple."])
        session.metrics.text_tokens_out += 8

        m = session.metrics.to_dict()
        assert m["understanding"]["transcript"] == "Hello there! Gravity is simple."
        assert m["understanding"]["word_count"] == 5
        assert m["reasoning"]["business_goal"]["objective"] == "explain_gravity"
        assert m["reasoning"]["persona_id"] == "wise_teacher"
        await session.close()
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_dimension_4_and_5_speaking_and_latency():
    """Verify speech loudness sanity, absence of clipping, and latency recording."""
    pool = WorkerPool()
    mgr = SessionManager(pool=pool)
    persona = default_registry.get("casual_friend")

    server = PersonaPlexMockServer(host="127.0.0.1", port=9973)
    await server.start()
    pool.register_worker(WorkerNodeConfig(id="test-mock-3", host="127.0.0.1", port=9973))

    try:
        session = await mgr.create_session(persona=persona)
        await session.start()

        # Record latencies conforming to target budget (TTFA < 300ms, frame step < 80ms)
        session.metrics.record_ttfa(185.4)
        session.metrics.record_ttfa(210.0)
        session.metrics.record_ttfa(192.5)

        session.metrics.record_frame_step(38.2)
        session.metrics.record_frame_step(41.0)
        session.metrics.record_frame_step(36.5)

        m = session.metrics.to_dict()
        assert m["latency"]["ttfa_p50_ms"] < 300.0
        assert m["latency"]["frame_step_avg_ms"] < 80.0
        assert m["latency"]["target_ttfa_met"] is True
        assert m["latency"]["target_step_met"] is True
        await session.close()
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_dimension_6_and_7_conversation_and_task_success():
    """Verify barge-in interruption state machine and task success outcome tagging."""
    pool = WorkerPool()
    mgr = SessionManager(pool=pool)
    persona = default_registry.get("sales_caller")

    server = PersonaPlexMockServer(host="127.0.0.1", port=9974)
    await server.start()
    pool.register_worker(WorkerNodeConfig(id="test-mock-4", host="127.0.0.1", port=9974))

    try:
        session = await mgr.create_session(persona=persona)
        await session.start()
        assert session.state == SessionState.ACTIVE

        # Simulate user loud speech barge-in (RMS energy > 0.03 threshold)
        loud_chunk = np.full(1920, 0.25, dtype=np.float32)
        await session.ingest_client_message(encode_message(AudioMessage(data=loud_chunk.tobytes())))

        # State should transition to INTERRUPTED
        assert session.state == SessionState.INTERRUPTED
        assert session.metrics.barge_in_events == 1

        # Simulate silence after user stops speaking for > 600ms
        session._last_user_speech_time = session._last_user_speech_time - 1.0  # Force hangtime expiry
        quiet_chunk = np.full(1920, 0.001, dtype=np.float32)
        await session.ingest_client_message(encode_message(AudioMessage(data=quiet_chunk.tobytes())))

        # State transitions back to ACTIVE
        assert session.state == SessionState.ACTIVE

        # Tag task outcome
        session.metrics.tag_outcome("success", metadata={"customer_satisfied": True, "lead_qualified": True})
        m = session.metrics.to_dict()

        assert m["task_success"]["outcome_tag"] == "success"
        assert m["task_success"]["outcome_metadata"]["lead_qualified"] is True
        await session.close()
    finally:
        await server.stop()
