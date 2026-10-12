"""
Unit & Integration Tests for Phase 1: Core Robustness & Lifecycle.

Validates:
1. Dual idle/silence timeouts (soft ping prompt, hard teardown, user speech timer reset).
2. WorkerPool bounded queue backpressure and fast capacity rejection.
3. Worker client connect retry with exponential backoff.
4. MetricsCollector latency percentiles, frame drops, active sessions gauge.
5. Structured logging correlation contextvars binding.
6. Health (/healthz), Readiness (/readyz), and Metrics (/metrics) endpoints.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from orchestration.gateway.app import create_app
from orchestration.pipeline.end_detector import EndOfCallDetector
from orchestration.shared.logging import bind_session_context, clear_session_context
from orchestration.telemetry.metrics import MetricsCollector, metrics_collector
from orchestration.worker.client import PersonaPlexWorkerClient, WorkerConnectionError, WorkerStatus
from orchestration.worker.pool import PoolCapacityExceededError, WorkerNodeConfig, WorkerPool


# =====================================================================
# 1. Dual Idle/Silence Timeouts
# =====================================================================

def test_dual_silence_timeouts_soft_then_hard():
    """Verify soft silence nudge triggers first, then hard silence triggers hangup."""
    detector = EndOfCallDetector(
        ending_text="goodbye",
        silence_timeout_sec=20.0,      # hard timeout
        soft_silence_timeout_sec=10.0, # soft nudge
    )
    t0 = 1000.0
    detector.start_session(t0)

    # At 5s silence: neither triggered
    assert detector.check_soft_silence(t0 + 5.0) is False
    assert detector.check_termination(t0 + 5.0) is None
    assert detector.get_silence_duration(t0 + 5.0) == 5.0

    # At 10.1s silence: soft silence triggers
    assert detector.check_soft_silence(t0 + 10.1) is True
    # Calling again before speech should NOT re-trigger (triggered once per epoch)
    assert detector.check_soft_silence(t0 + 10.5) is False
    # Hard termination not yet reached
    assert detector.check_termination(t0 + 10.5) is None

    # At 20.1s silence: hard termination triggers
    term = detector.check_termination(t0 + 20.1)
    assert term == "silence_timeout"


def test_dual_silence_reset_on_user_speech():
    """Verify user speaking resets both soft and hard silence timers."""
    detector = EndOfCallDetector(
        ending_text="goodbye",
        silence_timeout_sec=20.0,
        soft_silence_timeout_sec=10.0,
    )
    t0 = 1000.0
    detector.start_session(t0)

    # Advance to 12s -> soft silence triggers
    assert detector.check_soft_silence(t0 + 12.0) is True

    # User speaks at 15s
    detector.on_user_speech(t0 + 15.0)
    assert detector.get_silence_duration(t0 + 15.0) == 0.0

    # Soft silence should not trigger immediately
    assert detector.check_soft_silence(t0 + 16.0) is False

    # After another 10s from speech (t0 + 25.1s), soft silence can trigger again
    assert detector.check_soft_silence(t0 + 25.1) is True

    # User finishes speaking, timer resets again
    detector.on_user_speech_end(t0 + 26.0)
    assert detector.get_silence_duration(t0 + 26.0) == 0.0


# =====================================================================
# 2. Worker Pool Bounded Queue Backpressure
# =====================================================================

@pytest.mark.asyncio
async def test_bounded_worker_pool_backpressure_rejection():
    """Verify WorkerPool rejects excess requests when queue depth limit is exceeded."""
    pool = WorkerPool(wait_timeout=2.0, max_queue_size=2)
    # Register 1 worker
    cfg = WorkerNodeConfig(id="worker-gpu-1", host="127.0.0.1", port=8998)
    client = pool.register_worker(cfg)

    # Lease worker to session 1
    w1 = await pool.acquire_worker(session_id="session-1")
    assert w1.status == WorkerStatus.CONNECTING
    assert pool.get_stats()["busy_workers"] == 0  # Still connecting

    # Simulate worker becoming busy
    w1._status = WorkerStatus.BUSY

    # Fill the wait queue with 2 tasks waiting
    waiter1_task = asyncio.create_task(pool.acquire_worker(session_id="waiter-1", timeout=5.0))
    waiter2_task = asyncio.create_task(pool.acquire_worker(session_id="waiter-2", timeout=5.0))
    await asyncio.sleep(0.05)

    stats = pool.get_stats()
    assert stats["waiting_queue_depth"] == 2
    assert stats["max_queue_size"] == 2

    # A 3rd waiter should immediately be rejected with PoolCapacityExceededError (backpressure)
    with pytest.raises(PoolCapacityExceededError, match="Worker wait queue saturated"):
        await pool.acquire_worker(session_id="waiter-3", timeout=5.0)

    # Cancel background waiters
    waiter1_task.cancel()
    waiter2_task.cancel()
    await asyncio.gather(waiter1_task, waiter2_task, return_exceptions=True)

    # Release worker back to IDLE
    await pool.release_worker("worker-gpu-1", success=True)
    assert pool.get_stats()["idle_workers"] == 1


# =====================================================================
# 3. Worker Client Connection Retry with Backoff
# =====================================================================

@pytest.mark.asyncio
async def test_worker_connect_retry_exponential_backoff():
    """Verify connect_with_retry retries on transient connection error before succeeding."""
    client = PersonaPlexWorkerClient(worker_id="test-retry", host="127.0.0.1", port=9999)

    attempts = 0

    async def mock_connect(session_id, persona):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionRefusedError(f"Simulated network drop attempt {attempts}")
        client._status = WorkerStatus.BUSY

    with patch.object(client, "connect", side_effect=mock_connect):
        from orchestration.persona.registry import PersonaConfig
        persona = PersonaConfig(id="agent-1", name="Aarav", voice_prompt="NATM1.pt", system_prompt="Hello")

        # Call with 3 attempts and 0.01s backoff for fast testing
        await client.connect_with_retry(
            session_id="sess-retry-1",
            persona=persona,
            max_attempts=3,
            backoff_sec=0.01,
        )

        assert attempts == 3
        assert client.status == WorkerStatus.BUSY


@pytest.mark.asyncio
async def test_worker_connect_retry_exhaustion_raises():
    """Verify connect_with_retry raises WorkerConnectionError if all retries fail."""
    client = PersonaPlexWorkerClient(worker_id="test-fail", host="127.0.0.1", port=9999)

    with patch.object(client, "connect", side_effect=ConnectionRefusedError("Host unreachable")):
        from orchestration.persona.registry import PersonaConfig
        persona = PersonaConfig(id="agent-1", name="Aarav", voice_prompt="NATM1.pt", system_prompt="Hello")

        with pytest.raises(WorkerConnectionError, match="failed to connect after 2 attempts"):
            await client.connect_with_retry(
                session_id="sess-fail-1",
                persona=persona,
                max_attempts=2,
                backoff_sec=0.01,
            )


# =====================================================================
# 4. Metrics Collector Latency Percentiles & Gauges
# =====================================================================

def test_metrics_collector_percentiles_and_telemetry():
    """Verify MetricsCollector correctly calculates p50, p95, p99 and frame counts."""
    collector = MetricsCollector()
    collector.reset()

    collector.record_session_start()
    assert collector.get_summary()["active_sessions"] == 1

    # Record 100 latency steps from 10ms to 109ms
    for lat in range(10, 110):
        collector.record_frame_step_latency(float(lat))

    collector.record_ttfa(450.0)
    collector.record_frames(in_delta=100, out_delta=95, drop_delta=2)
    collector.record_barge_in()
    collector.record_retry()
    collector.record_session_end(reason="user_hangup")

    summary = collector.get_summary(pool_stats={"total_workers": 2, "idle_workers": 1, "busy_workers": 1})

    assert summary["active_sessions"] == 0
    assert summary["sessions_total_started"] == 1
    assert summary["sessions_total_completed"] == 1
    assert summary["disconnect_reasons"]["user_hangup"] == 1

    # Percentiles for uniform range 10..109 (size 100)
    assert 58.0 <= summary["latency_ms"]["p50_frame_step"] <= 61.0
    assert 103.0 <= summary["latency_ms"]["p95_frame_step"] <= 106.0
    assert 107.0 <= summary["latency_ms"]["p99_frame_step"] <= 109.0
    assert summary["latency_ms"]["p50_ttfa"] == 450.0

    assert summary["frames"]["inbound_total"] == 100
    assert summary["frames"]["dropped_total"] == 2
    assert summary["frames"]["barge_in_events"] == 1
    assert summary["retries_total"] == 1
    assert summary["worker_pool"]["total_workers"] == 2


# =====================================================================
# 5. Structured Logging Correlation Binding
# =====================================================================

def test_structured_logging_session_context_binding():
    """Verify bind_session_context and clear_session_context execute cleanly."""
    bind_session_context(session_id="call-test-123", agent_id="agent-aarav", caller_ip="127.0.0.1")
    clear_session_context()


# =====================================================================
# 6. API Probes (/healthz, /readyz, /metrics)
# =====================================================================

def test_health_ready_and_metrics_endpoints():
    """Verify FastAPI /healthz, /readyz, and /metrics endpoints respond with 200 OK."""
    app = create_app()
    client = TestClient(app)

    # 1. /healthz
    res = client.get("/healthz")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

    # 2. /readyz
    res = client.get("/readyz")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ready"
    assert data["database"] == "connected"

    # 3. /metrics
    res = client.get("/metrics")
    assert res.status_code == 200
    metrics_data = res.json()
    assert "active_sessions" in metrics_data
    assert "latency_ms" in metrics_data
    assert "worker_pool" in metrics_data
    assert "hardware" in metrics_data
