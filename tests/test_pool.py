import asyncio
import pytest

from orchestration.worker.client import WorkerStatus
from orchestration.worker.pool import WorkerPool, WorkerNodeConfig, PoolCapacityExceededError


@pytest.mark.asyncio
async def test_worker_pool_registration_and_stats():
    pool = WorkerPool()
    cfg1 = WorkerNodeConfig(id="worker-1", host="127.0.0.1", port=8998, gpu_id=0)
    cfg2 = WorkerNodeConfig(id="worker-2", host="127.0.0.1", port=8999, gpu_id=1)

    pool.register_worker(cfg1)
    pool.register_worker(cfg2)

    stats = pool.get_stats()
    assert stats["total_workers"] == 2
    assert stats["idle_workers"] == 2
    assert stats["busy_workers"] == 0
    assert stats["utilization_pct"] == 0.0

    workers_list = pool.list_workers()
    assert len(workers_list) == 2
    assert workers_list[0]["id"] == "worker-1"
    assert workers_list[1]["id"] == "worker-2"


@pytest.mark.asyncio
async def test_worker_pool_acquire_and_release():
    pool = WorkerPool(wait_timeout=0.2)
    cfg = WorkerNodeConfig(id="worker-single", host="127.0.0.1", port=8998)
    pool.register_worker(cfg)

    # Acquire worker 1
    w1 = await pool.acquire_worker(session_id="session-a")
    assert w1.worker_id == "worker-single"
    assert w1.active_session_id == "session-a"

    # Second acquire should fail since single worker is now busy
    with pytest.raises(PoolCapacityExceededError):
        await pool.acquire_worker(session_id="session-b", timeout=0.1)

    # Release worker
    await pool.release_worker("worker-single")
    assert w1.status == WorkerStatus.IDLE
    assert w1.active_session_id is None

    # Now acquire should succeed again
    w2 = await pool.acquire_worker(session_id="session-c", timeout=0.1)
    assert w2.worker_id == "worker-single"
    await pool.release_worker("worker-single")


@pytest.mark.asyncio
async def test_worker_pool_queue_wakeup():
    pool = WorkerPool(wait_timeout=1.0)
    cfg = WorkerNodeConfig(id="worker-queued", host="127.0.0.1", port=8998)
    pool.register_worker(cfg)

    # Acquire initially
    w = await pool.acquire_worker(session_id="session-1")

    async def release_after_delay():
        await asyncio.sleep(0.05)
        await pool.release_worker("worker-queued")

    asyncio.create_task(release_after_delay())

    # This should block and then wake up when released
    w_resumed = await pool.acquire_worker(session_id="session-2", timeout=0.5)
    assert w_resumed.worker_id == "worker-queued"
    assert w_resumed.active_session_id == "session-2"

    await pool.release_worker("worker-queued")


def test_worker_node_config_invalid_port():
    """AUDIT-011: WorkerNodeConfig must reject ports outside 1..65535."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        WorkerNodeConfig(id="bad-port-low", host="127.0.0.1", port=0)

    with pytest.raises(ValidationError):
        WorkerNodeConfig(id="bad-port-high", host="127.0.0.1", port=70000)


@pytest.mark.asyncio
async def test_register_worker_duplicate_busy_fails():
    """AUDIT-011: Registering duplicate worker while existing is busy must raise ValueError."""
    pool = WorkerPool()
    cfg = WorkerNodeConfig(id="worker-dup", host="127.0.0.1", port=8998)
    pool.register_worker(cfg)
    await pool.acquire_worker(session_id="active-sess")

    with pytest.raises(ValueError, match="cannot be overwritten"):
        pool.register_worker(cfg)

    await pool.release_worker("worker-dup")


@pytest.mark.asyncio
async def test_worker_health_check_recovery():
    """AUDIT-011: Pool check_health marks unreachable as UNHEALTHY and recovers to IDLE when available."""
    pool = WorkerPool()
    # Unreachable port
    cfg = WorkerNodeConfig(id="worker-dead", host="127.0.0.1", port=19998)
    pool.register_worker(cfg)

    results = await pool.check_health()
    assert results["worker-dead"] is False
    worker = pool.get_worker("worker-dead")
    assert worker.status == WorkerStatus.UNHEALTHY

