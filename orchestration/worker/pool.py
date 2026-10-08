"""
Worker Pool Dispatcher for managing PersonaPlex model instances.

Handles:
- Worker instance discovery and registration
- Single-concurrency lease / release mechanics
- Health tracking and auto-recovery
- Capacity limiting and wait queues
"""

from __future__ import annotations

import asyncio
import logging
import time

from pydantic import BaseModel, Field

from .client import PersonaPlexWorkerClient, WorkerStatus

logger = logging.getLogger(__name__)


class WorkerNodeConfig(BaseModel):
    id: str = Field(..., description="Unique worker node identifier")
    host: str = Field(default="localhost", description="Worker host or IP")
    port: int = Field(default=8998, ge=1, le=65535, description="Worker port")
    use_ssl: bool = Field(default=False, description="Use WSS if True")
    use_opus: bool = Field(default=False, description="Use Opus transcoding for upstream moshi server")
    gpu_id: int | None = Field(default=None, description="GPU device index worker is pinned to")


class PoolCapacityExceededError(Exception):
    pass


class WorkerPool:
    """
    Manages a pool of PersonaPlex worker clients, routing sessions
    to idle workers and managing their full-duplex session leases.
    """

    def __init__(self, wait_timeout: float = 30.0, probe_on_acquire: bool = False):
        self.wait_timeout = wait_timeout
        self.probe_on_acquire = probe_on_acquire
        self._workers: dict[str, PersonaPlexWorkerClient] = {}
        self._configs: dict[str, WorkerNodeConfig] = {}
        self._condition = asyncio.Condition()

    def register_worker(self, config: WorkerNodeConfig) -> PersonaPlexWorkerClient:
        """Register a worker node with the pool."""
        if config.id in self._workers:
            existing = self._workers[config.id]
            if not existing.is_available:
                raise ValueError(
                    f"Worker {config.id} is currently busy with session {existing.active_session_id} and cannot be overwritten"
                )

        client = PersonaPlexWorkerClient(
            worker_id=config.id,
            host=config.host,
            port=config.port,
            use_ssl=config.use_ssl,
            use_opus=config.use_opus,
        )
        self._workers[config.id] = client
        self._configs[config.id] = config
        logger.info(f"Registered worker node {config.id} ({config.host}:{config.port})")
        return client

    def unregister_worker(self, worker_id: str) -> None:
        """Remove a worker node from the pool."""
        if worker_id in self._workers:
            del self._workers[worker_id]
            del self._configs[worker_id]
            logger.info(f"Unregistered worker {worker_id}")

    async def acquire_worker(
        self,
        session_id: str,
        timeout: float | None = None,
        probe_health: bool | None = None,
    ) -> PersonaPlexWorkerClient:
        """
        Lease an available IDLE worker.
        Probes worker health before leasing when probe_health is enabled.
        If all workers are BUSY or unhealthy, waits until a worker is released or timeout expires.
        """
        should_probe = self.probe_on_acquire if probe_health is None else probe_health
        wait_limit = timeout if timeout is not None else self.wait_timeout
        deadline = time.time() + wait_limit

        async with self._condition:
            while True:
                # Look for an available worker
                for worker in self._workers.values():
                    if worker.is_available:
                        if should_probe:
                            # Probe worker health before leasing
                            is_healthy = await worker.probe_health()
                            if not is_healthy:
                                worker._status = WorkerStatus.UNHEALTHY
                                logger.warning(
                                    f"Worker {worker.worker_id} failed health probe during acquire; marked UNHEALTHY"
                                )
                                continue

                        # Mark as connecting immediately to prevent race conditions
                        worker._status = WorkerStatus.CONNECTING
                        worker._active_session_id = session_id
                        logger.info(f"Leased worker {worker.worker_id} to session {session_id}")
                        return worker

                remaining = deadline - time.time()
                if remaining <= 0:
                    raise PoolCapacityExceededError(
                        f"All {len(self._workers)} PersonaPlex workers are currently saturated or unhealthy. "
                        f"Please retry later."
                    )

                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=remaining)
                except TimeoutError:
                    raise PoolCapacityExceededError(
                        f"Timed out waiting for available worker ({wait_limit:.1f}s). All workers busy or unhealthy."
                    )

    async def release_worker(self, worker_id: str, success: bool = True) -> None:
        """Release a worker back to the pool. Probes health if call failed."""
        async with self._condition:
            worker = self._workers.get(worker_id)
            if worker:
                if not success:
                    # Probe health before returning to IDLE
                    is_healthy = await worker.probe_health()
                    await worker.close(mark_idle=is_healthy)
                    if not is_healthy:
                        worker._status = WorkerStatus.UNHEALTHY
                        logger.warning(
                            f"Worker {worker_id} failed call and failed health probe; marked UNHEALTHY"
                        )
                    else:
                        logger.info(
                            f"Worker {worker_id} passed health probe after failed call; reset to IDLE"
                        )
                else:
                    await worker.close(mark_idle=True)
                    logger.info(f"Released worker {worker_id} back to IDLE pool")
                self._condition.notify_all()

    def get_worker(self, worker_id: str) -> PersonaPlexWorkerClient | None:
        return self._workers.get(worker_id)

    def list_workers(self) -> list[dict]:
        """Return status information for all registered workers."""
        result = []
        for wid, worker in self._workers.items():
            cfg = self._configs[wid]
            result.append({
                "id": wid,
                "host": cfg.host,
                "port": cfg.port,
                "gpu_id": cfg.gpu_id,
                "status": worker.status.value,
                "active_session": worker.active_session_id,
                "frames_sent": worker.frames_sent,
                "frames_received": worker.frames_received,
                "tokens_received": worker.tokens_received,
            })
        return result

    def get_stats(self) -> dict:
        """Return pool capacity metrics."""
        total = len(self._workers)
        idle = sum(1 for w in self._workers.values() if w.is_available)
        busy = sum(1 for w in self._workers.values() if w.status == WorkerStatus.BUSY)
        unhealthy = sum(1 for w in self._workers.values() if w.status == WorkerStatus.UNHEALTHY)

        return {
            "total_workers": total,
            "idle_workers": idle,
            "busy_workers": busy,
            "unhealthy_workers": unhealthy,
            "utilization_pct": round((busy / total * 100.0) if total > 0 else 0.0, 1),
        }

    async def check_health(self) -> dict[str, bool]:
        """Ping registered workers and transition unhealthy/recovered status."""
        results: dict[str, bool] = {}
        async with self._condition:
            changed = False
            for wid, worker in list(self._workers.items()):
                cfg = self._configs[wid]
                try:
                    _, writer = await asyncio.wait_for(
                        asyncio.open_connection(cfg.host, cfg.port),
                        timeout=0.5,
                    )
                    writer.close()
                    await writer.wait_closed()
                    results[wid] = True
                    if worker.status == WorkerStatus.UNHEALTHY:
                        worker._status = WorkerStatus.IDLE
                        changed = True
                        logger.info(f"Worker {wid} recovered and marked IDLE")
                except Exception:
                    results[wid] = False
                    if worker.is_available:
                        worker._status = WorkerStatus.UNHEALTHY
                        changed = True
                        logger.warning(f"Worker {wid} unreachable, marked UNHEALTHY")
            if changed:
                self._condition.notify_all()
        return results

