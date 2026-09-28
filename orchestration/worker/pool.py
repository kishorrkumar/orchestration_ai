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
from dataclasses import dataclass, field
import logging
import time
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from .client import PersonaPlexWorkerClient, WorkerStatus, WorkerConnectionError

logger = logging.getLogger(__name__)


class WorkerNodeConfig(BaseModel):
    id: str = Field(..., description="Unique worker node identifier")
    host: str = Field(default="localhost", description="Worker host or IP")
    port: int = Field(default=8998, description="Worker port")
    use_ssl: bool = Field(default=False, description="Use WSS if True")
    gpu_id: Optional[int] = Field(default=None, description="GPU device index worker is pinned to")


class PoolCapacityExceededError(Exception):
    pass


class WorkerPool:
    """
    Manages a pool of PersonaPlex worker clients, routing sessions
    to idle workers and managing their full-duplex session leases.
    """

    def __init__(self, wait_timeout: float = 5.0):
        self.wait_timeout = wait_timeout
        self._workers: Dict[str, PersonaPlexWorkerClient] = {}
        self._configs: Dict[str, WorkerNodeConfig] = {}
        self._condition = asyncio.Condition()

    def register_worker(self, config: WorkerNodeConfig) -> PersonaPlexWorkerClient:
        """Register a worker node with the pool."""
        client = PersonaPlexWorkerClient(
            worker_id=config.id,
            host=config.host,
            port=config.port,
            use_ssl=config.use_ssl,
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

    async def acquire_worker(self, session_id: str, timeout: Optional[float] = None) -> PersonaPlexWorkerClient:
        """
        Lease an available IDLE worker.
        If all workers are BUSY, waits until a worker is released or timeout expires.
        """
        wait_limit = timeout if timeout is not None else self.wait_timeout
        deadline = time.time() + wait_limit

        async with self._condition:
            while True:
                # Look for an available worker
                for worker in self._workers.values():
                    if worker.is_available:
                        # Mark as connecting immediately to prevent race conditions
                        worker._status = WorkerStatus.CONNECTING
                        worker._active_session_id = session_id
                        logger.info(f"Leased worker {worker.worker_id} to session {session_id}")
                        return worker

                remaining = deadline - time.time()
                if remaining <= 0:
                    raise PoolCapacityExceededError(
                        f"All {len(self._workers)} PersonaPlex workers are currently saturated. "
                        f"Please retry later."
                    )

                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    raise PoolCapacityExceededError(
                        f"Timed out waiting for available worker ({wait_limit:.1f}s). All workers busy."
                    )

    async def release_worker(self, worker_id: str) -> None:
        """Release a worker back to the IDLE pool."""
        async with self._condition:
            worker = self._workers.get(worker_id)
            if worker:
                await worker.close()
                logger.info(f"Released worker {worker_id} back to IDLE pool")
                self._condition.notify()

    def get_worker(self, worker_id: str) -> Optional[PersonaPlexWorkerClient]:
        return self._workers.get(worker_id)

    def list_workers(self) -> List[dict]:
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
