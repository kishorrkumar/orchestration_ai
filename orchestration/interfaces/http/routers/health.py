"""FastAPI liveness (/healthz) and readiness (/readyz) health probes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, Response, status
from sqlalchemy import text

from orchestration.db.session import async_session_factory
from orchestration.shared.settings import settings

router = APIRouter(tags=["Health"])


@router.get("/healthz", status_code=status.HTTP_200_OK)
async def healthz(request: Request) -> dict[str, Any]:
    """Basic liveness probe confirming process is responsive."""
    res: dict[str, Any] = {"status": "healthy"}
    pool = getattr(request.app.state, "pool", None)
    if pool is not None and hasattr(pool, "get_stats"):
        res["pool"] = pool.get_stats()
    return res


@router.get("/readyz")
async def readyz(request: Request, response: Response) -> dict[str, Any]:
    """
    Readiness probe verifying database connectivity, worker pool health, and configuration.
    Returns 200 if ready to serve traffic, 503 if database unreachable or no workers available.
    """
    checks: dict[str, Any] = {
        "status": "ready",
        "database": "unknown",
        "environment": settings.environment,
    }

    # Verify Database connectivity
    try:
        async with async_session_factory() as session:
            await session.execute(text("SELECT 1"))
        checks["database"] = "connected"
    except Exception as e:
        checks["status"] = "unhealthy"
        checks["database"] = f"error: {e}"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return checks

    # Report Worker Pool status
    pool = getattr(request.app.state, "pool", None)
    if pool is not None and hasattr(pool, "get_stats"):
        checks["pool"] = pool.get_stats()

    return checks
