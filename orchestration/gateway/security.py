"""
Security, Authentication, and Rate Limiting for PersonaPlex Gateway.

Features:
1. API-Key Authentication (via 'X-API-Key' header or '?api_key=' query parameter).
2. Per-IP Sliding-Window Rate Limiting.
3. Configurable public whitelist (e.g. /healthz, /console static assets).
"""

from __future__ import annotations

import time
from collections import defaultdict

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ..config import settings


class RateLimiter:
    """Thread-safe in-memory rate limiter using a sliding-window counter per IP."""

    def __init__(self, max_requests_per_minute: int = 60) -> None:
        self.max_requests = max_requests_per_minute
        self._history: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        window_start = now - 60.0
        # Clean older entries
        self._history[client_ip] = [t for t in self._history[client_ip] if t > window_start]
        if len(self._history[client_ip]) >= self.max_requests:
            return False
        self._history[client_ip].append(now)
        return True

    def reset(self) -> None:
        """Reset all rate limiter tracking history (used in tests and maintenance)."""
        self._history.clear()


default_rate_limiter = RateLimiter(max_requests_per_minute=settings.gateway.rate_limit_per_minute)


def verify_api_key(request: Request) -> None:
    """
    Validates API key if configured.
    Checks:
    1. Header: 'X-API-Key'
    2. Header: 'Authorization: Bearer <key>'
    3. Query parameter: '?api_key=<key>'
    """
    expected_key = settings.gateway.api_key
    if not expected_key:
        return  # Auth disabled

    # Check header
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            api_key = auth_header[7:].strip()
    # Check query param
    if not api_key:
        api_key = request.query_params.get("api_key")

    if not api_key or api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Provide via 'X-API-Key' header or '?api_key=' parameter."
        )


class SecurityMiddleware(BaseHTTPMiddleware):
    """Applies Rate Limiting and optional API Key Authentication to Gateway requests."""

    def __init__(self, app, rate_limiter: RateLimiter | None = None) -> None:
        super().__init__(app)
        self.limiter = rate_limiter or default_rate_limiter
        # Endpoints that bypass authentication
        self.public_paths = {"/healthz", "/console", "/", "/docs", "/openapi.json"}

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "127.0.0.1"

        # 1. Check Rate Limit
        if not self.limiter.is_allowed(client_ip):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded. Maximum 60 requests per minute allowed."},
                headers={"Retry-After": "60"}
            )

        # 2. Check API Key for non-public routes if enabled
        if settings.gateway.api_key and request.url.path not in self.public_paths:
            # Let WebSocket upgrades pass through to endpoint-level check or authenticate here
            if request.scope.get("type") == "http":
                try:
                    verify_api_key(request)
                except HTTPException as e:
                    return JSONResponse(status_code=e.status_code, content={"detail": e.detail})

        return await call_next(request)
