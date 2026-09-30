import asyncio
import pytest
from httpx import AsyncClient, ASGITransport
from orchestration.gateway.app import create_app
from orchestration.gateway.security import RateLimiter, SecurityMiddleware
from orchestration.persona.registry import PersonaRegistry
from orchestration.worker.pool import WorkerPool
from orchestration.session.manager import SessionManager
from orchestration.config import settings


@pytest.mark.asyncio
async def test_rate_limiter_blocks_excessive_traffic():
    limiter = RateLimiter(max_requests_per_minute=5)
    ip = "192.168.1.100"

    # 5 requests pass
    for _ in range(5):
        assert limiter.is_allowed(ip) is True

    # 6th request fails
    assert limiter.is_allowed(ip) is False


@pytest.mark.asyncio
async def test_api_key_authentication_enforcement():
    # Set test API key
    settings.gateway.api_key = "secret-test-key-12345"

    pool = WorkerPool()
    registry = PersonaRegistry()
    mgr = SessionManager(pool=pool)
    app = create_app(pool=pool, registry=registry, session_manager=mgr)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Public route /healthz succeeds without API key
        r = await client.get("/healthz")
        assert r.status_code == 200

        # 2. Protected route without API key fails with 401
        r = await client.get("/v1/agents")
        assert r.status_code == 401
        assert "Invalid or missing API key" in r.json()["detail"]

        # 3. Protected route with incorrect API key fails with 401
        r = await client.get("/v1/agents", headers={"X-API-Key": "wrong-key"})
        assert r.status_code == 401

        # 4. Protected route with valid X-API-Key succeeds
        r = await client.get("/v1/agents", headers={"X-API-Key": "secret-test-key-12345"})
        assert r.status_code == 200

        # 5. Protected route with Bearer token succeeds
        r = await client.get("/v1/agents", headers={"Authorization": "Bearer secret-test-key-12345"})
        assert r.status_code == 200

        # 6. Protected route with ?api_key query param succeeds
        r = await client.get("/v1/agents?api_key=secret-test-key-12345")
        assert r.status_code == 200

    # Reset API key
    settings.gateway.api_key = None


@pytest.mark.asyncio
async def test_websocket_api_key_authentication_enforcement():
    """AUDIT-005: /v1/realtime WebSocket must enforce API key validation when configured."""
    from websockets.asyncio.client import connect as ws_connect
    import uvicorn
    import websockets.exceptions

    settings.gateway.api_key = "ws-secret-key-999"
    try:
        pool = WorkerPool()
        registry = PersonaRegistry()
        mgr = SessionManager(pool=pool)
        app = create_app(pool=pool, registry=registry, session_manager=mgr)

        gw_port = 8767
        config = uvicorn.Config(app, host="127.0.0.1", port=gw_port, log_level="warning")
        server = uvicorn.Server(config)
        server_task = asyncio.create_task(server.serve())
        await asyncio.sleep(0.3)

        # 1. Connecting without API key should be rejected
        url_no_key = f"ws://127.0.0.1:{gw_port}/v1/realtime?persona_id=indian_pro"
        with pytest.raises((websockets.exceptions.InvalidStatus, websockets.exceptions.ConnectionClosed)):
            async with ws_connect(url_no_key) as ws:
                await ws.recv()

        # 2. Connecting with wrong API key should be rejected
        url_wrong_key = f"ws://127.0.0.1:{gw_port}/v1/realtime?persona_id=indian_pro&api_key=wrong"
        with pytest.raises((websockets.exceptions.InvalidStatus, websockets.exceptions.ConnectionClosed)):
            async with ws_connect(url_wrong_key) as ws:
                await ws.recv()

        # 3. Connecting with correct API key should succeed
        url_valid_key = f"ws://127.0.0.1:{gw_port}/v1/realtime?persona_id=indian_pro&api_key=ws-secret-key-999"
        async with ws_connect(url_valid_key) as ws:
            # First frame received is handshake (0x00)
            frame = await ws.recv()
            assert frame[0] == 0x00

        server.should_exit = True
        await server_task
    finally:
        settings.gateway.api_key = None

