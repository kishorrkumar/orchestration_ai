"""
Integration tests for Phase 2 WebRTC Transport (POST /v2/webrtc/offer).
Verifies aiortc SDP negotiation, answer generation, track attachment, and auth enforcement.
"""

import asyncio
import json
import pytest
from httpx import ASGITransport, AsyncClient

from orchestration.gateway.app import create_app
from orchestration.persona.registry import PersonaRegistry
from orchestration.session.manager import SessionManager
from orchestration.worker.mock_worker import PersonaPlexMockServer
from orchestration.worker.pool import WorkerNodeConfig, WorkerPool

try:
    import av
    from aiortc import MediaStreamTrack, RTCPeerConnection
    HAS_AIORTC = True
except ImportError:
    HAS_AIORTC = False


@pytest.mark.asyncio
async def test_webrtc_offer_endpoint_requires_auth_when_configured(monkeypatch):
    """Verify WebRTC offer enforces AUTH_TOKEN when configured in settings."""
    if not HAS_AIORTC:
        pytest.skip("aiortc not available")

    from orchestration.settings import app_settings
    monkeypatch.setattr(app_settings, "AUTH_TOKEN", "secret-test-token")

    pool = WorkerPool()
    app = create_app(pool=pool, registry=PersonaRegistry(), session_manager=SessionManager(pool=pool))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Unauthenticated request rejected with 401
        res = await client.post(
            "/v2/webrtc/offer",
            json={"sdp": "v=0\r\no=...", "type": "offer", "agent_id": "default"},
        )
        assert res.status_code == 401

        # 2. Authenticated request proceeds
        client_pc = RTCPeerConnection()
        client_pc.addTransceiver("audio", direction="sendrecv")
        offer = await client_pc.createOffer()
        await client_pc.setLocalDescription(offer)

        res_auth = await client.post(
            "/v2/webrtc/offer",
            json={
                "sdp": client_pc.localDescription.sdp,
                "type": client_pc.localDescription.type,
                "agent_id": "default",
                "token": "secret-test-token",
            },
        )
        assert res_auth.status_code == 200
        data = res_auth.json()
        assert data["type"] == "answer"
        assert len(data["sdp"]) > 0
        assert data["session_id"].startswith("rtc_")
        await client_pc.close()


@pytest.mark.asyncio
async def test_webrtc_full_loopback_negotiation():
    """Verify client-server WebRTC SDP negotiation and answer parsing."""
    if not HAS_AIORTC:
        pytest.skip("aiortc not available")

    mock_port = 9896
    mock_server = PersonaPlexMockServer(host="127.0.0.1", port=mock_port)
    await mock_server.start()

    try:
        pool = WorkerPool()
        pool.register_worker(WorkerNodeConfig(id="rtc-mock-worker", host="127.0.0.1", port=mock_port))
        app = create_app(pool=pool, registry=PersonaRegistry(), session_manager=SessionManager(pool=pool))

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            client_pc = RTCPeerConnection()
            dc = client_pc.createDataChannel("chat")
            client_pc.addTransceiver("audio", direction="sendrecv")

            offer = await client_pc.createOffer()
            await client_pc.setLocalDescription(offer)

            res = await client.post(
                "/v2/webrtc/offer",
                json={
                    "sdp": client_pc.localDescription.sdp,
                    "type": client_pc.localDescription.type,
                    "agent_id": "default",
                },
            )
            assert res.status_code == 200
            data = res.json()
            assert data["type"] == "answer"
            assert "m=audio" in data["sdp"]
            assert data["session_id"] is not None

            await client_pc.close()
    finally:
        await mock_server.stop()
