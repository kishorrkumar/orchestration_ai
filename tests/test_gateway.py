import asyncio
import numpy as np
import pytest
from httpx import AsyncClient, ASGITransport

from orchestration.gateway.app import create_app
from orchestration.persona.registry import PersonaRegistry, PersonaConfig
from orchestration.worker.pool import WorkerPool, WorkerNodeConfig
from orchestration.worker.mock_worker import PersonaPlexMockServer
from orchestration.session.manager import SessionManager
from orchestration.protocol.messages import (
    MessageType,
    AudioMessage,
    encode_message,
    decode_message,
)
from orchestration.protocol.audio import FRAME_SIZE


@pytest.mark.asyncio
async def test_gateway_rest_routes():
    pool = WorkerPool()
    registry = PersonaRegistry()
    mgr = SessionManager(pool=pool)
    app = create_app(pool=pool, registry=registry, session_manager=mgr)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Healthz
        r = await client.get("/healthz")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert data["pool"]["total_workers"] == 0

        # 2. Metrics
        r = await client.get("/metrics")
        assert r.status_code == 200
        m = r.json()
        assert m["workers_total"] == 0

        # 3. List agents
        r = await client.get("/v1/agents")
        assert r.status_code == 200
        agents = r.json()["agents"]
        assert len(agents) >= 2

        # 4. Register custom agent
        custom_agent = {
            "id": "drone_support",
            "name": "Drone Expert",
            "voice_prompt": "NATM2.pt",
            "text_prompt": "Provide troubleshooting for flight controllers.",
        }
        r = await client.post("/v1/agents", json=custom_agent)
        assert r.status_code == 201
        assert r.json()["status"] == "created"

        r = await client.get("/v1/agents/drone_support")
        assert r.status_code == 200
        assert r.json()["name"] == "Drone Expert"

        # 4b. Update / Save agent
        custom_agent["name"] = "Drone Chief"
        r = await client.put("/v1/agents/drone_support", json=custom_agent)
        assert r.status_code == 200
        assert r.json()["agent"]["name"] == "Drone Chief"

        # 5. Workers registration
        worker_cfg = {
            "id": "gpu-node-0",
            "host": "127.0.0.1",
            "port": 8998,
            "gpu_id": 0,
        }
        r = await client.post("/v1/workers", json=worker_cfg)
        assert r.status_code == 201

        r = await client.get("/v1/workers")
        assert r.status_code == 200
        workers = r.json()["workers"]
        assert len(workers) == 1
        assert workers[0]["id"] == "gpu-node-0"

        # 6. Delete custom agent
        r = await client.delete("/v1/agents/drone_support")
        assert r.status_code == 200

        # 7. Test Voice Presets catalog
        r = await client.get("/v1/voices")
        assert r.status_code == 200
        voices_data = r.json()
        assert "voices" in voices_data
        assert len(voices_data["voices"]) >= 18
        assert any(v["id"] == "NATF2.pt" for v in voices_data["voices"])

        # 7b. Voice detail and preview
        r = await client.get("/v1/voices/NATF0.pt")
        assert r.status_code == 200
        assert r.json()["gender"] == "Female"

        r = await client.get("/v1/voices/NATF0.pt/preview")
        assert r.status_code == 200
        assert "audio/wav" in r.headers["content-type"]
        assert len(r.content) > 1000

        r = await client.get("/v1/voices/NONEXISTENT_VOICE.pt")
        assert r.status_code == 404

        # 7c. Reject persona with invalid voice
        bad_persona = {
            "id": "bad_voice_agent",
            "name": "Bad Agent",
            "voice_ref": "TOTALLY_FAKE_VOICE.pt",
            "system_prompt": "Hello world",
        }
        r = await client.post("/v1/personas", json=bad_persona)
        assert r.status_code == 400
        assert "Invalid voice" in r.json()["detail"]

        # 7d. Prevent deleting official preset
        r = await client.delete("/v1/voices/NATF0.pt")
        assert r.status_code == 400

        # 8. Test RAG knowledge endpoints
        # Clear first
        await client.post("/v1/rag/clear")

        # Ingest text document
        rag_doc = {
            "title": "server_specs.txt",
            "text": "The primary compute cluster has 8 NVIDIA H100 GPUs connected via NVLink with 3.2 Terabits bandwidth.",
        }
        r = await client.post("/v1/rag/text", json=rag_doc)
        assert r.status_code == 200
        doc_info = r.json()["document"]
        doc_id = doc_info["doc_id"]

        # List documents
        r = await client.get("/v1/rag/documents")
        assert r.status_code == 200
        assert len(r.json()["documents"]) == 1

        # Query knowledge base
        r = await client.post("/v1/rag/query", json={"query": "how many H100 GPUs in cluster?", "top_k": 2})
        assert r.status_code == 200
        query_res = r.json()
        assert len(query_res["matches"]) >= 1
        assert "8 NVIDIA H100 GPUs" in query_res["matches"][0]["text"]
        assert query_res["grounded_response"] is not None

        # Delete document
        r = await client.delete(f"/v1/rag/documents/{doc_id}")
        assert r.status_code == 200

        # 9. Test Apple Studio UI (/console and /)
        r = await client.get("/console")
        assert r.status_code == 200
        assert "PersonaPlex Studio" in r.text
        assert "orb-canvas" in r.text

        r = await client.get("/")
        assert r.status_code == 200
        assert "PersonaPlex Studio" in r.text


@pytest.mark.asyncio
async def test_gateway_websocket_realtime():
    port = 9899
    mock = PersonaPlexMockServer(host="127.0.0.1", port=port, prompt_init_delay=0.01, frame_interval_sec=0.03)
    await mock.start()

    try:
        pool = WorkerPool()
        pool.register_worker(WorkerNodeConfig(id="worker-gw-test", host="127.0.0.1", port=port))
        app = create_app(pool=pool)

        # Connect using httpx test client or websocket client via live app
        import uvicorn
        from websockets.asyncio.client import connect as ws_connect

        gw_port = 8765
        config = uvicorn.Config(app, host="127.0.0.1", port=gw_port, log_level="warning")
        server = uvicorn.Server(config)
        server_task = asyncio.create_task(server.serve())

        # Wait for server to spin up
        await asyncio.sleep(0.3)

        url = f"ws://127.0.0.1:{gw_port}/v1/realtime?persona_id=wise_teacher"
        async with ws_connect(url) as ws:
            # 1. First message must be Handshake (0x00)
            msg1 = await ws.recv()
            assert isinstance(msg1, bytes)
            d1 = decode_message(msg1)
            assert d1.type == MessageType.HANDSHAKE

            # 2. Second message is Metadata (0x04) with session_started
            msg2 = await ws.recv()
            d2 = decode_message(msg2)
            assert d2.type == MessageType.METADATA
            assert d2.data["event"] == "session_started"

            # 3. Send audio frames (0x01)
            frame = np.zeros(FRAME_SIZE, dtype=np.float32)
            await ws.send(encode_message(AudioMessage(data=frame.tobytes())))

            # 4. Receive agent audio and text tokens
            got_audio = False
            got_text = False
            for _ in range(5):
                incoming = await asyncio.wait_for(ws.recv(), timeout=2.0)
                d = decode_message(incoming)
                if d.type == MessageType.AUDIO:
                    got_audio = True
                elif d.type == MessageType.TEXT:
                    got_text = True

            assert got_audio

        # Verify worker is released back to pool
        await asyncio.sleep(0.1)
        w = pool.get_worker("worker-gw-test")
        assert w.is_available

        server.should_exit = True
        await server_task

    finally:
        await mock.stop()
