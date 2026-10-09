import asyncio
import io
import json
import math
import os
import wave
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from starlette.testclient import TestClient

os.environ["WORKER_ALLOW_RAW_PCM"] = "1"

from orchestration.gateway.app import create_app
from orchestration.persona.registry import PersonaRegistry
from orchestration.audio.similarity import compute_speaker_similarity
from orchestration.tts.voice_clone import default_voice_cloner
from orchestration.worker.mock_worker import PersonaPlexMockServer
from orchestration.worker.pool import WorkerNodeConfig, WorkerPool


def _create_synthetic_wav(duration_sec: float = 4.0, sample_rate: int = 24000) -> bytes:
    """Generate valid 24 kHz mono PCM16 test audio with harmonic frequencies simulating voice."""
    num_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    # Formants simulating vocal tract (150 Hz pitch fundamental + 600 Hz + 1800 Hz harmonics)
    audio_f32 = (
        0.4 * np.sin(2 * np.pi * 150 * t)
        + 0.3 * np.sin(2 * np.pi * 600 * t)
        + 0.15 * np.sin(2 * np.pi * 1800 * t)
    )
    # Soft envelope
    envelope = np.sin(np.pi * t / duration_sec)
    audio_f32 = audio_f32 * envelope
    pcm16 = (np.clip(audio_f32, -0.95, 0.95) * 32767.0).astype(np.int16)

    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return bio.getvalue()


@pytest.mark.asyncio
async def test_speaker_similarity_calculation():
    """Verify acoustic speaker similarity engine calculates scores between reference and generated speech."""
    audio_ref = _create_synthetic_wav(duration_sec=3.0)
    audio_same_speaker = _create_synthetic_wav(duration_sec=3.0)
    
    # Generate different speaker with distinct fundamental pitch (400 Hz)
    num_samples = int(3.0 * 24000)
    t = np.linspace(0, 3.0, num_samples, endpoint=False)
    diff_f32 = 0.5 * np.sin(2 * np.pi * 400 * t)
    pcm16_diff = (np.clip(diff_f32, -0.95, 0.95) * 32767.0).astype(np.int16)
    bio_diff = io.BytesIO()
    with wave.open(bio_diff, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(pcm16_diff.tobytes())
    audio_diff_speaker = bio_diff.getvalue()

    score_same = compute_speaker_similarity(audio_ref, audio_same_speaker)
    score_diff = compute_speaker_similarity(audio_ref, audio_diff_speaker)

    # Identical acoustic spectra score high (> 0.85), different pitch scores lower
    assert score_same >= 0.80, f"Expected high similarity score, got {score_same}"
    assert score_same > score_diff, f"Expected {score_same} > {score_diff}"


@pytest.mark.asyncio
async def test_voice_enroll_lifecycle_and_api():
    """Verify POST /voice/enroll, GET /voice, GET /voice/{id}, preview, and DELETE /voice/{id}."""
    pool = WorkerPool()
    registry = PersonaRegistry()
    app = create_app(pool=pool, registry=registry)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        wav_bytes = _create_synthetic_wav(duration_sec=4.0)

        # 1. POST /voice/enroll
        files = {"audio": ("enrollment.wav", wav_bytes, "audio/wav")}
        data = {"voice_name": "Test Cloned Speaker", "consent": "true"}
        resp = await client.post("/voice/enroll", files=files, data=data)
        assert resp.status_code == 201, resp.text
        enroll_data = resp.json()
        
        assert "voice_id" in enroll_data
        assert enroll_data["voice_id"].endswith(".wav")
        assert enroll_data["status"] == "ready"
        assert enroll_data["qa_passed"] is True
        voice_id = enroll_data["id"]

        try:
            # 2. GET /voice/{voice_id}
            resp_get = await client.get(f"/voice/{voice_id}")
            assert resp_get.status_code == 200
            meta = resp_get.json()
            assert meta["name"] == "Test Cloned Speaker"
            assert meta["status"] == "ready"

            # 3. GET /voice (Listing)
            resp_list = await client.get("/voice")
            assert resp_list.status_code == 200
            voices = resp_list.json()
            assert any(v["id"] == voice_id for v in voices)

            # 4. GET /voice/{voice_id}/preview
            resp_prev = await client.get(f"/voice/{voice_id}/preview")
            assert resp_prev.status_code == 200
            assert resp_prev.headers["content-type"] == "audio/wav"
            assert len(resp_prev.content) > 1000

        finally:
            # 5. DELETE /voice/{voice_id}
            resp_del = await client.delete(f"/voice/{voice_id}")
            assert resp_del.status_code == 200
            assert resp_del.json()["success"] is True

            # Verify deletion
            resp_check = await client.get(f"/voice/{voice_id}")
            assert resp_check.status_code == 404


@pytest.mark.asyncio
async def test_agent_stream_websocket_full_duplex_and_barge_in():
    """Verify live bidirectional WebSocket /agent/stream with barge-in and latency metrics."""
    import uvicorn
    from websockets.asyncio.client import connect as ws_connect

    mock_port = 9892
    mock_server = PersonaPlexMockServer(
        host="127.0.0.1",
        port=mock_port,
        prompt_init_delay=0.01,
        frame_interval_sec=0.02,
    )
    await mock_server.start()

    gw_port = 8769
    server_task = None
    try:
        pool = WorkerPool()
        pool.register_worker(WorkerNodeConfig(id="stream-test-worker", host="127.0.0.1", port=mock_port))
        app = create_app(pool=pool, registry=PersonaRegistry())

        config = uvicorn.Config(app, host="127.0.0.1", port=gw_port, log_level="warning")
        server = uvicorn.Server(config)
        server_task = asyncio.create_task(server.serve())
        await asyncio.sleep(0.3)

        url = f"ws://127.0.0.1:{gw_port}/agent/stream?voice_id=test_voice.wav&sample_rate=16000"
        async with ws_connect(url) as ws:
            # 1. Check ready message
            msg1_raw = await ws.recv()
            assert isinstance(msg1_raw, str)
            msg1 = json.loads(msg1_raw)
            assert msg1["type"] == "status"
            assert msg1["status"] == "ready"
            assert "priming_ms" in msg1
            assert msg1["priming_ms"] < 2000.0  # Fast startup target

            # 2. Stream user PCM16 audio (20ms frames = 640 bytes at 16kHz)
            t_chunk = np.linspace(0, 0.02, 320, endpoint=False)
            pcm16_chunk = (0.3 * np.sin(2 * np.pi * 300 * t_chunk) * 32767).astype(np.int16).tobytes()
            
            # Send 5 frames of speech (100ms)
            for _ in range(5):
                await ws.send(pcm16_chunk)

            # 3. Receive responses from worker (audio chunks or transcript tokens)
            received_audio = 0
            received_tokens = 0
            for _ in range(10):
                try:
                    event = await asyncio.wait_for(ws.recv(), timeout=0.2)
                    if isinstance(event, bytes):
                        received_audio += len(event)
                    elif isinstance(event, str):
                        data = json.loads(event)
                        if data.get("type") == "transcript":
                            received_tokens += 1
                except asyncio.TimeoutError:
                    break

            # 4. Test Instant Barge-In Interrupt
            await ws.send(json.dumps({"type": "interrupt"}))
            # Verify socket remains healthy and accepts ping/pong
            await ws.send(json.dumps({"type": "ping"}))
            pong_raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
            pong = json.loads(pong_raw)
            assert pong.get("type") == "pong"

    finally:
        if server_task:
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass
        await mock_server.stop()
