"""
Unit tests for custom voice prompt upload and discovery (/v2/agents/voices).
"""

import io
import pathlib
import numpy as np
import pytest
import soundfile as sf
from httpx import ASGITransport, AsyncClient

from orchestration.gateway.app import create_app
from orchestration.persona.registry import PersonaRegistry
from orchestration.worker.pool import WorkerPool


@pytest.mark.asyncio
async def test_custom_voice_upload_and_listing():
    """Verify uploading a 24 kHz WAV adds a custom voice preset to /v2/agents/voices."""
    app = create_app(pool=WorkerPool(), registry=PersonaRegistry())
    transport = ASGITransport(app=app)

    # 1. Create a 3-second 24 kHz mono test WAV
    sr = 24000
    t = np.linspace(0, 3.0, sr * 3, endpoint=False)
    sine_audio = (np.sin(2 * np.pi * 300 * t) * 0.5).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, sine_audio, sr, format="WAV", subtype="PCM_16")
    wav_bytes = buf.getvalue()

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Check initial voices
        res = await client.get("/v2/agents/voices")
        assert res.status_code == 200
        initial_voices = res.json()
        assert len(initial_voices) >= 18

        # Upload custom voice
        files = {"file": ("aarav_custom_ref.wav", wav_bytes, "audio/wav")}
        data = {
            "name": "Aarav Indian Reference",
            "accent": "Indian English",
            "gender": "male",
        }
        upload_res = await client.post("/v2/agents/voices/upload", files=files, data=data)
        assert upload_res.status_code == 201
        uploaded = upload_res.json()
        assert "aarav_custom_ref.wav" in uploaded["id"]
        assert "Aarav Indian Reference" in uploaded["name"]
        assert uploaded["accent"] == "Indian English"

        # Verify it now appears in /v2/agents/voices
        res2 = await client.get("/v2/agents/voices")
        assert res2.status_code == 200
        all_voices = res2.json()
        assert any(v["id"] == "aarav_custom_ref.wav" for v in all_voices)

    # Cleanup test file
    test_file = pathlib.Path("voices/aarav_custom_ref.wav")
    if test_file.exists():
        test_file.unlink()
