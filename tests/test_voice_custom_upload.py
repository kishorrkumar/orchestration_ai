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


@pytest.mark.asyncio
async def test_custom_voice_clone_preview_and_delete_lifecycle():
    """Verify end-to-end voice cloning, preview audio retrieval, and deletion."""
    app = create_app(pool=WorkerPool(), registry=PersonaRegistry())
    transport = ASGITransport(app=app)

    # Generate a clean 5-second harmonic vocal WAV for cloning
    sr = 24000
    t = np.linspace(0, 5.0, sr * 5, endpoint=False)
    vocal_audio = (
        0.5 * np.sin(2 * np.pi * 200 * t) +
        0.25 * np.sin(2 * np.pi * 400 * t)
    ).astype(np.float32) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))

    buf = io.BytesIO()
    sf.write(buf, vocal_audio, sr, format="WAV", subtype="PCM_16")
    wav_bytes = buf.getvalue()

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Reject without consent
        files = {"file": ("my_recording.wav", wav_bytes, "audio/wav")}
        data_no_consent = {"name": "Kishore Voice", "consent": "false"}
        res_no_consent = await client.post("/v2/agents/voices/clone", files=files, data=data_no_consent)
        assert res_no_consent.status_code == 400
        assert "consent" in res_no_consent.text.lower()

        # 2. Successfully clone with consent
        files = {"file": ("my_recording.wav", wav_bytes, "audio/wav")}
        data_valid = {"name": "Kishore Live Voice", "consent": "true", "gender": "male"}
        res_clone = await client.post("/v2/agents/voices/clone", files=files, data=data_valid)
        assert res_clone.status_code == 201
        cloned_meta = res_clone.json()
        assert cloned_meta["is_cloned"] is True
        assert "kishorelivevoice" in cloned_meta["id"].lower()
        assert cloned_meta["preview_url"] is not None
        cloned_id = cloned_meta["id"]

        # 3. Check listing contains cloned voice
        res_list = await client.get("/v2/agents/voices")
        assert res_list.status_code == 200
        voices = res_list.json()
        match = next((v for v in voices if v["id"] == cloned_id), None)
        assert match is not None
        assert match["is_cloned"] is True
        assert match["duration_sec"] is not None

        # 4. Preview audio
        res_preview = await client.get(f"/v2/agents/voices/{cloned_id}/preview")
        assert res_preview.status_code == 200
        assert "audio/wav" in res_preview.headers["content-type"]
        assert len(res_preview.content) > 1000

        # 5. Delete cloned voice
        res_del = await client.delete(f"/v2/agents/voices/{cloned_id}")
        assert res_del.status_code == 200
        assert res_del.json()["success"] is True

        # 6. Verify deleted from listing
        res_list_after = await client.get("/v2/agents/voices")
        assert not any(v["id"] == cloned_id for v in res_list_after.json())
