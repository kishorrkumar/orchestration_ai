"""
Unit & Integration Tests for Sarvam AI Engine B (Saaras STT, Sarvam LLM, Bulbul TTS).
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from orchestration.domain.providers.catalog import ProviderCatalog
from orchestration.infrastructure.engines.sarvam_engine import (
    SarvamClient,
    resolve_sarvam_api_key,
)
from orchestration.infrastructure.vault.tester import (
    test_provider_connection as check_provider_connection,
)
from orchestration.settings import app_settings


@pytest.mark.asyncio
async def test_resolve_sarvam_api_key_passed_key_precedence():
    """Verify passed UI key takes highest precedence."""
    with patch.dict(os.environ, {"SARVAM_API_KEY": "env_sarvam_key"}):
        resolved = await resolve_sarvam_api_key(passed_key="pasted_sarvam_key_999")
        assert resolved == "pasted_sarvam_key_999"


@pytest.mark.asyncio
async def test_resolve_sarvam_api_key_from_env_vars():
    """Verify resolution falls back to SARVAM_API_KEY env var."""
    with patch.dict(os.environ, {"SARVAM_API_KEY": "env_sarvam_key"}):
        resolved = await resolve_sarvam_api_key(passed_key=None)
        assert resolved == "env_sarvam_key"


@pytest.mark.asyncio
async def test_resolve_sarvam_api_key_none_when_empty():
    """Verify resolution returns None when no keys exist."""
    with patch.dict(os.environ, {"SARVAM_API_KEY": ""}, clear=True):
        with patch.object(app_settings, "SARVAM_API_KEY", None):
            resolved = await resolve_sarvam_api_key(passed_key="")
            assert resolved is None


def test_provider_catalog_loads_sarvam_manifests():
    """Verify ProviderCatalog loads sarvam_stt, sarvam_llm, and sarvam_tts."""
    catalog = ProviderCatalog()
    sarvam_stt = catalog.get("sarvam_stt")
    assert sarvam_stt is not None
    assert sarvam_stt.kind == "stt"
    assert sarvam_stt.default_model == "saaras:v4"
    assert "hi-IN" in sarvam_stt.supported_languages

    sarvam_llm = catalog.get("sarvam_llm")
    assert sarvam_llm is not None
    assert sarvam_llm.kind == "llm"
    assert sarvam_llm.default_model == "sarvam-105b-conversations"

    sarvam_tts = catalog.get("sarvam_tts")
    assert sarvam_tts is not None
    assert sarvam_tts.kind == "tts"
    assert sarvam_tts.default_model == "bulbul:v4-flash"
    assert any(v.id == "shubh" for v in sarvam_tts.voices)
    assert any(v.id == "ritu" for v in sarvam_tts.voices)


@pytest.mark.asyncio
async def test_vault_tester_sarvam_valid():
    """Verify vault connection tester verifies Sarvam successfully."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        res = await check_provider_connection("sarvam_stt", "test-sarvam-key")
        assert res.status == "valid"
        assert res.latency_ms > 0

        res_llm = await check_provider_connection("sarvam_llm", "test-sarvam-key")
        assert res_llm.status == "valid"

        res_tts = await check_provider_connection("sarvam_tts", "test-sarvam-key")
        assert res_tts.status == "valid"


@pytest.mark.asyncio
async def test_sarvam_client_stt_transcribe():
    """Verify SarvamClient generates valid WAV and parses transcript response."""
    client = SarvamClient(api_key="test_key_123")
    fake_pcm = b"\x00\x00" * 3200  # 200ms of audio

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"transcript": "नमस्ते, मैं आपकी कैसे सहायता कर सकता हूँ?"}

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        transcript = await client.transcribe(fake_pcm, sample_rate=16000, language_code="hi-IN", model="saaras:v4")
        assert transcript == "नमस्ते, मैं आपकी कैसे सहायता कर सकता हूँ?"
        assert mock_post.called
        call_kwargs = mock_post.call_args[1]
        assert "api-subscription-key" in call_kwargs["headers"]


@pytest.mark.asyncio
async def test_sarvam_client_tts_synthesize():
    """Verify SarvamClient synthesizes audio and decodes base64 payload."""
    import base64
    import io
    import wave

    # Create dummy WAV payload
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x10\x00" * 160)
    fake_wav_bytes = wav_buf.getvalue()
    b64_str = base64.b64encode(fake_wav_bytes).decode("ascii")

    client = SarvamClient(api_key="test_key_123")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"audios": [b64_str]}

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        raw_pcm = await client.synthesize(
            text="Welcome to Snapserve",
            speaker="shubh",
            target_language_code="en-IN",
            model="bulbul:v4-flash",
            sample_rate=16000,
        )
        assert len(raw_pcm) == 320  # 160 frames * 2 bytes
