"""
Unit and Integration Tests for Google Gemini Multimodal Live API & UI Paste Key Integration.
"""

from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from orchestration.infrastructure.engines.gemini_live import (
    GeminiLiveSession,
    resolve_gemini_api_key,
)
from orchestration.audio.resample import resample_oneshot
from orchestration.tools import default_registry, ToolResult


# ---------------------------------------------------------------------------
# Key Resolution Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_resolve_gemini_api_key_passed_key_precedence():
    """Verify explicitly passed key (UI paste option) takes highest priority."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "env_key_123", "GOOGLE_API_KEY": "google_key_456"}):
        resolved = await resolve_gemini_api_key(passed_key="pasted_ui_key_789")
        assert resolved == "pasted_ui_key_789"


@pytest.mark.asyncio
async def test_resolve_gemini_api_key_from_env_vars():
    """Verify resolution falls back to GEMINI_API_KEY or GOOGLE_API_KEY."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "env_gemini_key", "GOOGLE_API_KEY": ""}):
        resolved = await resolve_gemini_api_key(passed_key=None)
        assert resolved == "env_gemini_key"

    with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": "env_google_key"}):
        resolved = await resolve_gemini_api_key(passed_key=None)
        assert resolved == "env_google_key"


@pytest.mark.asyncio
async def test_resolve_gemini_api_key_none_when_empty():
    """Verify returns None when no key is set anywhere."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""}, clear=True):
        from orchestration.settings import app_settings
        with patch.object(app_settings, "GEMINI_API_KEY", None):
            resolved = await resolve_gemini_api_key(passed_key="")
            assert resolved is None


# ---------------------------------------------------------------------------
# Gemini Live Session Configuration & Handshake Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gemini_live_session_init():
    session = GeminiLiveSession(
        api_key="test_api_key",
        model="models/gemini-2.0-flash-exp",
        voice_name="Aoede",
        system_instruction="You are Ananya at Snapserve.",
        sample_rate=16000,
        enable_tools=True,
    )
    assert session.model == "models/gemini-2.0-flash-exp"
    assert session.voice_name == "Aoede"
    assert session.system_instruction == "You are Ananya at Snapserve."
    assert session.is_connected is False


@pytest.mark.asyncio
async def test_gemini_live_tool_execution():
    """Verify Gemini Live tool calls dispatch to default_registry and reply with toolResponse."""
    session = GeminiLiveSession(
        api_key="test_key",
        voice_name="Aoede",
        enable_tools=True,
    )
    mock_ws = AsyncMock()
    session.ws = mock_ws
    session.is_connected = True

    tool_call_payload = {
        "functionCalls": [
            {
                "id": "call_12345",
                "name": "query_pricing",
                "args": {"plan_tier": "growth"},
            }
        ]
    }

    event_record = []
    async def mock_tool_event(name, data):
        event_record.append((name, data))

    session._on_tool_event = mock_tool_event

    await session._handle_gemini_tool_call(tool_call_payload)

    # Check mock_ws was sent toolResponse JSON
    assert mock_ws.send.called
    sent_json = json.loads(mock_ws.send.call_args[0][0])
    assert "toolResponse" in sent_json
    func_resps = sent_json["toolResponse"]["functionResponses"]
    assert len(func_resps) == 1
    assert func_resps[0]["id"] == "call_12345"
    assert "output" in func_resps[0]["response"]
    assert func_resps[0]["response"]["output"]["pricing"]["base_usd"] == 199

    assert len(event_record) == 1
    assert event_record[0][0] == "query_pricing"
    assert event_record[0][1]["status"] == "success"


# ---------------------------------------------------------------------------
# Audio Resampling Quality Tests (24 kHz Gemini Native -> 16 kHz Web Player)
# ---------------------------------------------------------------------------

def test_gemini_audio_resampling_clean():
    """Verify 24kHz PCM16 audio from Gemini Live downsamples to 16kHz with intact duration."""
    sr_src = 24000
    sr_dst = 16000
    dur_sec = 0.100  # 100 ms

    # Generate 100ms of 440 Hz sine wave at 24kHz
    n_samples_24k = int(sr_src * dur_sec)
    t = np.linspace(0, dur_sec, n_samples_24k, endpoint=False)
    pcm_24k_samples = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    pcm_24k_bytes = pcm_24k_samples.tobytes()

    pcm_16k_bytes = resample_oneshot(pcm_24k_bytes, sr_src, sr_dst)
    pcm_16k_samples = np.frombuffer(pcm_16k_bytes, dtype=np.int16)

    # 100 ms at 16kHz = 1600 samples
    assert len(pcm_16k_samples) == int(sr_dst * dur_sec)
    assert len(pcm_16k_bytes) == len(pcm_16k_samples) * 2
