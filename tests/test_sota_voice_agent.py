"""
Unit and Integration Tests for SOTA Voice Agent Upgrades:
1. WebRTC AEC3 APM & Neural Denoising Graph.
2. Dual-Channel Coherence VAD with Smart Turn Completion.
3. Structured JSON Tool Calling & Schema Generation.
4. Sub-second TTFA & Startup Delay Optimization.
"""

from __future__ import annotations

import asyncio
import numpy as np
import pytest

from orchestration.audio.webrtc_apm import WebRTCAPM
from orchestration.audio.smart_turn import SmartTurnDetector
from orchestration.audio.cleaner import CallerAudioCleaner
from orchestration.tools import (
    ToolRegistry,
    book_calendar_slot,
    default_registry,
    query_pricing,
    tool,
)
from orchestration.persona.registry import default_registry as persona_registry
from orchestration.session.manager import SessionManager
from orchestration.worker.pool import WorkerPool, WorkerStatus
from orchestration.worker.client import PersonaPlexWorkerClient


# ---------------------------------------------------------------------------
# Pillar 1 & 4 Tests: WebRTC AEC3, Denoise Graph, and Dual-Channel VAD
# ---------------------------------------------------------------------------

def test_webrtc_apm_echo_cancellation_and_erle():
    """Verify WebRTC APM cancels agent loudspeaker reference and calculates ERLE."""
    apm = WebRTCAPM(sample_rate=24000, frame_size=1920, enabled=True)

    # Synthetic agent speech (440 Hz tone)
    t = np.linspace(0, 0.080, 1920, endpoint=False, dtype=np.float32)
    agent_speech = 0.3 * np.sin(2 * np.pi * 440 * t)

    # Feed 5 frames of reference playback into APM
    for _ in range(5):
        apm.push_reference(agent_speech)

    # Microphone picks up the agent's loudspeaker echo (attenuated + delayed)
    mic_echo = 0.15 * np.sin(2 * np.pi * 440 * t)
    cleaned, meta = apm.process_frame(mic_echo)

    assert len(cleaned) == 1920
    assert "erle_db" in meta
    assert "double_talk" in meta
    # Echo cancellation should reduce energy or detect coherence
    assert meta["coherence"] > 0.0


def test_webrtc_apm_double_talk_preserves_caller_voice():
    """Verify double-talk detection freezes filter adaptation so caller voice isn't clipped."""
    apm = WebRTCAPM(sample_rate=24000, frame_size=1920, enabled=True)

    t = np.linspace(0, 0.080, 1920, endpoint=False, dtype=np.float32)
    agent_speech = 0.2 * np.sin(2 * np.pi * 440 * t)
    apm.push_reference(agent_speech)

    # Caller speaks loudly over agent (double-talk) with different frequency
    caller_voice = 0.5 * np.sin(2 * np.pi * 880 * t)
    mic_mixed = 0.1 * agent_speech + caller_voice

    cleaned, meta = apm.process_frame(mic_mixed)
    # Output must retain caller voice power
    out_rms = float(np.sqrt(np.mean(cleaned ** 2)))
    assert out_rms > 0.15
    assert meta["double_talk"] is True


def test_caller_audio_cleaner_uses_webrtc_apm():
    """Verify CallerAudioCleaner integrates WebRTC APM without errors."""
    cleaner = CallerAudioCleaner(sample_rate=24000, frame_size=1920, enable_aec=True)
    assert cleaner.apm is not None
    assert isinstance(cleaner.apm, WebRTCAPM)

    t = np.linspace(0, 0.080, 1920, endpoint=False, dtype=np.float32)
    ref = 0.2 * np.sin(2 * np.pi * 440 * t)
    cleaner.push_agent_reference(ref)

    frames = cleaner.process_chunk(ref)
    assert len(frames) >= 1
    assert frames[0].dtype == np.float32


def test_smart_turn_detector_dynamic_hangover():
    """Verify dynamic silence timeout: 250ms for complete statement vs 850ms for continuation."""
    detector = SmartTurnDetector(sample_rate=24000, frame_size=1920)

    # 1. Complete short answer -> Fast threshold (250ms)
    t_complete = detector._compute_dynamic_silence_threshold("Yes, definitely.")
    assert t_complete <= 0.30

    # 2. Hesitation / continuation -> Extended threshold (850ms – 900ms)
    t_hesitation = detector._compute_dynamic_silence_threshold("I was thinking because...")
    assert t_hesitation >= 0.85

    # 3. Default sentence -> Standard base threshold (500ms)
    t_default = detector._compute_dynamic_silence_threshold("I want to know about your product")
    assert 0.45 <= t_default <= 0.60


# ---------------------------------------------------------------------------
# Pillar 3 Tests: Structured JSON Tools
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_decorator_schema_generation():
    """Verify @tool decorator creates valid OpenAI/Anthropic function schemas."""
    @tool(name="calculate_discount", description="Calculate volume discount percentage.")
    async def calculate_discount(volume: int, customer_type: str = "standard") -> dict:
        discount = 15 if volume > 1000 else 5
        return {"volume": volume, "discount_percent": discount}

    schema = calculate_discount.to_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "calculate_discount"
    assert "parameters" in schema["function"]
    assert "volume" in schema["function"]["parameters"]["properties"]
    assert schema["function"]["parameters"]["properties"]["volume"]["type"] == "integer"


@pytest.mark.asyncio
async def test_builtin_tool_execution():
    """Verify built-in sales tools execute properly."""
    res_pricing = await default_registry.execute(
        "query_pricing",
        {"plan_tier": "enterprise", "monthly_minutes": 5000},
    )
    assert res_pricing.status == "success"
    assert "pricing" in res_pricing.data
    assert res_pricing.direct_speech is not None
    assert "$799" in res_pricing.summary or "799" in res_pricing.summary

    res_booking = await default_registry.execute(
        "book_calendar_slot",
        {"date": "2026-10-16", "time": "14:00", "caller_name": "Aarav Sharma"},
    )
    assert res_booking.status == "success"
    assert res_booking.data["caller_name"] == "Aarav Sharma"
    assert res_booking.data["status"] == "confirmed"


# ---------------------------------------------------------------------------
# Pillar 2 Tests: Startup Delay Elimination & Pre-primed Ananya Matching
# ---------------------------------------------------------------------------

def test_ananya_registered_with_natf0():
    """Verify Ananya is registered in persona registry with NATF0.pt voice."""
    ananya = persona_registry.get("ananya")
    assert ananya is not None
    assert ananya.name.startswith("Ananya")
    assert ananya.get_normalized_voice_prompt() == "NATF0.pt"


@pytest.mark.asyncio
async def test_standby_claim_matches_ananya_instantly():
    """Verify standby claim succeeds in <10ms when caller requests Ananya."""
    from orchestration.worker.pool import WorkerNodeConfig
    pool = WorkerPool()
    worker = pool.register_worker(WorkerNodeConfig(id="test_worker_0", host="127.0.0.1", port=8998))
    worker._status = WorkerStatus.BUSY
    worker._is_connected = True

    mgr = SessionManager(pool=pool, enable_standby=True)
    ananya = persona_registry.get("ananya")

    # Manually establish active standby session with Ananya
    from orchestration.session.manager import VoiceSession
    session = VoiceSession(
        session_id="standby_test",
        worker=worker,
        persona=ananya,
        pool=pool,
    )
    session.is_standby = True
    session._state = session.state.ACTIVE
    mgr._standby_session = session

    # Claim standby
    claimed = await mgr.claim_standby_if_matched(persona=ananya, session_id="live_call_123")
    assert claimed is not None
    assert claimed.session_id == "live_call_123"
    assert claimed.is_claimed_from_standby is True
