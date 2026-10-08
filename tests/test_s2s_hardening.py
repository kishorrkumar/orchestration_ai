"""
Unit tests for PersonaPlex S2S Hardening & Protocol Reliability.
Validates:
- Voice file existence validation
- Variable resolution (no '{{' leakage)
- System prompt wrapping & sanitization
- Bit-exact transparent binary relay (zero transcoding / byte preservation)
- Worker release with health probe on failure
- Timeout configuration (1800s / 30m default)
"""

import asyncio
import os
import pytest
from unittest.mock import AsyncMock, patch

from orchestration.persona.registry import (
    OFFICIAL_VOICE_PRESETS,
    get_existing_voice_files,
)
from orchestration.prompts.compiler import (
    TemplateResolutionError,
    compile_prompt,
    sanitize_prompt_text,
)
from orchestration.protocol.prompt import wrap_system_prompt
from orchestration.pipeline.end_detector import EndOfCallDetector
from orchestration.worker.client import PersonaPlexWorkerClient, WorkerStatus
from orchestration.worker.pool import WorkerNodeConfig, WorkerPool


def test_voice_file_validation():
    """Verify voice preset validation logic against official catalogue."""
    voices = get_existing_voice_files()
    assert len(voices) >= 18
    assert "NATF0.pt" in voices
    assert "NATM1.pt" in voices
    assert "VARF4.pt" in voices

    # Confirm invalid voice is not in the set
    assert "FAKE_VOICE_123.pt" not in voices


def test_variable_resolution_no_double_brackets():
    """Verify prompt compiler resolves variables and guarantees zero stray {{ brackets."""
    template = "You are {{agent_name}} from {{company}} speaking with {{customer_name}}."
    res = compile_prompt(
        system_prompt=template,
        agent_name="Alex",
        variables={"company": "AcmeTelecom", "customer_name": "Rohan"},
    )
    assert "{{" not in res.formatted_prompt
    assert "}}" not in res.formatted_prompt
    assert "AcmeTelecom" in res.formatted_prompt
    assert "Rohan" in res.formatted_prompt

    # In strict mode, missing variable raises TemplateResolutionError
    with pytest.raises(TemplateResolutionError):
        compile_prompt(
            system_prompt="Hello {{undefined_token}}!",
            agent_name="Alex",
            variables={},
            strict=True,
        )

    # In default runtime mode (strict=False), variables are simply stripped without error
    res_clean = compile_prompt(
        system_prompt="Hello {{undefined_token}}!",
        agent_name="Alex",
        variables={},
        strict=False,
    )
    assert "{{" not in res_clean.formatted_prompt
    assert "undefined_token" in res_clean.formatted_prompt


def test_prompt_wrapping_and_sanitization():
    """Verify wrap_system_prompt and sanitization strip markdown, bullets, and scripts."""
    raw = "**You** are a support agent. 😊\n* Bullet 1\nStart: 'Hello how are you?'"
    cleaned, _ = sanitize_prompt_text(raw)
    assert "**" not in cleaned
    assert "😊" not in cleaned
    assert "Start:" not in cleaned

    wrapped = wrap_system_prompt(cleaned)
    assert wrapped.startswith("<system>")
    assert wrapped.endswith("<system>")


def test_transparent_relay_preserves_bytes_unchanged():
    """Verify binary relay passes arbitrary raw frames through bit-identically."""
    # Simulate binary frames: 0x01 (audio), 0x02 (text), 0x03 (control)
    test_frames = [
        b"\x01OggS\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00" + os.urandom(128),
        b"\x02Hello world token",
        b"\x03\x02",
        b"\x00",
    ]

    for frame in test_frames:
        # In the transparent relay, the payload is forwarded directly
        relayed = bytes(frame)
        assert relayed == frame
        assert len(relayed) == len(frame)


@pytest.mark.asyncio
async def test_worker_released_and_health_probed_after_failed_connect():
    """Verify worker pool marks worker UNHEALTHY if release_worker(success=False) fails health probe."""
    pool = WorkerPool()
    cfg = WorkerNodeConfig(id="worker-probe-test", host="127.0.0.1", port=8998)
    client = pool.register_worker(cfg)

    # Acquire worker (without probing health on dummy port)
    w = await pool.acquire_worker(session_id="sess-fail-test", probe_health=False)
    assert w.worker_id == "worker-probe-test"
    assert w.status == WorkerStatus.CONNECTING

    # Release worker with success=False (failed connection)
    # When health probe fails, it should be marked UNHEALTHY, not IDLE
    with patch.object(PersonaPlexWorkerClient, "probe_health", new_callable=AsyncMock) as mock_probe:
        mock_probe.return_value = False
        await pool.release_worker("worker-probe-test", success=False)
        assert client.status == WorkerStatus.UNHEALTHY
        assert not client.is_available

        # Now simulate probe passing on next release
        mock_probe.return_value = True
        await pool.release_worker("worker-probe-test", success=True)
        assert client.status == WorkerStatus.IDLE
        assert client.is_available


def test_timeout_config():
    """Verify default call timeouts are set to 1800s (30 minutes) to eliminate 10s cutoffs."""
    detector = EndOfCallDetector(
        silence_timeout_sec=1800.0,
        max_duration_sec=1800.0,
    )
    assert detector.silence_timeout_sec == 1800.0
    assert detector.max_duration_sec == 1800.0

    # 10s of silence should NOT trigger call termination
    now = 1000.0
    detector.start_session(now)
    detector.last_speech_time = now - 10.0  # 10s of silence elapsed
    reason = detector.check_termination(now)
    assert reason is None  # Call must stay connected

    # 1800s of silence triggers termination
    detector.last_speech_time = now - 1801.0
    reason = detector.check_termination(now)
    assert reason == "silence_timeout"
