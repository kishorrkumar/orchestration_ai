"""
Unit and Integration test suite for Phase 3: Domain, Application, and REST Interfaces.
Validates:
- Pure domain entities and rules (Agent, Version, TimeContext, PromptLinter, EndOfCallDetector)
- Strict adherence to <system> {prompt} <system> delimiter
- Half-hour timezones (Asia/Kolkata, Asia/Kathmandu)
- Application services (Create, Update, Publish, Revert, Token Enforce)
- RFC 9457 Problem Details error handling
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from httpx import ASGITransport, AsyncClient

from orchestration.application.agents.commands import (
    CreateAgentCommand,
    PublishVersionCommand,
    UpdateAgentCommand,
)
from orchestration.application.agents.service import AgentApplicationService
from orchestration.application.prompts.service import PromptCompilerUseCase
from orchestration.domain.agent import AgentStatus
from orchestration.domain.detector import DetectionAction, EndOfCallDetector
from orchestration.domain.prompt import (
    SYSTEM_TAG_CLOSE,
    SYSTEM_TAG_OPEN,
    PromptLinter,
    compute_local_time_context,
    wrap_system_prompt,
)
from orchestration.gateway.app import create_app
from orchestration.infrastructure.clock.system_clock import FrozenClock
from orchestration.infrastructure.db.repositories.agent_repository import SqlAlchemyAgentRepository
from orchestration.infrastructure.tokenizer.sentencepiece_adapter import SentencePieceTokenizerAdapter
from orchestration.shared.errors import PromptTooLongError


@pytest.fixture(autouse=True)
async def setup_database():
    from orchestration.db.session import init_db
    await init_db()


# ==============================================================================
# 1. Pure Domain Layer Tests
# ==============================================================================

def test_system_tag_delimiter_exact():
    """Verify delimiter is literally <system> on both ends."""
    assert SYSTEM_TAG_OPEN == "<system>"
    assert SYSTEM_TAG_CLOSE == "<system>"

    wrapped = wrap_system_prompt("Hello world")
    assert wrapped == "<system> Hello world <system>"

    # Idempotent (no double-wrapping)
    assert wrap_system_prompt(wrapped) == wrapped


def test_time_context_half_hour_timezones():
    """Verify half-hour timezones (Asia/Kolkata +5:30, Asia/Kathmandu +5:45)."""
    ref_utc = datetime(2026, 10, 5, 12, 0, 0, tzinfo=UTC)

    # Asia/Kolkata is UTC+5:30 -> 17:30 (5:30 PM, evening)
    line_kolkata, part_kolkata = compute_local_time_context("Asia/Kolkata", ref_utc)
    assert "5:30 PM" in line_kolkata
    assert part_kolkata == "evening"

    # Asia/Kathmandu is UTC+5:45 -> 17:45 (5:45 PM, evening)
    line_ktm, part_ktm = compute_local_time_context("Asia/Kathmandu", ref_utc)
    assert "5:45 PM" in line_ktm
    assert part_ktm == "evening"


def test_prompt_linter_voice_hints():
    """Verify voice linter flags written conventions that sound unnatural when spoken."""
    bad_text = (
        "# Welcome!\n"
        "Please visit https://acme.org for more information.\n"
        "- Step 1: Call NASA\n"
        "Check (or uncheck) the box."
    )
    warnings = PromptLinter.lint(bad_text)
    rules = [w.rule for w in warnings]

    assert "no_urls" in rules
    assert "no_markdown" in rules
    assert "no_bullet_points" in rules
    assert "no_parentheticals" in rules
    assert "acronym_clarity" in rules


def test_end_of_call_detector_lifecycle():
    """Verify EndOfCallDetector fuzzy matching, quiet window, speech cancellation, and timeouts."""
    detector = EndOfCallDetector(
        ending_phrase="Thanks for your time, goodbye!",
        quiet_window_sec=1.5,
        silence_timeout_sec=10.0,
        max_duration_sec=60.0,
    )

    # 1. Ingest matching goodbye text
    detector.ingest_agent_token("Thanks", timestamp=10.0)
    detector.ingest_agent_token(" for your time, goodbye!", timestamp=10.5)

    # Within quiet window (<1.5s): Should CONTINUE to allow audio drain
    res1 = detector.evaluate_tick(current_time=11.2, call_elapsed_sec=11.2)
    assert res1.action == DetectionAction.CONTINUE

    # 2. Caller speaks within quiet window: Hangup is cancelled
    detector.record_caller_speech(timestamp=11.4)
    res2 = detector.evaluate_tick(current_time=12.5, call_elapsed_sec=12.5)
    assert res2.action == DetectionAction.CONTINUE

    # 3. Agent repeats goodbye
    detector.ingest_agent_token("Alright then, take care, goodbye!", timestamp=15.0)
    # Quiet window elapsed without interruption -> HANG_UP with agent_closed
    res3 = detector.evaluate_tick(current_time=16.6, call_elapsed_sec=16.6)
    assert res3.action == DetectionAction.HANG_UP
    assert res3.reason == "agent_closed"


# ==============================================================================
# 2. Application Service Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_agent_application_service_full_lifecycle():
    """Verify agent create -> versioning -> publish enforcement -> revert."""
    from orchestration.db.session import async_session_factory

    clock = FrozenClock(datetime(2026, 10, 5, 10, 0, 0, tzinfo=UTC))
    tokenizer = SentencePieceTokenizerAdapter()
    compiler = PromptCompilerUseCase(tokenizer=tokenizer, clock=clock)

    async with async_session_factory() as session:
        repo = SqlAlchemyAgentRepository(session)
        service = AgentApplicationService(repository=repo, compiler=compiler, clock=clock)

        # 1. Create Agent
        cmd = CreateAgentCommand(
            name="Clinic Helper",
            voice_id="natural_calm",
            greeting="Hello {{customer_name}}, this is {{agent_name}}.",
            system_prompt="You are a clinic assistant helping patients with appointments.",
            ending="Thanks for calling, have a wonderful day!",
            timezone_str="America/New_York",
        )
        agent = await service.create_agent(cmd)
        assert agent.status == AgentStatus.DRAFT
        assert agent.current_version == 1

        # Check v1 version was snapshotted
        versions = await service.list_versions(agent.id)
        assert "clinic assistant" in versions[0].compiled_prompt.lower()

        # 2. Update Agent (creates version 2)
        clock.advance(3600)
        update_cmd = UpdateAgentCommand(
            system_prompt="You are an expert clinic assistant. Speak clearly and concisely.",
            change_note="Refined system prompt instructions",
        )
        updated = await service.update_agent(agent.id, update_cmd)
        assert updated.current_version == 2

        versions_after_update = await service.list_versions(agent.id)
        assert len(versions_after_update) == 2
        assert versions_after_update[0].version_number == 2
        assert "expert clinic assistant" in versions_after_update[0].compiled_prompt

        # 3. Publish Agent
        pub_cmd = PublishVersionCommand(change_note="Ready for production")
        published = await service.publish_version(agent.id, pub_cmd)
        assert published.status == AgentStatus.PUBLISHED
        assert published.published_version == 2

        # 4. Enforce Hard Token Limit (>350 tokens blocks publish)
        too_long_prompt = "word " * 450
        await service.update_agent(agent.id, UpdateAgentCommand(system_prompt=too_long_prompt))
        with pytest.raises(PromptTooLongError) as exc_info:
            await service.publish_version(agent.id, PublishVersionCommand())
        assert "exceeding hard limit of 350" in str(exc_info.value)

        # 5. Revert back to Version 1
        reverted = await service.revert_version(agent.id, 1)
        assert "You are a clinic assistant" in reverted.system_prompt


# ==============================================================================
# 3. HTTP Interface & RFC 9457 Problem Details Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_rfc_9457_problem_details_and_v2_routes():
    """Verify REST endpoints and RFC 9457 Problem Details responses."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Health & readiness
        r_ready = await client.get("/readyz")
        assert r_ready.status_code == 200
        assert r_ready.json()["status"] == "ready"

        # 2. List voices (18 presets)
        r_voices = await client.get("/v2/agents/voices")
        assert r_voices.status_code == 200
        voices = r_voices.json()
        assert len(voices) == 18

        # 3. Live Compile Prompt
        r_compile = await client.post(
            "/v2/prompts/compile",
            json={
                "system_prompt": "You are a friendly agent on a phone call. Visit https://example.com",
                "greeting": "Hello!",
                "ending": "Goodbye!",
                "timezone_str": "Asia/Kolkata",
            },
        )
        assert r_compile.status_code == 200
        data = r_compile.json()
        assert data["token_count"] > 0
        assert any(w["rule"] == "no_urls" for w in data["warnings"])

        # 4. Create Agent via REST
        r_create = await client.post(
            "/v2/agents",
            json={
                "name": "Integration Test Agent",
                "voice_id": "warm_conversational",
                "greeting": "Hey there!",
                "system_prompt": "Speak like a warm friend.",
                "ending": "See you soon, bye!",
                "timezone_str": "UTC",
            },
        )
        assert r_create.status_code == 201
        agent_data = r_create.json()
        agent_id = agent_data["id"]
        assert agent_id.startswith("agt_")
        assert agent_data["current_version"] == 1

        # 4b. Verify PATCH draft with silence_timeout_sec=1800 (Aarav default) succeeds
        r_patch = await client.patch(
            f"/v2/agents/{agent_id}",
            json={
                "system_prompt": "You enjoy having a good conversation. You are Aarav.",
                "silence_timeout_sec": 1800.0,
                "voice_id": "NATM1.pt",
            },
        )
        assert r_patch.status_code == 200, r_patch.text
        assert r_patch.json()["system_prompt"] == "You enjoy having a good conversation. You are Aarav."

        # 5. RFC 9457: Not Found Error (404)
        r_404 = await client.get("/v2/agents/agt_nonexistent12345")
        assert r_404.status_code == 404
        assert r_404.headers["content-type"] == "application/problem+json"
        prob_404 = r_404.json()
        assert prob_404["code"] == "NOT_FOUND"
        assert prob_404["status"] == 404
        assert "error_id" in prob_404
        assert prob_404["error_id"].startswith("err_")

        # 6. RFC 9457: Request Validation Error (422)
        r_422 = await client.post(
            "/v2/agents",
            json={
                "name": "",  # min_length 1 violation
                "voice_id": "natural_calm",
            },
        )
        assert r_422.status_code == 422
        assert r_422.headers["content-type"] == "application/problem+json"
        prob_422 = r_422.json()
        assert prob_422["code"] == "VALIDATION_ERROR"
        assert len(prob_422["invalid_params"]) > 0
