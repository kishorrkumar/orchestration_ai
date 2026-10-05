"""
Phase 1 Automated Test Suite: Lean S2S Voice Agent Platform.
Tests:
- System prompt delimiter (<system> ... <system>)
- SentencePiece token counting and compiler token budgeting
- Local time zone resolution (including Asia/Kolkata half-hour offset)
- EndOfCallDetector fuzzy matching, quiet window, and cancel-on-speech
- REST API: Agent CRUD, draft editing, immutable publishing, versions, restore, duplicate
"""

import datetime
import pytest
import zoneinfo
import httpx

from orchestration.gateway.app import create_app
from orchestration.pipeline.end_detector import EndOfCallDetector
from orchestration.prompts.compiler import (
    IDEAL_SYSTEM_PROMPT_TOKENS,
    MAX_SYSTEM_PROMPT_TOKENS,
    compile_prompt,
    count_tokens,
    get_local_time_context,
)
from orchestration.protocol.prompt import (
    PERSONAPLEX_SYSTEM_DELIMITER,
    wrap_system_prompt,
)


def test_delimiter_constant_and_wrapping():
    """Verify delimiter matches upstream PersonaPlex format strictly."""
    assert PERSONAPLEX_SYSTEM_DELIMITER == "<system>"
    wrapped = wrap_system_prompt("Hello world")
    assert wrapped == "<system> Hello world <system>"

    # Idempotent wrapping
    assert wrap_system_prompt("<system> Hello world <system>") == "<system> Hello world <system>"


def test_local_time_context():
    """Verify time context computation across standard and half-hour timezones."""
    # Test fixed timestamp in Asia/Kolkata (UTC +05:30)
    # 2026-10-05 15:30 UTC = 2026-10-05 21:00 IST (night)
    fixed_dt = datetime.datetime(2026, 10, 5, 15, 30, tzinfo=datetime.timezone.utc)
    ctx = get_local_time_context("Asia/Kolkata", dt=fixed_dt)
    assert ctx["weekday"] == "Monday"
    assert "9:00 PM" in ctx["current_time"]
    assert ctx["day_part"] == "night"
    assert "Monday" in ctx["time_line"]

    # 2026-10-05 04:30 UTC = 2026-10-05 10:00 IST (morning)
    fixed_morning = datetime.datetime(2026, 10, 5, 4, 30, tzinfo=datetime.timezone.utc)
    ctx_m = get_local_time_context("Asia/Kolkata", dt=fixed_morning)
    assert ctx_m["day_part"] == "morning"
    assert "10:00 AM" in ctx_m["current_time"]


def test_sentencepiece_token_counting_and_compiler():
    """Verify real SentencePiece token counting and compiler envelope."""
    system_prompt = "You are Alex, a helpful support agent. Speak in short sentences."
    greeting = "Hello! How can I help you today?"
    ending = "Thanks for calling. Goodbye!"

    compiled = compile_prompt(
        system_prompt=system_prompt,
        greeting_text=greeting,
        greeting_mode="agent_first",
        ending_text=ending,
        agent_name="Alex",
        timezone="Asia/Kolkata",
    )

    assert compiled.text.startswith("<system>")
    assert compiled.text.endswith("<system>")
    assert "Start: Open the call by saying:" in compiled.text
    assert "Close: When the conversation is done, say:" in compiled.text
    assert compiled.token_count > 0
    assert compiled.token_count <= MAX_SYSTEM_PROMPT_TOKENS
    assert compiled.can_publish is True


def test_three_seed_agents_token_budgets():
    """Verify all 3 starter seed agents from Section 11 fit under 350 tokens."""
    # Starter A
    seed_a = compile_prompt(
        system_prompt=(
            "You are {{agent_name}} from {{company}}, on a live phone call with {{customer_name}}. Your goal: {{goal}}.\n\n"
            "Talk like a warm, relaxed real person, not a script. Keep each turn to one or two short sentences and ask one question at a time. "
            "Use natural little reactions like 'mm-hm', 'right', 'oh, got it', 'sure thing'. "
            "Let them finish before you speak, and stop talking the moment they cut in. "
            "If you didn't catch something, say so and ask them to repeat it. Don't make things up; if you're not sure, say someone will follow up.\n\n"
            "Call flow: greet them and say who you are, then after they reply, say why you're calling in one line and listen. "
            "Help them step by step, confirming anything important like names, dates and numbers by repeating it back. "
            "Before you wrap up, check if there's anything else they need.\n\n"
            "If anyone asks, tell them honestly that you're an AI assistant."
        ),
        greeting_text="Hi {{customer_name}}, good {{day_part}}! This is {{agent_name}} from {{company}}. How are you doing today?",
        ending_text="Alright, thanks so much for your time, {{customer_name}}. Take care, bye!",
        agent_name="Alex",
        timezone="Asia/Kolkata",
        variables={"customer_name": "Ravi", "company": "Apex", "goal": "check in on onboarding"},
    )
    assert seed_a.token_count <= 350, f"Seed A exceeded budget: {seed_a.token_count}"

    # Starter B
    seed_b = compile_prompt(
        system_prompt=(
            "You are {{agent_name}} from {{company}}, on a live phone call with the patient. "
            "Your goal: help the caller book, move or cancel an appointment: collect name, preferred day and time, and a callback number "
            "(read digits back in groups); say the team will confirm by text. "
            "Speak in clear, calm, short sentences. Confirm each detail before moving to the next. "
            "If symptoms sound urgent or emergency, instruct the caller to seek emergency medical care immediately."
        ),
        greeting_text="Good {{day_part}}, thanks for calling {{company}}! This is {{agent_name}}. How can I help you today?",
        ending_text="You're all set. Thanks for calling, take care, bye!",
        agent_name="Sarah",
        timezone="Asia/Kolkata",
        variables={"company": "City Clinic"},
    )
    assert seed_b.token_count <= 350, f"Seed B exceeded budget: {seed_b.token_count}"

    # Starter C
    seed_c = compile_prompt(
        system_prompt=(
            "You are Alex from {{company}} support. "
            "Your goal: find out the problem in at most three questions, give one step at a time, check it worked, offer a follow-up if it didn't. "
            "Empathize first ('oh no, that's frustrating'). "
            "Keep your explanations brief and natural, and check understanding after each instruction."
        ),
        greeting_text="Hi, this is Alex from {{company}} support. What's going on today?",
        ending_text="Glad we got that sorted. Have a good one, bye!",
        agent_name="Alex",
        timezone="Asia/Kolkata",
        variables={"company": "CloudDesk"},
    )
    assert seed_c.token_count <= 350, f"Seed C exceeded budget: {seed_c.token_count}"


def test_end_of_call_detector_exact_and_fuzzy_match():
    """Verify EndOfCallDetector matches closing phrase and initiates 1.5s quiet window."""
    detector = EndOfCallDetector(
        ending_text="Alright, thanks so much for your time. Take care, bye!",
        quiet_window_sec=1.5,
    )
    t0 = 100.0
    detector.start_session(t0)

    # Ingest casual non-closing tokens
    matched = detector.on_agent_token("I can certainly help you with your question.", t0 + 1.0)
    assert not matched
    assert not detector.pending_hangup

    # Ingest closing tokens (fuzzy match)
    matched = detector.on_agent_token(" That is all for today. Take care, bye!", t0 + 3.0)
    assert matched
    assert detector.matched_in_current_turn

    # Agent audio finishes at t0 + 4.0 -> schedules hangup at t0 + 5.5 (4.0 + 1.5)
    detector.on_agent_speaking_stopped(t0 + 4.0)
    assert detector.pending_hangup
    assert detector.hangup_target_time == t0 + 5.5

    # Check at t0 + 5.0 (quiet window not yet elapsed)
    assert detector.check_termination(t0 + 5.0) is None

    # Check at t0 + 5.6 (quiet window elapsed without caller speech)
    assert detector.check_termination(t0 + 5.6) == "agent_closed"


def test_end_of_call_detector_cancel_on_speech():
    """Verify that caller speaking during the 1.5s quiet window cancels pending hangup."""
    detector = EndOfCallDetector(
        ending_text="Have a good one, bye!",
        quiet_window_sec=1.5,
    )
    t0 = 100.0
    detector.start_session(t0)

    # Agent speaks closer
    detector.on_agent_token("Have a good one, bye!", t0 + 2.0)
    detector.on_agent_speaking_stopped(t0 + 3.0)
    assert detector.pending_hangup

    # Caller cuts in at t0 + 3.8 (inside 1.5s window): "Wait, one more question!"
    canceled = detector.on_user_speech(t0 + 3.8)
    assert canceled is True
    assert detector.pending_hangup is False

    # Check after window would have passed: call should NOT terminate!
    assert detector.check_termination(t0 + 5.0) is None


def test_end_of_call_detector_timeouts():
    """Verify silence timeout and max call duration."""
    detector = EndOfCallDetector(
        silence_timeout_sec=10.0,
        max_duration_sec=30.0,
    )
    t0 = 0.0
    detector.start_session(t0)

    # At 5s, no timeout
    assert detector.check_termination(5.0) is None

    # At 11s with no speech, silence timeout
    assert detector.check_termination(11.0) == "silence_timeout"

    # Reset with speech
    detector.start_session(t0)
    detector.on_user_speech(25.0)  # speech resets silence
    assert detector.check_termination(26.0) is None

    # At 31s, max duration triggers
    assert detector.check_termination(31.0) == "max_duration"


@pytest.mark.asyncio
async def test_agent_api_full_lifecycle():
    """Verify Agent CRUD, draft updates, publishing, versioning, restore, and duplicate."""
    app = create_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # 1. Create Agent
        payload = {
            "name": "Integration Test Bot",
            "voice_id": "NATF1.pt",
            "greeting_text": "Welcome! How can I help?",
            "greeting_mode": "agent_first",
            "system_prompt": "You are a friendly bot.",
            "ending_text": "Goodbye!",
            "timezone": "Asia/Kolkata",
            "auto_publish": True,
        }
        res = await client.post("/v1/agents", json=payload)
        assert res.status_code == 201
        agent_data = res.json()
        agent_id = agent_data["id"]
        assert agent_data["name"] == "Integration Test Bot"
        assert agent_data["status"] == "created"
        assert agent_data["agent"]["status"] == "published"
        assert agent_data["current_version_no"] == 1

        # 2. Get Agent
        res = await client.get(f"/v1/agents/{agent_id}")
        assert res.status_code == 200
        assert res.json()["id"] == agent_id

        # 3. Patch Draft
        patch_payload = {
            "name": "Renamed Bot",
            "draft_greeting_text": "Updated greeting text.",
        }
        res = await client.patch(f"/v1/agents/{agent_id}", json=patch_payload)
        assert res.status_code == 200
        assert res.json()["name"] == "Renamed Bot"
        assert res.json()["status"] == "updated"
        assert res.json()["agent_status"] == "draft"

        # 4. Publish Version 2
        res = await client.post(f"/v1/agents/{agent_id}/publish", json={"change_note": "v2 release"})
        assert res.status_code == 200
        ver_data = res.json()
        assert ver_data["version_no"] == 2
        assert ver_data["change_note"] == "v2 release"

        # 5. List Versions
        res = await client.get(f"/v1/agents/{agent_id}/versions")
        assert res.status_code == 200
        versions = res.json()
        assert len(versions) == 2

        # 6. Restore Version 1
        res = await client.post(f"/v1/agents/{agent_id}/restore/1")
        assert res.status_code == 200
        restored = res.json()
        assert restored["draft_greeting_text"] == "Welcome! How can I help?"

        # 7. Duplicate
        res = await client.post(f"/v1/agents/{agent_id}/duplicate")
        assert res.status_code == 200
        dup = res.json()
        assert dup["id"] != agent_id
        assert "Copy" in dup["name"]

        # 8. Compile Endpoint
        compile_req = {
            "system_prompt": "You are a test agent.",
            "greeting_text": "Hello!",
            "ending_text": "Bye!",
            "agent_name": "TestBot",
            "timezone": "Asia/Kolkata",
        }
        res = await client.post("/v1/prompts/compile", json=compile_req)
        assert res.status_code == 200
        compiled = res.json()
        assert compiled["token_count"] > 0
        assert compiled["can_publish"] is True
        assert "<system>" in compiled["compiled_text"]

        # 9. Clean up
        res = await client.delete(f"/v1/agents/{agent_id}")
        assert res.status_code == 200
        res = await client.delete(f"/v1/agents/{dup['id']}")
        assert res.status_code == 200
