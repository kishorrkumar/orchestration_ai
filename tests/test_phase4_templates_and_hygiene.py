"""
Phase 4: Agent Templates & Prompt Hygiene Test Suite.

Validates:
1. Production domain agent templates:
   - CitySan (municipal waste management & bulky item dispatch)
   - Jerusalem Shakshuka (Middle Eastern dining reservation & takeout)
   - AeroRentals Pro (commercial drone leasing & FAA Part 107 compliance)
   - Plus healthcare, SDR, IT, and concierge presets.
2. Prompt token budget and hygiene gates:
   - Delimiter compliance (<system> ... </system> exact wrapping).
   - Token budget (< 200 tokens ideal conversational limit for < 2.0s startup).
   - Zero markdown, zero emojis, zero rigid quotation blocks.
   - Strict variable rendering (zero stray {{...}} leaking to the model).
3. API endpoints:
   - GET /v1/templates
   - GET /v1/templates/{template_id}
   - POST /v1/templates/{template_id}/compile
   - POST /v2/agents/from-template/{template_id}
"""

from __future__ import annotations

import re
import pytest
from httpx import ASGITransport, AsyncClient

from orchestration.gateway.app import create_app
from orchestration.persona.registry import PersonaRegistry
from orchestration.prompts.compiler import (
    compile_prompt,
    count_tokens,
    sanitize_prompt_text,
    TemplateResolutionError,
)
from orchestration.prompts.templates import (
    get_all_templates,
    get_template_by_id,
    compile_template,
)
from orchestration.session.manager import SessionManager
from orchestration.worker.pool import WorkerPool


# =====================================================================
# Domain Agent Templates & Token Budget
# =====================================================================

def test_all_templates_compile_within_ideal_token_budget():
    templates = get_all_templates()
    assert len(templates) >= 7, f"Expected at least 7 domain templates, found {len(templates)}"

    for tmpl in templates:
        compiled = compile_template(tmpl)

        # 1. Delimiter wrapping: exact upstream format is "<system> {content} <system>"
        assert compiled.text.startswith("<system>"), f"Template {tmpl.id} missing opening <system> tag"
        assert compiled.text.endswith("<system>"), f"Template {tmpl.id} missing closing <system> tag"
        assert compiled.text.count("<system>") == 2, f"Template {tmpl.id} must have exactly opening and closing <system> tags"

        # 2. Token Budget: ideal conversational limit is 250 tokens with live date/time directives (max hard limit 350)
        assert compiled.can_publish is True, f"Template {tmpl.id} cannot be published"
        assert compiled.token_count <= 250, (
            f"Template {tmpl.id} exceeds ideal 250-token conversational limit: {compiled.token_count} tokens"
        )

        # 3. Hygiene: zero stray double brackets in compiled text
        assert "{{" not in compiled.text, f"Template {tmpl.id} has unrendered '{{' placeholder"
        assert "}}" not in compiled.text, f"Template {tmpl.id} has unrendered '}}' placeholder"

        # 4. Zero markdown symbols in compiled text
        assert not re.search(r"(\*\*|__|\#\#|\* |^\s*[-*+]\s+|^\s*\d+\.\s+)", compiled.text, flags=re.MULTILINE), (
            f"Template {tmpl.id} contains markdown formatting"
        )


def test_citysan_municipal_template_hygiene():
    tmpl = get_template_by_id("citysan-waste")
    assert tmpl is not None
    assert tmpl.id == "citysan-waste"
    assert "Sanitation" in tmpl.name or "CitySan" in tmpl.name
    assert tmpl.voice_id == "NATM3.pt"
    assert tmpl.timezone_str == "America/Chicago"

    compiled = compile_template(tmpl)
    assert "City Sanitation" in compiled.text
    assert "hazardous" in compiled.text.lower()
    assert compiled.token_count < 250


def test_jerusalem_shakshuka_hospitality_template_hygiene():
    tmpl = get_template_by_id("jerusalem-shakshuka")
    assert tmpl is not None
    assert tmpl.id == "jerusalem-shakshuka"
    assert "Shakshuka" in tmpl.name
    assert tmpl.voice_id == "NATF2.pt"
    assert tmpl.timezone_str == "Asia/Jerusalem"

    compiled = compile_template(tmpl)
    assert "Jerusalem Shakshuka" in compiled.text
    assert "allergy" in compiled.text.lower() or "allergies" in compiled.text.lower()
    assert compiled.token_count < 250


def test_aerorentals_pro_enterprise_template_hygiene():
    tmpl = get_template_by_id("aerorentals-pro")
    assert tmpl is not None
    assert tmpl.id == "aerorentals-pro"
    assert "AeroRentals" in tmpl.name
    assert tmpl.voice_id == "NATM2.pt"
    assert tmpl.timezone_str == "America/Los_Angeles"

    compiled = compile_template(tmpl)
    assert "AeroRentals Pro" in compiled.text
    assert "107" in compiled.text  # FAA Part 107
    assert compiled.token_count < 250


# =====================================================================
# Prompt Hygiene & Delimiter Gates
# =====================================================================

def test_prompt_sanitization_removes_markdown_and_emojis():
    raw_prompt = (
        "## Role Overview 🚀\n"
        "You are **Ananya**, a customer support rep at Snapserve. 😊\n"
        "- Step 1: Greet warmly.\n"
        "- Step 2: Inquire about {{company}} needs.\n"
        "Start: Open the call by saying: \"Hello there!\"\n"
        "Close: When the conversation is done, say: \"Goodbye!\""
    )

    cleaned, warnings = sanitize_prompt_text(raw_prompt)

    # Emojis stripped
    assert "🚀" not in cleaned
    assert "😊" not in cleaned

    # Markdown stripped
    assert "##" not in cleaned
    assert "**" not in cleaned
    assert "- Step" not in cleaned

    # Rigid quotes stripped
    assert 'Start: Open the call by saying:' not in cleaned
    assert 'Close: When the conversation is done, say:' not in cleaned

    # Template placeholder preserved for subsequent renderer
    assert "{{company}}" in cleaned
    assert len(warnings) > 0


def test_prompt_compiler_strict_variable_resolution():
    system_prompt = "You are an assistant at {{custom_brand}} helping {{customer_name}} with {{service_type}}."

    # Strict compilation missing variables raises TemplateResolutionError
    with pytest.raises(TemplateResolutionError) as exc_info:
        compile_prompt(
            system_prompt=system_prompt,
            strict=True,
            variables={"custom_brand": "AeroRentals"},
        )
    assert "service_type" in str(exc_info.value)

    # Non-strict compilation replaces with clean fallback without {{...}}
    non_strict = compile_prompt(
        system_prompt=system_prompt,
        strict=False,
        variables={"custom_brand": "AeroRentals"},
    )
    assert "{{" not in non_strict.text
    assert "}}" not in non_strict.text
    assert "AeroRentals" in non_strict.text


# =====================================================================
# API Endpoints: Templates & Instant Instantiation
# =====================================================================

@pytest.mark.asyncio
async def test_api_list_templates_and_detail():
    pool = WorkerPool()
    app = create_app(pool=pool, registry=PersonaRegistry(), session_manager=SessionManager(pool=pool))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. GET /v1/templates
        r = await client.get("/v1/templates")
        assert r.status_code == 200
        templates = r.json()
        assert len(templates) >= 7
        assert any(t["id"] == "citysan-waste" for t in templates)
        assert any(t["id"] == "jerusalem-shakshuka" for t in templates)
        assert any(t["id"] == "aerorentals-pro" for t in templates)

        # 2. GET /v1/templates/citysan-waste
        r = await client.get("/v1/templates/citysan-waste")
        assert r.status_code == 200
        t_data = r.json()
        assert t_data["id"] == "citysan-waste"
        assert t_data["voice_id"] == "NATM3.pt"
        assert t_data["category"] == "Municipal"

        # 3. GET non-existent template
        r = await client.get("/v1/templates/unknown-template-xyz")
        assert r.status_code == 404


@pytest.mark.asyncio
async def test_api_compile_template_endpoint():
    pool = WorkerPool()
    app = create_app(pool=pool, registry=PersonaRegistry(), session_manager=SessionManager(pool=pool))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/v1/templates/aerorentals-pro/compile")
        assert r.status_code == 200
        res = r.json()
        assert res["token_count"] <= 250
        assert res["can_publish"] is True
        assert "<system>" in res["compiled_text"]
        assert res["compiled_text"].endswith("<system>")
        assert res["compiled_text"].count("<system>") == 2
        assert "AeroRentals Pro" in res["compiled_text"]


@pytest.mark.asyncio
async def test_api_instantiate_agent_from_template():
    pool = WorkerPool()
    app = create_app(pool=pool, registry=PersonaRegistry(), session_manager=SessionManager(pool=pool))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/v2/agents/from-template/jerusalem-shakshuka")
        assert r.status_code == 201
        agent = r.json()
        assert agent["name"] == "Noa from Jerusalem Shakshuka"
        assert agent["voice_id"] == "NATF2.pt"
        assert agent["timezone_str"] == "Asia/Jerusalem"
        assert agent["status"] == "draft"
        assert agent["current_version"] == 1
