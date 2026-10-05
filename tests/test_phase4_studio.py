"""
Integration tests for Phase 4: Lean React 18 Studio and UI Endpoints.
Verifies studio routes, prompt compilation preview, timezone directory, and legacy fallback.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from orchestration.gateway.app import create_app
from orchestration.persona.registry import PersonaRegistry
from orchestration.session.manager import SessionManager
from orchestration.worker.pool import WorkerPool


@pytest.mark.asyncio
async def test_lean_studio_ui_routes():
    """Verify Lean Studio and legacy console routing."""
    pool = WorkerPool()
    app = create_app(pool=pool, registry=PersonaRegistry(), session_manager=SessionManager(pool=pool))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Main Lean Studio at /
        r = await client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "PersonaPlex Voice Agent Studio" in r.text
        assert 'id="root"' in r.text
        assert "react" in r.text.lower()

        # 2. Studio at /studio
        r = await client.get("/studio")
        assert r.status_code == 200
        assert "PersonaPlex Voice Agent Studio" in r.text

        # 3. Legacy developer console at /console/legacy
        r = await client.get("/console/legacy")
        assert r.status_code == 200
        assert "PersonaPlex" in r.text

        # 4. Legacy console alias /console
        r = await client.get("/console")
        assert r.status_code == 200


@pytest.mark.asyncio
async def test_studio_api_prompt_compile_and_timezones():
    """Verify backend APIs powering the Lean Studio UI."""
    pool = WorkerPool()
    app = create_app(pool=pool, registry=PersonaRegistry(), session_manager=SessionManager(pool=pool))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Timezones directory
        r = await client.get("/v1/timezones")
        assert r.status_code == 200
        tz_list = r.json()
        assert isinstance(tz_list, list)
        assert any(t["name"] == "Asia/Kolkata" for t in tz_list)
        assert any(t["name"] == "America/New_York" for t in tz_list)

        # 2. Live Prompt Compilation endpoint
        compile_req = {
            "system_prompt": "You are a friendly scheduling assistant at Bright Smile Clinic.",
            "greeting_text": "Good morning, welcome to Bright Smile Clinic!",
            "greeting_mode": "agent_first",
            "ending_text": "Have a wonderful day, goodbye!",
            "agent_name": "Smile Bot",
            "timezone": "America/New_York",
        }
        r = await client.post("/v1/prompts/compile", json=compile_req)
        assert r.status_code == 200
        compiled = r.json()
        assert compiled["token_count"] > 0
        assert compiled["token_count"] <= 350
        assert compiled["can_publish"] is True
        assert "<system>" in compiled["compiled_text"]
        assert "Bright Smile Clinic" in compiled["compiled_text"]
        assert "America/New_York" in compiled["time_line"] or compiled["day_part"] in ("morning", "afternoon", "evening", "night")
