"""Tests for Engine B Agent Editor integration, persistence, and prompt validation."""

import json

import pytest
from httpx import ASGITransport, AsyncClient

from orchestration.domain.engines.spec import (
    CascadedPipelineSpec,
    LLMPipelineConfig,
    STTPipelineConfig,
    TTSPipelineConfig,
    TurnPipelineConfig,
)
from orchestration.gateway.app import create_app
from orchestration.gateway.security import default_rate_limiter


@pytest.mark.asyncio
async def test_agent_engine_b_full_editor_lifecycle():
    default_rate_limiter.reset()
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Create Engine B agent with custom pipeline_json, language, and non-PersonaPlex voice
        spec = CascadedPipelineSpec(
            stt=STTPipelineConfig(provider_id="fake_stt", model="mock-fast-stt", language="ta-IN"),
            llm=LLMPipelineConfig(provider_id="fake_llm", model="mock-stream-llm", temperature=0.7),
            tts=TTSPipelineConfig(provider_id="fake_tts", model="mock-fast-tts", voice="mock-alex"),
            turn=TurnPipelineConfig(strategy="auto"),
        )
        pipeline_json_str = spec.model_dump_json()

        long_prompt = "You are a customer support agent. " * 50  # ~450 tokens, exceeds 350 hard limit

        create_payload = {
            "name": "Engine B Tamil Assistant",
            "voice_id": "mock-alex",  # Not in OFFICIAL_PRESETS!
            "greeting": "Vanakkam, epdi irukinga?",
            "agent_speaks_first": True,
            "system_prompt": long_prompt,
            "ending": "Nandri, vanakkam!",
            "end_call_timeout_sec": 2.0,
            "silence_timeout_sec": 15.0,
            "max_duration_sec": 600.0,
            "timezone_str": "Asia/Kolkata",
            "engine": "cascaded_cloud",
            "language": "ta-IN",
            "pipeline_json": pipeline_json_str,
        }

        # POST /v2/agents
        res = await client.post("/v2/agents", json=create_payload)
        assert res.status_code == 201, res.text
        data = res.json()
        agent_id = data["id"]
        assert data["engine"] == "cascaded_cloud"
        assert data["language"] == "ta-IN"
        assert data["voice_id"] == "mock-alex"
        parsed_spec = json.loads(data["pipeline_json"])
        assert parsed_spec["stt"]["language"] == "ta-IN"

        # 2. GET /v2/agents/{id}
        res_get = await client.get(f"/v2/agents/{agent_id}")
        assert res_get.status_code == 200
        get_data = res_get.json()
        assert get_data["engine"] == "cascaded_cloud"
        assert get_data["language"] == "ta-IN"

        # 3. Publish Engine B agent - must succeed even though prompt exceeds 350 tokens!
        res_pub = await client.post(f"/v2/agents/{agent_id}/publish", json={"change_note": "First Tamil release"})
        assert res_pub.status_code == 200, res_pub.text
        pub_data = res_pub.json()
        assert pub_data["status"] == "published"
        assert pub_data["published_version"] == 1

        # 4. Update agent (Version 2) to Hindi
        updated_spec = CascadedPipelineSpec(
            stt=STTPipelineConfig(provider_id="fake_stt", model="mock-fast-stt", language="hi-IN"),
            llm=LLMPipelineConfig(provider_id="fake_llm", model="mock-stream-llm"),
            tts=TTSPipelineConfig(provider_id="fake_tts", model="mock-fast-tts", voice="mock-priya"),
            turn=TurnPipelineConfig(strategy="provider_eot"),
        )
        res_update = await client.patch(
            f"/v2/agents/{agent_id}",
            json={
                "language": "hi-IN",
                "voice_id": "mock-priya",
                "greeting": "Namaste, kaise hain aap?",
                "pipeline_json": updated_spec.model_dump_json(),
                "change_note": "Switch to Hindi",
            },
        )
        assert res_update.status_code == 200
        up_data = res_update.json()
        assert up_data["current_version"] == 2
        assert up_data["language"] == "hi-IN"
        assert up_data["voice_id"] == "mock-priya"

        # 5. List versions
        res_vers = await client.get(f"/v2/agents/{agent_id}/versions")
        assert res_vers.status_code == 200
        versions = res_vers.json()
        assert len(versions) >= 2
        assert versions[0]["version_number"] == 2
        assert versions[0]["language"] == "hi-IN"
        assert versions[1]["version_number"] == 1
        assert versions[1]["language"] == "ta-IN"

        # 6. Revert to Version 1
        res_rev = await client.post(f"/v2/agents/{agent_id}/revert/1")
        assert res_rev.status_code == 200
        rev_data = res_rev.json()
        assert rev_data["language"] == "ta-IN"
        assert rev_data["voice_id"] == "mock-alex"
        assert json.loads(rev_data["pipeline_json"])["stt"]["language"] == "ta-IN"


@pytest.mark.asyncio
async def test_agent_engine_a_enforces_350_token_limit():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        long_prompt = "You are a customer support agent. " * 50  # ~450 tokens

        create_payload = {
            "name": "Engine A Long Prompt Agent",
            "voice_id": "natural_calm",
            "greeting": "Hello there!",
            "agent_speaks_first": True,
            "system_prompt": long_prompt,
            "ending": "Goodbye!",
            "end_call_timeout_sec": 2.0,
            "silence_timeout_sec": 12.0,
            "max_duration_sec": 600.0,
            "timezone_str": "UTC",
            "engine": "personaplex_s2s",
        }
        res = await client.post("/v2/agents", json=create_payload)
        assert res.status_code == 201
        agent_id = res.json()["id"]

        # Publishing must fail with 422 PromptTooLongError on Engine A
        res_pub = await client.post(f"/v2/agents/{agent_id}/publish", json={"change_note": "Try publish"})
        assert res_pub.status_code == 422
        assert "exceeding hard limit" in res_pub.text


@pytest.mark.asyncio
async def test_v1_agents_backward_compatibility_engine_b():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/v1/agents",
            json={
                "name": "V1 Created Engine B",
                "voice_id": "mock-alex",
                "greeting_text": "Hello from v1",
                "system_prompt": "You are a test agent.",
                "ending_text": "Bye",
                "engine": "cascaded_cloud",
                "language": "en-IN",
                "pipeline_json": '{"stt": {"language": "en-IN"}}',
                "auto_publish": False,
            },
        )
        assert res.status_code == 201, res.text
        data = res.json()
        assert data["engine"] == "cascaded_cloud"
        assert data["language"] == "en-IN"
        assert "en-IN" in data["pipeline_json"]
