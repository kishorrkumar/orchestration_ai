"""
Integration tests for Provider Catalog and Encrypted Vault REST APIs.
Includes strict security tests ensuring zero key leakage, SSRF blocking, and log auditing.
"""

import logging

import pytest
from httpx import ASGITransport, AsyncClient

from orchestration.db.session import init_db
from orchestration.gateway.app import create_app


from orchestration.gateway.security import default_rate_limiter


@pytest.fixture
async def client():
    default_rate_limiter.reset()
    await init_db()
    app = create_app(worker_type="mock")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


@pytest.mark.asyncio
async def test_get_catalog(client: AsyncClient):
    resp = await client.get("/v1/providers/catalog")
    assert resp.status_code == 200
    catalog = resp.json()
    assert len(catalog) >= 11
    ids = [p["id"] for p in catalog]
    assert "deepgram_stt" in ids
    assert "openai_llm" in ids
    assert "cartesia_tts" in ids


@pytest.mark.asyncio
async def test_get_provider_models_and_voices(client: AsyncClient):
    resp = await client.get("/v1/providers/openai_llm/models")
    assert resp.status_code == 200
    models = resp.json()
    assert len(models) >= 2
    model_ids = [m["id"] for m in models]
    assert "gpt-4o-mini" in model_ids

    # Voices
    resp_v = await client.get("/v1/providers/cartesia_tts/voices")
    assert resp_v.status_code == 200
    voices = resp_v.json()
    assert len(voices) >= 1

    # Non-existent provider -> 404 Problem Details
    resp_404 = await client.get("/v1/providers/unknown_xyz/models")
    assert resp_404.status_code == 404
    err = resp_404.json()
    assert err["code"] == "NOT_FOUND"
    assert "unknown_xyz" in err["detail"]


@pytest.mark.asyncio
async def test_credentials_lifecycle_and_zero_key_leakage(client: AsyncClient, caplog):
    caplog.set_level(logging.DEBUG)
    secret_key_plant = "SECRET-SUPER-SENSITIVE-KEY-4f2a"

    # 1. Save credential
    save_resp = await client.post(
        "/v1/providers/fake_stt/credentials",
        json={
            "api_key": secret_key_plant,
            "label": "My Test Fake STT",
            "config": {"custom_option": "value1"},
        },
    )
    assert save_resp.status_code == 200
    saved = save_resp.json()
    assert saved["provider_id"] == "fake_stt"
    assert saved["last4"] == "4f2a"
    assert saved["status"] == "valid"

    # Security check on save response: plaintext key and ciphertext must NEVER be returned
    assert secret_key_plant not in save_resp.text
    assert "ciphertext" not in saved
    assert "nonce" not in saved

    # 2. List credentials
    list_resp = await client.get("/v1/providers/credentials")
    assert list_resp.status_code == 200
    items = list_resp.json()
    assert len(items) >= 1
    found = next((c for c in items if c["provider_id"] == "fake_stt"), None)
    assert found is not None
    assert found["last4"] == "4f2a"
    assert secret_key_plant not in list_resp.text

    # Security check on logs: verify raw key was never logged
    for record in caplog.records:
        assert secret_key_plant not in record.message

    # 3. Test stored credentials
    test_resp = await client.post(
        "/v1/providers/fake_stt/credentials/test",
        json={},
    )
    assert test_resp.status_code == 200
    test_result = test_resp.json()
    assert test_result["status"] == "valid"
    assert test_result["latency_ms"] > 0

    # 4. Test explicit invalid key without saving
    test_bad = await client.post(
        "/v1/providers/fake_stt/credentials/test",
        json={"api_key": "invalid-test-key"},
    )
    assert test_bad.status_code == 200
    assert test_bad.json()["status"] == "invalid"

    # 5. Delete credential
    del_resp = await client.delete("/v1/providers/fake_stt/credentials")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is True

    # 6. Verify deleted
    list_resp_after = await client.get("/v1/providers/credentials")
    remaining = [c for c in list_resp_after.json() if c["provider_id"] == "fake_stt"]
    assert len(remaining) == 0


@pytest.mark.asyncio
async def test_ssrf_blocked_on_save_credential(client: AsyncClient):
    # Attempt to save a private / AWS metadata URL as custom base_url
    resp = await client.post(
        "/v1/providers/openai_llm/credentials",
        json={
            "api_key": "sk-test-key",
            "config": {"base_url": "http://169.254.169.254/latest/meta-data"},
        },
    )
    assert resp.status_code == 422
    err = resp.json()
    assert err["code"] == "VALIDATION_ERROR"
    assert "blocked" in err["detail"].lower()
