"""
HTTP REST Router for Engine B Providers and Encrypted Vault Management.
Implements RFC 9457 error contracts, rate-limited connection testing, and strict key security.
"""

from __future__ import annotations

import datetime
import json
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from orchestration.db.models import ProviderCredential
from orchestration.domain.providers.catalog import (
    ModelDescriptor,
    ProviderCatalog,
    ProviderManifest,
    VoiceDescriptor,
)
from orchestration.infrastructure.vault.crypto import (
    decrypt_api_key,
    encrypt_api_key,
)
from orchestration.infrastructure.vault.ssrf import validate_base_url
from orchestration.infrastructure.vault.tester import (
    ProviderTestResult,
    test_provider_connection,
)
from orchestration.interfaces.http.dependencies import get_db_session
from orchestration.shared.errors import NotFoundError, ValidationError

logger = logging.getLogger("orchestration.interfaces.http.providers")

router = APIRouter(prefix="/v1/providers", tags=["Providers & Vault"])
_catalog = ProviderCatalog()


# Simple sliding-window rate limiter for probe testing (max 12 tests/min per IP)
class TestRateLimiter:
    def __init__(self, max_per_minute: int = 15) -> None:
        self.max_per_minute = max_per_minute
        self.calls: Dict[str, List[float]] = defaultdict(list)

    def check(self, client_ip: str) -> None:
        now = time.time()
        window = now - 60.0
        self.calls[client_ip] = [t for t in self.calls[client_ip] if t > window]
        if len(self.calls[client_ip]) >= self.max_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Connection test rate limit exceeded. Please wait a minute before re-testing.",
            )
        self.calls[client_ip].append(now)


_rate_limiter = TestRateLimiter()


def verify_credential_access(request: Request) -> None:
    """
    Guards credential access. Allows loopback/local dev requests freely;
    enforces authorization header for non-loopback requests if configured.
    """
    client_host = request.client.host if request.client else "127.0.0.1"
    is_loopback = client_host in ("127.0.0.1", "::1", "localhost", "testclient")

    from orchestration.config import settings
    expected_key = getattr(settings.gateway, "api_key", None)

    if not is_loopback and expected_key:
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                api_key = auth_header[7:].strip()
        if api_key != expected_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized access to credentials vault",
            )


# DTO Schemas
class SaveCredentialRequest(BaseModel):
    api_key: str = Field(..., min_length=1, description="Raw provider API key to encrypt")
    label: str = Field("", description="Optional user label for credential")
    config: Dict[str, Any] = Field(default_factory=dict, description="Additional config (e.g. base_url, region)")


class CredentialSummaryResponse(BaseModel):
    id: str
    workspace_id: str
    provider_id: str
    label: str
    last4: str
    status: str
    last_tested_at: datetime.datetime | None = None
    last_latency_ms: float | None = None
    config: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime.datetime


class TestCredentialRequest(BaseModel):
    api_key: str | None = Field(None, description="Optional plaintext key to test without saving")
    config: Dict[str, Any] | None = Field(None, description="Optional config to test (e.g., custom base_url)")


# --- Catalog Endpoints ---

@router.get("/catalog", response_model=List[ProviderManifest])
async def list_provider_catalog() -> List[ProviderManifest]:
    """Retrieve full catalog of supported STT, LLM, and TTS providers."""
    return _catalog.list_all()


@router.get("/{provider_id}/models", response_model=List[ModelDescriptor])
async def list_provider_models(provider_id: str) -> List[ModelDescriptor]:
    """Retrieve available models for a given provider."""
    manifest = _catalog.get(provider_id)
    if not manifest:
        raise NotFoundError(f"Provider '{provider_id}' not found in catalog")
    return manifest.models


@router.get("/{provider_id}/voices", response_model=List[VoiceDescriptor])
async def list_provider_voices(provider_id: str) -> List[VoiceDescriptor]:
    """Retrieve available voices for a TTS provider."""
    manifest = _catalog.get(provider_id)
    if not manifest:
        raise NotFoundError(f"Provider '{provider_id}' not found in catalog")
    return manifest.voices


# --- Vault Endpoints ---

@router.get("/credentials", response_model=List[CredentialSummaryResponse])
async def list_stored_credentials(
    request: Request,
    workspace_id: str = Query("wks_default"),
    session: AsyncSession = Depends(get_db_session),
) -> List[CredentialSummaryResponse]:
    """
    List all stored provider credentials in the workspace.
    SECURITY GUARANTEE: Never returns plaintext keys or ciphertexts.
    """
    verify_credential_access(request)

    stmt = select(ProviderCredential).where(ProviderCredential.workspace_id == workspace_id)
    res = await session.execute(stmt)
    records = res.scalars().all()

    items = []
    for r in records:
        try:
            cfg = json.loads(r.config_json)
        except Exception:
            cfg = {}
        items.append(
            CredentialSummaryResponse(
                id=r.id,
                workspace_id=r.workspace_id,
                provider_id=r.provider_id,
                label=r.label,
                last4=r.last4,
                status=r.status,
                last_tested_at=r.last_tested_at,
                last_latency_ms=r.last_latency_ms,
                config=cfg,
                created_at=r.created_at,
            )
        )
    return items


@router.post("/{provider_id}/credentials", response_model=CredentialSummaryResponse)
async def save_provider_credential(
    provider_id: str,
    payload: SaveCredentialRequest,
    request: Request,
    workspace_id: str = Query("wks_default"),
    session: AsyncSession = Depends(get_db_session),
) -> CredentialSummaryResponse:
    """
    Encrypts and saves an API key in the provider vault using AES-256-GCM.
    Runs a fast probe test to populate the initial status.
    """
    verify_credential_access(request)

    manifest = _catalog.get(provider_id)
    if not manifest:
        raise NotFoundError(f"Provider '{provider_id}' not found in catalog")

    raw_key = payload.api_key.strip()
    if not raw_key:
        raise ValidationError("API key cannot be empty", invalid_params=[{"name": "api_key", "reason": "empty"}])

    # Validate SSRF on custom base_url if present
    custom_base_url = payload.config.get("base_url")
    if custom_base_url:
        validate_base_url(custom_base_url)

    # Encrypt key with AES-256-GCM
    ciphertext_b64, nonce_b64, last4, key_ver = encrypt_api_key(raw_key)

    # Quick probe test
    probe = await test_provider_connection(provider_id, raw_key, payload.config)

    # Check for existing record to upsert
    stmt = select(ProviderCredential).where(
        ProviderCredential.workspace_id == workspace_id,
        ProviderCredential.provider_id == provider_id,
    )
    existing = (await session.execute(stmt)).scalar_one_or_none()

    now = datetime.datetime.now(datetime.UTC)
    config_str = json.dumps(payload.config)

    if existing:
        existing.label = payload.label or existing.label
        existing.ciphertext = ciphertext_b64
        existing.nonce = nonce_b64
        existing.key_version = key_ver
        existing.last4 = last4
        existing.status = probe.status
        existing.last_tested_at = now
        existing.last_latency_ms = probe.latency_ms
        existing.config_json = config_str
        record = existing
    else:
        record = ProviderCredential(
            workspace_id=workspace_id,
            provider_id=provider_id,
            label=payload.label or manifest.display_name,
            ciphertext=ciphertext_b64,
            nonce=nonce_b64,
            key_version=key_ver,
            last4=last4,
            status=probe.status,
            last_tested_at=now,
            last_latency_ms=probe.latency_ms,
            config_json=config_str,
        )
        session.add(record)

    await session.commit()
    await session.refresh(record)

    logger.info("Saved provider credential for '%s' (last4: %s, status: %s)", provider_id, last4, probe.status)

    return CredentialSummaryResponse(
        id=record.id,
        workspace_id=record.workspace_id,
        provider_id=record.provider_id,
        label=record.label,
        last4=record.last4,
        status=record.status,
        last_tested_at=record.last_tested_at,
        last_latency_ms=record.last_latency_ms,
        config=payload.config,
        created_at=record.created_at,
    )


@router.post("/{provider_id}/credentials/test", response_model=ProviderTestResult)
async def test_provider_credential(
    provider_id: str,
    payload: TestCredentialRequest,
    request: Request,
    workspace_id: str = Query("wks_default"),
    session: AsyncSession = Depends(get_db_session),
) -> ProviderTestResult:
    """
    Performs an authenticated probe test against the provider.
    Rate-limited to prevent abuse.
    """
    verify_credential_access(request)
    client_ip = request.client.host if request.client else "unknown"
    _rate_limiter.check(client_ip)

    config = payload.config or {}

    # Case A: Plaintext key passed in request
    if payload.api_key:
        if config.get("base_url"):
            validate_base_url(config["base_url"])
        return await test_provider_connection(provider_id, payload.api_key, config)

    # Case B: Test already-stored key from vault
    stmt = select(ProviderCredential).where(
        ProviderCredential.workspace_id == workspace_id,
        ProviderCredential.provider_id == provider_id,
    )
    existing = (await session.execute(stmt)).scalar_one_or_none()
    if not existing:
        raise NotFoundError(f"No stored credentials found for provider '{provider_id}'")

    try:
        decrypted_key = decrypt_api_key(
            existing.ciphertext,
            existing.nonce,
            key_version=existing.key_version,
        )
    except Exception as e:
        logger.error("Failed to decrypt stored key for provider %s: %s", provider_id, e)
        existing.status = "invalid"
        await session.commit()
        return ProviderTestResult(status="invalid", latency_ms=0.0, error_message="Decryption failed; key corrupted")

    try:
        stored_cfg = json.loads(existing.config_json)
    except Exception:
        stored_cfg = {}

    merged_cfg = {**stored_cfg, **config}
    if merged_cfg.get("base_url"):
        validate_base_url(merged_cfg["base_url"])

    result = await test_provider_connection(provider_id, decrypted_key, merged_cfg)

    # Update DB record with fresh test results
    existing.status = result.status
    existing.last_tested_at = datetime.datetime.now(datetime.UTC)
    existing.last_latency_ms = result.latency_ms
    await session.commit()

    return result


@router.delete("/{provider_id}/credentials")
async def delete_provider_credential(
    provider_id: str,
    request: Request,
    workspace_id: str = Query("wks_default"),
    session: AsyncSession = Depends(get_db_session),
) -> Dict[str, Any]:
    """Removes stored provider credentials from vault."""
    verify_credential_access(request)

    stmt = delete(ProviderCredential).where(
        ProviderCredential.workspace_id == workspace_id,
        ProviderCredential.provider_id == provider_id,
    )
    res = await session.execute(stmt)
    await session.commit()

    if res.rowcount == 0:
        raise NotFoundError(f"No credentials found for provider '{provider_id}'")

    logger.info("Deleted provider credential for '%s'", provider_id)
    return {"deleted": True, "provider_id": provider_id}
