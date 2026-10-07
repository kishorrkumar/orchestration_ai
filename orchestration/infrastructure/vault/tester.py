"""
Lightweight Authenticated Connection Testers for Voice Engine Providers.
Makes minimal, non-billable / low-cost authenticated API calls to verify API key validity and measure server-to-vendor latency.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Literal

import httpx
from pydantic import BaseModel

from .ssrf import validate_base_url


class ProviderTestResult(BaseModel):
    status: Literal["valid", "invalid", "rate_limited", "error"]
    latency_ms: float
    error_message: str | None = None


async def test_provider_connection(
    provider_id: str,
    api_key: str,
    config: Dict[str, Any] | None = None,
) -> ProviderTestResult:
    """
    Executes a fast, safe probe request to the provider's API.
    Measures round-trip latency in milliseconds.
    """
    config = config or {}
    t0 = time.perf_counter()

    # 1. Fake / Mock providers for tests & CI
    if provider_id in ("fake_stt", "fake_llm", "fake_tts"):
        if api_key == "invalid-test-key":
            return ProviderTestResult(
                status="invalid",
                latency_ms=10.0,
                error_message="Simulated key validation failure",
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000 + 12.0
        return ProviderTestResult(status="valid", latency_ms=round(elapsed_ms, 2))

    # Real HTTP probe
    timeout = httpx.Timeout(8.0, connect=5.0)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # 2. Deepgram (STT / TTS)
            if provider_id in ("deepgram_stt", "deepgram_tts"):
                resp = await client.get(
                    "https://api.deepgram.com/v1/projects",
                    headers={"Authorization": f"Token {api_key.strip()}"},
                )

            # 3. OpenAI & Compatible
            elif provider_id == "openai_llm":
                base_url = config.get("base_url") or "https://api.openai.com/v1"
                validated_base = validate_base_url(base_url)
                resp = await client.get(
                    f"{validated_base}/models",
                    headers={"Authorization": f"Bearer {api_key.strip()}"},
                )

            # 4. Anthropic
            elif provider_id == "anthropic_llm":
                resp = await client.get(
                    "https://api.anthropic.com/v1/models",
                    headers={
                        "x-api-key": api_key.strip(),
                        "anthropic-version": "2023-06-01",
                    },
                )

            # 5. Cartesia
            elif provider_id == "cartesia_tts":
                resp = await client.get(
                    "https://api.cartesia.ai/voices",
                    headers={
                        "X-API-Key": api_key.strip(),
                        "Cartesia-Version": "2024-06-10",
                    },
                )

            # 6. ElevenLabs
            elif provider_id == "elevenlabs_tts":
                resp = await client.get(
                    "https://api.elevenlabs.io/v1/user",
                    headers={"xi-api-key": api_key.strip()},
                )

            # 7. Sarvam (STT / TTS)
            elif provider_id in ("sarvam_stt", "sarvam_tts"):
                # Lightweight call to Sarvam speech API info
                resp = await client.get(
                    "https://api.sarvam.ai/speech-to-text",
                    headers={"api-subscription-key": api_key.strip()},
                )

            else:
                return ProviderTestResult(
                    status="invalid",
                    latency_ms=0.0,
                    error_message=f"Unknown provider '{provider_id}'",
                )

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        if resp.status_code in (200, 201, 204):
            return ProviderTestResult(status="valid", latency_ms=latency_ms)
        elif resp.status_code == 429:
            return ProviderTestResult(
                status="rate_limited",
                latency_ms=latency_ms,
                error_message="Provider rate limit exceeded (HTTP 429)",
            )
        elif resp.status_code in (401, 403):
            return ProviderTestResult(
                status="invalid",
                latency_ms=latency_ms,
                error_message=f"Authentication failed (HTTP {resp.status_code})",
            )
        else:
            return ProviderTestResult(
                status="error",
                latency_ms=latency_ms,
                error_message=f"Unexpected status code HTTP {resp.status_code}",
            )

    except httpx.ConnectTimeout:
        return ProviderTestResult(
            status="error",
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            error_message="Connection timed out while reaching provider endpoint",
        )
    except httpx.ConnectError as e:
        return ProviderTestResult(
            status="error",
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            error_message=f"Connection error: {e}",
        )
    except Exception as e:
        return ProviderTestResult(
            status="error",
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            error_message=str(e),
        )
