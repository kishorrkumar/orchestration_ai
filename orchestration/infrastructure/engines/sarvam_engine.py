"""
Sarvam AI Engine B Pipeline Implementation (Saaras STT + Sarvam LLM + Bulbul TTS).

Features:
- Native Indic full-duplex speech-to-speech with 22+ languages (Hindi, Tamil, Telugu, Kannada, Malayalam, etc.)
- Saaras ASR (saaras:v4 / saaras:v3) with auto language detection and code-mixing support
- Sarvam LLM (sarvam-105b-conversations / sarvam-105b) with OpenAI-compatible streaming
- Bulbul TTS (bulbul:v4-flash / bulbul:v3) with expressive voice personas
- Full integration with Snapserve tools, interruption handling, and Vault key resolution
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import re
import time
import wave
from typing import Any, AsyncIterator, Callable, Coroutine

import httpx
from sqlalchemy import select

from ...audio.dsp import compute_rms
from ...audio.resample import resample_oneshot
from ...db.models import ProviderCredential
from ...infrastructure.vault.crypto import decrypt_api_key
from ...settings import app_settings
from ...tools import default_registry

logger = logging.getLogger("orchestration.engines.sarvam")

SARVAM_API_BASE = "https://api.sarvam.ai"


async def resolve_sarvam_api_key(
    passed_key: str | None = None,
    session_factory: Any = None,
) -> str | None:
    """
    Resolves Sarvam AI Subscription Key with waterfall priority:
    1. Passed parameter (e.g. pasted directly in Test Call UI modal)
    2. Environment variable SARVAM_API_KEY
    3. app_settings.SARVAM_API_KEY
    4. Encrypted ProviderCredential in Vault DB (sarvam_stt, sarvam_llm, sarvam_tts, sarvam)
    """
    if passed_key and passed_key.strip():
        return passed_key.strip()

    env_key = os.environ.get("SARVAM_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    if getattr(app_settings, "SARVAM_API_KEY", None) and app_settings.SARVAM_API_KEY.strip():
        return app_settings.SARVAM_API_KEY.strip()

    if session_factory:
        try:
            async with session_factory() as db:
                stmt = select(ProviderCredential).where(
                    ProviderCredential.provider_id.in_(("sarvam_stt", "sarvam_llm", "sarvam_tts", "sarvam"))
                )
                res = await db.execute(stmt)
                records = res.scalars().all()
                for rec in records:
                    if rec.ciphertext:
                        decrypted = decrypt_api_key(rec.ciphertext)
                        if decrypted and decrypted.strip():
                            return decrypted.strip()
        except Exception as e:
            logger.debug(f"Failed to query Sarvam key from vault: {e}")

    return None


class SarvamClient:
    """Async HTTP client for Sarvam AI Speech-to-Text, Chat, and Text-to-Speech APIs."""

    def __init__(self, api_key: str, base_url: str = SARVAM_API_BASE) -> None:
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "api-subscription-key": self.api_key,
            "Authorization": f"Bearer {self.api_key}",
        }

    async def transcribe(
        self,
        pcm16_bytes: bytes,
        sample_rate: int = 16000,
        language_code: str = "hi-IN",
        model: str = "saaras:v4",
    ) -> str:
        """
        Transcribes raw PCM16 audio using Sarvam Saaras ASR.
        Encodes in-memory WAV container and posts to /speech-to-text.
        """
        if not pcm16_bytes or len(pcm16_bytes) < 3200:  # < 100ms
            return ""

        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16_bytes)
        wav_bytes = wav_buf.getvalue()

        url = f"{self.base_url}/speech-to-text"
        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data = {
            "model": model,
            "language_code": language_code if language_code != "auto" else "unknown",
        }

        timeout = httpx.Timeout(12.0, connect=4.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, headers={"api-subscription-key": self.api_key}, files=files, data=data)
            if resp.status_code == 200:
                body = resp.json()
                return body.get("transcript", "").strip()
            logger.warning(f"Sarvam STT failed HTTP {resp.status_code}: {resp.text}")
            return ""

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str = "sarvam-105b-conversations",
        temperature: float = 0.6,
        max_tokens: int = 250,
    ) -> str:
        """Non-streaming chat completion from Sarvam LLM."""
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        timeout = httpx.Timeout(15.0, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, headers=self.headers, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "").strip()
            logger.warning(f"Sarvam LLM failed HTTP {resp.status_code}: {resp.text}")
            return ""

    async def stream_chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str = "sarvam-105b-conversations",
        temperature: float = 0.6,
        max_tokens: int = 250,
    ) -> AsyncIterator[str]:
        """Streaming chat completion yielding tokens as they arrive."""
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        timeout = httpx.Timeout(20.0, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, headers=self.headers, json=payload) as resp:
                if resp.status_code != 200:
                    err_text = await resp.aread()
                    logger.warning(f"Sarvam stream failed HTTP {resp.status_code}: {err_text.decode('utf-8', errors='ignore')}")
                    return

                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                    except Exception:
                        continue

    async def synthesize(
        self,
        text: str,
        speaker: str = "shubh",
        target_language_code: str = "hi-IN",
        model: str = "bulbul:v4-flash",
        sample_rate: int = 16000,
    ) -> bytes:
        """
        Synthesizes speech using Sarvam Bulbul TTS.
        Returns raw linear PCM16 audio bytes at requested sample rate.
        """
        clean_text = text.strip()
        if not clean_text:
            return b""

        url = f"{self.base_url}/text-to-speech"
        payload = {
            "text": clean_text,
            "target_language_code": target_language_code if target_language_code != "auto" else "en-IN",
            "speaker": speaker,
            "model": model,
            "pace": 1.0,
            "speech_sample_rate": sample_rate,
        }

        timeout = httpx.Timeout(15.0, connect=4.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, headers=self.headers, json=payload)
            if resp.status_code != 200:
                logger.warning(f"Sarvam TTS failed HTTP {resp.status_code}: {resp.text}")
                return b""

            data = resp.json()
            audios = data.get("audios", [])
            if not audios:
                return b""

            audio_b64 = audios[0]
            audio_bytes = base64.b64decode(audio_b64)

            # If wrapped in WAV container, extract raw PCM frames
            if audio_bytes.startswith(b"RIFF"):
                try:
                    with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                        orig_rate = wf.getframerate()
                        raw_pcm = wf.readframes(wf.getnframes())
                        if orig_rate != sample_rate:
                            raw_pcm = resample_oneshot(raw_pcm, orig_rate, sample_rate)
                        return raw_pcm
                except Exception as e:
                    logger.warning(f"Failed parsing Sarvam WAV container: {e}")
                    return audio_bytes[44:]  # Skip standard 44-byte WAV header

            return audio_bytes
