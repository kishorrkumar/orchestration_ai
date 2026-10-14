"""
Google Gemini Multimodal Live API (Bidirectional Speech-to-Speech Engine).

Features:
- Sub-400ms conversational turn latency over full-duplex WebSocket.
- Native 24 kHz speech synthesis with natural voice presets (Aoede, Puck, Charon, Fenrir, Kore).
- Streaming bidirectional audio (inbound PCM16 16k/24k -> outbound PCM16 24k).
- Automatic server-side interruption handling & barge-in.
- Full integration with Snapserve Structured JSON tools (calendar, pricing, CRM).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from typing import Any, Callable, Coroutine

import numpy as np
import websockets
from websockets.client import WebSocketClientProtocol

from ...audio.resample import resample_oneshot
from ...tools import default_registry

logger = logging.getLogger("orchestration.engines.gemini_live")

GEMINI_LIVE_WS_BASE = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent"


class GeminiLiveSession:
    """
    Manages a single full-duplex real-time voice session with Gemini Multimodal Live API.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "models/gemini-2.0-flash-exp",
        voice_name: str = "Aoede",
        system_instruction: str = "",
        sample_rate: int = 16000,
        enable_tools: bool = True,
    ) -> None:
        self.api_key = api_key.strip()
        self.model = model if model.startswith("models/") else f"models/{model}"
        self.voice_name = voice_name
        self.system_instruction = system_instruction
        self.sample_rate = sample_rate
        self.enable_tools = enable_tools

        self.ws: WebSocketClientProtocol | None = None
        self.is_connected: bool = False
        self.is_ready: bool = False
        self._receive_task: asyncio.Task | None = None
        self._on_audio_chunk: Callable[[bytes], Coroutine[Any, Any, None]] | None = None
        self._on_text_token: Callable[[str], Coroutine[Any, Any, None]] | None = None
        self._on_interrupted: Callable[[], Coroutine[Any, Any, None]] | None = None
        self._on_tool_event: Callable[[str, dict[str, Any]], Coroutine[Any, Any, None]] | None = None

        self.turns_count = 0
        self.connected_at: float = 0.0

    async def connect(self) -> None:
        """Establish WebSocket connection to Google Gemini Live API and perform setup handshake."""
        t0 = time.perf_counter()
        url = f"{GEMINI_LIVE_WS_BASE}?key={self.api_key}"
        logger.info(f"Connecting to Gemini Live API ({self.model}, voice: {self.voice_name})...")

        try:
            self.ws = await websockets.connect(
                url,
                ping_interval=15,
                ping_timeout=10,
                close_timeout=5,
                max_size=10 * 1024 * 1024,
            )
            self.is_connected = True
            self.connected_at = time.time()
        except Exception as e:
            logger.error(f"Failed to connect to Gemini Live API: {e}")
            raise ConnectionError(f"Could not connect to Gemini Live: {e}") from e

        # Prepare setup payload with tools and system prompt
        setup_payload: dict[str, Any] = {
            "setup": {
                "model": self.model,
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {
                                "voiceName": self.voice_name,
                            }
                        }
                    },
                },
            }
        }

        if self.system_instruction:
            setup_payload["setup"]["systemInstruction"] = {
                "parts": [{"text": self.system_instruction}]
            }

        if self.enable_tools:
            schemas = default_registry.get_schemas()
            if schemas:
                func_declarations = []
                for s in schemas:
                    fn = s.get("function", {})
                    if fn.get("name"):
                        func_declarations.append({
                            "name": fn["name"],
                            "description": fn.get("description", ""),
                            "parameters": fn.get("parameters", {}),
                        })
                if func_declarations:
                    setup_payload["setup"]["tools"] = [
                        {"functionDeclarations": func_declarations}
                    ]

        await self.ws.send(json.dumps(setup_payload))
        logger.info(f"Sent Gemini Live setup handshake. Waiting for setupComplete...")

        # Wait for setup confirmation
        resp_raw = await self.ws.recv()
        resp = json.loads(resp_raw)
        if "setupComplete" in resp:
            self.is_ready = True
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
            logger.info(f"Gemini Live session ready in {elapsed_ms} ms!")
        else:
            logger.warning(f"Unexpected initial Gemini Live response: {resp}")
            self.is_ready = True

    def start_receiving(
        self,
        on_audio: Callable[[bytes], Coroutine[Any, Any, None]],
        on_text: Callable[[str], Coroutine[Any, Any, None]] | None = None,
        on_interrupted: Callable[[], Coroutine[Any, Any, None]] | None = None,
        on_tool_event: Callable[[str, dict[str, Any]], Coroutine[Any, Any, None]] | None = None,
    ) -> asyncio.Task:
        """Start listening for downstream audio and events from Gemini."""
        self._on_audio_chunk = on_audio
        self._on_text_token = on_text
        self._on_interrupted = on_interrupted
        self._on_tool_event = on_tool_event
        self._receive_task = asyncio.create_task(self._receive_loop())
        return self._receive_task

    async def send_audio_frame(self, pcm_bytes: bytes, sample_rate: int | None = None) -> None:
        """
        Send raw PCM16 microphone audio chunk to Gemini Live API.
        Default format: audio/pcm;rate=16000 or rate=24000.
        """
        if not self.ws or not self.is_connected:
            return

        rate = sample_rate or self.sample_rate
        b64_audio = base64.b64encode(pcm_bytes).decode("utf-8")
        payload = {
            "realtimeInput": {
                "mediaChunks": [
                    {
                        "mimeType": f"audio/pcm;rate={rate}",
                        "data": b64_audio,
                    }
                ]
            }
        }
        await self.ws.send(json.dumps(payload))

    async def send_text_message(self, text: str) -> None:
        """Send explicit text message to Gemini session."""
        if not self.ws or not self.is_connected:
            return

        payload = {
            "clientContent": {
                "turns": [
                    {
                        "role": "user",
                        "parts": [{"text": text}],
                    }
                ],
                "turnComplete": True,
            }
        }
        await self.ws.send(json.dumps(payload))

    async def _receive_loop(self) -> None:
        """Loop reading messages from Gemini WebSocket."""
        try:
            while self.is_connected and self.ws:
                msg_raw = await self.ws.recv()
                msg = json.loads(msg_raw)

                server_content = msg.get("serverContent")
                if server_content:
                    # 1. Check for model interruption (Barge-In)
                    if server_content.get("interrupted"):
                        logger.info("[GeminiLive] Native server-side interruption signal received")
                        if self._on_interrupted:
                            await self._on_interrupted()

                    # 2. Extract model turn audio & text
                    model_turn = server_content.get("modelTurn")
                    if model_turn:
                        parts = model_turn.get("parts", [])
                        for part in parts:
                            inline_data = part.get("inlineData")
                            if inline_data:
                                mime = inline_data.get("mimeType", "")
                                b64_data = inline_data.get("data", "")
                                if b64_data and "audio/pcm" in mime:
                                    raw_pcm24k = base64.b64decode(b64_data)
                                    # Gemini emits 24,000 Hz PCM16 by default
                                    if self._on_audio_chunk:
                                        await self._on_audio_chunk(raw_pcm24k)

                            text_part = part.get("text")
                            if text_part and self._on_text_token:
                                await self._on_text_token(text_part)

                # 3. Handle Tool Calls
                tool_call = msg.get("toolCall")
                if tool_call:
                    await self._handle_gemini_tool_call(tool_call)

        except asyncio.CancelledError:
            pass
        except websockets.exceptions.ConnectionClosed as cc:
            logger.info(f"Gemini Live connection closed: {cc.code}")
        except Exception as e:
            logger.warning(f"Gemini Live receive error: {e}")
        finally:
            self.is_connected = False

    async def _handle_gemini_tool_call(self, tool_call: dict[str, Any]) -> None:
        """Execute tool call received from Gemini and reply with toolResponse."""
        function_calls = tool_call.get("functionCalls", [])
        function_responses = []

        for call in function_calls:
            call_id = call.get("id", "")
            fn_name = call.get("name", "")
            fn_args = call.get("args", {})
            logger.info(f"[GeminiLive TOOL] Gemini invoked tool: {fn_name}({fn_args})")

            # Execute tool in Snapserve registry
            t_res = await default_registry.execute(fn_name, fn_args)

            if self._on_tool_event:
                await self._on_tool_event(fn_name, {
                    "args": fn_args,
                    "status": t_res.status,
                    "summary": t_res.summary,
                    "data": t_res.data,
                })

            function_responses.append({
                "response": {
                    "output": t_res.data or {"summary": t_res.summary, "status": t_res.status}
                },
                "id": call_id,
            })

        # Return tool response to Gemini Live
        tool_resp_payload = {
            "toolResponse": {
                "functionResponses": function_responses,
            }
        }
        if self.ws and self.is_connected:
            await self.ws.send(json.dumps(tool_resp_payload))
            logger.info(f"[GeminiLive TOOL] Sent toolResponse for {len(function_responses)} calls")

    async def close(self) -> None:
        """Close WebSocket connection cleanly."""
        self.is_connected = False
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
            self.ws = None
        logger.info("Closed Gemini Live session.")


async def resolve_gemini_api_key(
    passed_key: str | None = None,
    session_factory: Any | None = None,
) -> str | None:
    """
    Resolves the effective Google Gemini API key by checking in order:
    1. Plaintext key passed in request / UI query parameter
    2. Environment variables: GEMINI_API_KEY, GOOGLE_API_KEY
    3. app_settings.GEMINI_API_KEY
    4. Encrypted DB Vault (ProviderCredential where provider_id in ('gemini_live', 'google_gemini', 'gemini'))
    """
    if passed_key and passed_key.strip():
        return passed_key.strip()

    env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    from ...settings import app_settings

    if app_settings.GEMINI_API_KEY and app_settings.GEMINI_API_KEY.strip():
        return app_settings.GEMINI_API_KEY.strip()

    # Check database vault
    if session_factory is not None:
        try:
            from sqlalchemy import select
            from ...db.models import ProviderCredential
            from ..vault.crypto import decrypt_api_key

            async with session_factory() as session:
                stmt = select(ProviderCredential).where(
                    ProviderCredential.provider_id.in_(["gemini_live", "google_gemini", "gemini"])
                )
                res = await session.execute(stmt)
                records = res.scalars().all()
                for rec in records:
                    try:
                        k = decrypt_api_key(rec.ciphertext, rec.nonce, key_version=rec.key_version)
                        if k and k.strip():
                            return k.strip()
                    except Exception as e:
                        logger.warning(f"Failed to decrypt provider credential {rec.provider_id}: {e}")
        except Exception as e:
            logger.debug(f"DB vault key lookup notice: {e}")

    return None

