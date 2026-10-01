"""
Async WebSocket Client communicating with an upstream PersonaPlex server instance
(running `python -m moshi.server` or a PersonaPlexMockServer).
"""

from __future__ import annotations

import asyncio
import logging
import time
import urllib.parse
from collections.abc import AsyncGenerator
from enum import StrEnum

import numpy as np
import websockets
from websockets.asyncio.client import ClientConnection

from ..persona.registry import PersonaConfig
from ..protocol.messages import (
    AudioMessage,
    ControlAction,
    ControlMessage,
    MessageType,
    TextMessage,
    WSMessage,
    decode_message,
    encode_message,
)

logger = logging.getLogger(__name__)


class WorkerStatus(StrEnum):
    IDLE = "IDLE"
    CONNECTING = "CONNECTING"
    BUSY = "BUSY"
    UNHEALTHY = "UNHEALTHY"
    DISCONNECTED = "DISCONNECTED"


class WorkerConnectionError(Exception):
    pass


class PersonaPlexWorkerClient:
    """
    Client managing an exclusive session with an upstream PersonaPlex inference server.
    """

    def __init__(
        self,
        worker_id: str,
        host: str = "localhost",
        port: int = 8998,
        use_ssl: bool = False,
        use_opus: bool = True,
        sample_rate: int = 24000,
        connect_timeout: float = 15.0,
        handshake_timeout: float = 60.0,
    ):
        self.worker_id = worker_id
        self.host = host
        self.port = port
        self.use_ssl = use_ssl
        self.use_opus = use_opus
        self.sample_rate = sample_rate
        self.connect_timeout = connect_timeout
        self.handshake_timeout = handshake_timeout

        self._ws: ClientConnection | None = None
        self._status: WorkerStatus = WorkerStatus.IDLE
        self._active_session_id: str | None = None
        self._connected_at: float | None = None

        self._opus_reader = None
        self._opus_writer = None

        # Telemetry
        self.frames_sent: int = 0
        self.frames_received: int = 0
        self.tokens_received: int = 0
        self.last_latency_ms: float = 0.0

    @property
    def status(self) -> WorkerStatus:
        return self._status

    @property
    def is_available(self) -> bool:
        return self._status == WorkerStatus.IDLE

    @property
    def active_session_id(self) -> str | None:
        return self._active_session_id

    def build_url(self, persona: PersonaConfig) -> str:
        protocol = "wss" if self.use_ssl else "ws"
        raw_prompt = persona.get_formatted_text_prompt().strip()
        if raw_prompt.startswith("<system>") and raw_prompt.endswith("<system>"):
            clean_prompt = raw_prompt
        elif raw_prompt:
            clean_prompt = f"<system> {raw_prompt} <system>"
        else:
            clean_prompt = "<system> You enjoy having a good conversation. <system>"

        voice = persona.get_normalized_voice_prompt()
        if not voice.endswith(".pt"):
            voice = f"{voice}.pt"

        # Upstream moshi.server accepts ONLY these exact parameters:
        query_params = {
            "text_prompt": clean_prompt,
            "voice_prompt": voice,
            "audio_temperature": str(persona.audio_temperature),
            "text_temperature": str(persona.text_temperature),
            "audio_topk": str(persona.top_k_audio),
            "text_topk": str(persona.top_k_text),
        }
        if persona.seed is not None and persona.seed != -1:
            query_params["seed"] = str(persona.seed)

        qs = urllib.parse.urlencode(query_params)
        return f"{protocol}://{self.host}:{self.port}/api/chat?{qs}"

    async def connect(self, session_id: str, persona: PersonaConfig) -> None:
        """
        Connect to upstream PersonaPlex worker, execute prompt conditioning,
        and await handshake acknowledgement (0x00).
        """
        if self._status == WorkerStatus.BUSY:
            raise WorkerConnectionError(f"Worker {self.worker_id} is already busy serving session {self._active_session_id}")

        self._status = WorkerStatus.CONNECTING
        self._active_session_id = session_id
        url = self.build_url(persona)

        # Fresh Opus codecs per session to guarantee valid Ogg container headers
        if self.use_opus:
            try:
                import sphn
                self._opus_writer = sphn.OpusStreamWriter(self.sample_rate)
                self._opus_reader = sphn.OpusStreamReader(self.sample_rate)
            except Exception as e:
                logger.warning(f"sphn library unavailable. Opus transcoding disabled: {e}")
                self.use_opus = False

        try:
            logger.info(f"Connecting worker {self.worker_id} to {url} for session {session_id}")
            self._ws = await asyncio.wait_for(
                websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=10,
                    max_size=10 * 1024 * 1024,
                ),
                timeout=self.connect_timeout,
            )

            # Wait for handshake (0x00) which signals that system prompts have finished loading
            assert self._ws is not None
            first_msg = await asyncio.wait_for(self._ws.recv(), timeout=self.handshake_timeout)
            if not isinstance(first_msg, bytes):
                raise WorkerConnectionError(f"Expected binary handshake, received {type(first_msg)}")

            msg = decode_message(first_msg)
            if msg.type != MessageType.HANDSHAKE:
                raise WorkerConnectionError(f"Expected handshake 0x00, got {msg.type}")

            self._status = WorkerStatus.BUSY
            self._connected_at = time.time()
            self.frames_sent = 0
            self.frames_received = 0
            self.tokens_received = 0
            logger.info(f"Worker {self.worker_id} handshake complete for session {session_id}")

        except Exception as e:
            self._status = WorkerStatus.UNHEALTHY
            self._active_session_id = None
            if self._ws is not None:
                await self._ws.close()
                self._ws = None
            err_desc = f"{type(e).__name__}: {e!s}" if str(e).strip() else type(e).__name__
            raise WorkerConnectionError(f"Failed to connect worker {self.worker_id}: {err_desc}") from e

    async def send_audio(self, audio_data: bytes | np.ndarray) -> None:
        """Send audio frame upstream (Kind 0x01)."""
        if self._ws is None or self._status != WorkerStatus.BUSY:
            raise WorkerConnectionError(f"Worker {self.worker_id} is not connected")

        if self.use_opus and self._opus_writer is not None:
            if isinstance(audio_data, bytes):
                samples = np.frombuffer(audio_data, dtype=np.float32)
            elif audio_data.dtype != np.float32:
                samples = audio_data.astype(np.float32)
            else:
                samples = audio_data
            self._opus_writer.append_pcm(samples)
            payload = self._opus_writer.read_bytes()
            if not payload:
                return
        else:
            if isinstance(audio_data, np.ndarray):
                if audio_data.dtype == np.float32 or audio_data.dtype == np.int16:
                    payload = audio_data.tobytes()
                else:
                    payload = audio_data.astype(np.float32).tobytes()
            else:
                payload = audio_data

        msg_bytes = encode_message(AudioMessage(data=payload))
        await self._ws.send(msg_bytes)
        self.frames_sent += 1

    async def send_control(self, action: ControlAction) -> None:
        """Send control action upstream (Kind 0x03)."""
        if self._ws is None:
            return
        await self._ws.send(encode_message(ControlMessage(action=action)))

    async def send_text(self, text: str) -> None:
        """Send text message upstream (Kind 0x02)."""
        if self._ws is None:
            return
        await self._ws.send(encode_message(TextMessage(text=text)))

    async def recv_messages(self) -> AsyncGenerator[WSMessage, None]:
        """Async generator yielding incoming messages from the upstream worker."""
        if self._ws is None:
            raise WorkerConnectionError("Worker not connected")

        try:
            async for raw in self._ws:
                if not isinstance(raw, bytes):
                    continue
                try:
                    msg = decode_message(raw)
                    if isinstance(msg, AudioMessage):
                        self.frames_received += 1
                        if self.use_opus and self._opus_reader is not None and msg.data.startswith(b"OggS"):
                            self._opus_reader.append_bytes(msg.data)
                            pcm = self._opus_reader.read_pcm()
                            if len(pcm) > 0:
                                yield AudioMessage(data=pcm.astype(np.float32).tobytes())
                        else:
                            yield msg
                    elif isinstance(msg, TextMessage):
                        self.tokens_received += 1
                        yield msg
                    else:
                        yield msg
                except (GeneratorExit, asyncio.CancelledError):
                    break
                except Exception as ex:
                    if "closed channel" in str(ex).lower():
                        break
                    logger.warning(f"Error decoding worker message: {ex}")
        except websockets.ConnectionClosedOK:
            logger.info(f"Worker {self.worker_id} connection closed cleanly by upstream (1000 OK)")
        except websockets.ConnectionClosedError as ce:
            logger.warning(f"Worker {self.worker_id} connection closed with error code {ce.code}: {ce.reason}")
        except websockets.ConnectionClosed as cc:
            logger.info(f"Worker {self.worker_id} connection closed: {cc.code}")
        finally:
            await self.close()

    async def close(self) -> None:
        """Close connection and reset worker state to IDLE."""
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        self._status = WorkerStatus.IDLE
        self._active_session_id = None
        if self.use_opus:
            try:
                import sphn
                self._opus_reader = sphn.OpusStreamReader(self.sample_rate)
                self._opus_writer = sphn.OpusStreamWriter(self.sample_rate)
            except Exception:
                pass
