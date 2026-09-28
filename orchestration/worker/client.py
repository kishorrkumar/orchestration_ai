"""
Async WebSocket Client communicating with an upstream PersonaPlex server instance
(running `python -m moshi.server` or a PersonaPlexMockServer).
"""

from __future__ import annotations
import asyncio
from enum import Enum
import logging
import time
from typing import AsyncGenerator, Optional
import urllib.parse

import numpy as np
import websockets
from websockets.asyncio.client import ClientConnection

from ..persona.registry import PersonaConfig
from ..protocol.messages import (
    WSMessage,
    MessageType,
    HandshakeMessage,
    AudioMessage,
    TextMessage,
    ControlMessage,
    ControlAction,
    MetadataMessage,
    ErrorMessage,
    PingMessage,
    encode_message,
    decode_message,
)
from ..protocol.audio import FRAME_SIZE, float32_to_int16

logger = logging.getLogger(__name__)


class WorkerStatus(str, Enum):
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
        connect_timeout: float = 10.0,
        handshake_timeout: float = 15.0,
    ):
        self.worker_id = worker_id
        self.host = host
        self.port = port
        self.use_ssl = use_ssl
        self.connect_timeout = connect_timeout
        self.handshake_timeout = handshake_timeout

        self._ws: Optional[ClientConnection] = None
        self._status: WorkerStatus = WorkerStatus.IDLE
        self._active_session_id: Optional[str] = None
        self._connected_at: Optional[float] = None

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
    def active_session_id(self) -> Optional[str]:
        return self._active_session_id

    def build_url(self, persona: PersonaConfig) -> str:
        protocol = "wss" if self.use_ssl else "ws"
        query_params = {
            "text_prompt": persona.get_formatted_text_prompt(),
            "voice_prompt": persona.get_normalized_voice_prompt(),
            "audio_temperature": str(persona.audio_temperature),
            "text_temperature": str(persona.text_temperature),
            "audio_topk": str(persona.top_k_audio),
            "text_topk": str(persona.top_k_text),
        }
        if persona.seed is not None:
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
            raise WorkerConnectionError(f"Failed to connect worker {self.worker_id}: {str(e)}") from e

    async def send_audio(self, audio_data: bytes | np.ndarray) -> None:
        """Send audio frame upstream (Kind 0x01)."""
        if self._ws is None or self._status != WorkerStatus.BUSY:
            raise WorkerConnectionError(f"Worker {self.worker_id} is not connected")

        if isinstance(audio_data, np.ndarray):
            if audio_data.dtype == np.float32:
                # Upstream server accepts Opus or raw PCM
                payload = audio_data.tobytes()
            elif audio_data.dtype == np.int16:
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
                    if msg.type == MessageType.AUDIO:
                        self.frames_received += 1
                    elif msg.type == MessageType.TEXT:
                        self.tokens_received += 1
                    yield msg
                except Exception as ex:
                    logger.warning(f"Error decoding worker message: {ex}")
        except websockets.ConnectionClosed:
            logger.info(f"Worker {self.worker_id} connection closed by upstream")
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
