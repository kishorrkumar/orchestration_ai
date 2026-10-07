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

from ..persona.registry import OFFICIAL_VOICE_PRESETS, PersonaConfig
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
        self.last_connect_metrics: dict[str, float] = {}
        self.last_frame_step_ms: float = 0.0
        self._last_frame_recv_time: float = 0.0

    @property
    def status(self) -> WorkerStatus:
        return self._status

    @property
    def is_available(self) -> bool:
        return self._status == WorkerStatus.IDLE

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and self._status == WorkerStatus.BUSY


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
        if not voice.endswith(".pt") and not voice.endswith(".wav"):
            voice = f"{voice}.pt"
        if voice not in OFFICIAL_VOICE_PRESETS and not voice.endswith(".wav"):
            logger.warning(f"Voice preset '{voice}' is not an official PersonaPlex preset. Falling back to 'NATF2.pt'")
            voice = "NATF2.pt"

        # Upstream moshi.server accepts ONLY text_prompt, voice_prompt, etc.
        # Upstream moshi/server.py line 171 has a known bug: `request["seed"]` instead of `request.query["seed"]`,
        # which raises KeyError in aiohttp if seed is present in the query string.
        query_params = {
            "text_prompt": clean_prompt,
            "voice_prompt": voice,
            "audio_temperature": str(persona.audio_temperature),
            "text_temperature": str(persona.text_temperature),
            "audio_topk": str(persona.top_k_audio),
            "text_topk": str(persona.top_k_text),
        }

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

        t0 = time.perf_counter()
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
            t_tcp = time.perf_counter()

            # Wait for handshake (0x00) which signals that system prompts have finished loading
            assert self._ws is not None
            first_msg = await asyncio.wait_for(self._ws.recv(), timeout=self.handshake_timeout)
            t_handshake = time.perf_counter()

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
            self.last_connect_metrics = {
                "tcp_connect_ms": round((t_tcp - t0) * 1000, 2),
                "priming_wait_ms": round((t_handshake - t_tcp) * 1000, 2),
                "total_connect_ms": round((t_handshake - t0) * 1000, 2),
            }
            logger.info(
                f"Worker {self.worker_id} handshake complete for session {session_id} in "
                f"{self.last_connect_metrics['total_connect_ms']}ms "
                f"(TCP: {self.last_connect_metrics['tcp_connect_ms']}ms, "
                f"Priming: {self.last_connect_metrics['priming_wait_ms']}ms)"
            )

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

    async def send_raw(self, raw_bytes: bytes) -> None:
        """Send raw binary frame directly upstream to worker with zero transcoding."""
        if self._ws is None or self._status != WorkerStatus.BUSY:
            raise WorkerConnectionError(f"Worker {self.worker_id} is not connected")
        await self._ws.send(raw_bytes)
        self.frames_sent += 1

    async def recv_raw_frames(self) -> AsyncGenerator[bytes, None]:
        """Yield raw untouched binary frames from upstream worker."""
        if self._ws is None:
            raise WorkerConnectionError("Worker not connected")
        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    self.frames_received += 1
                    yield raw
        except websockets.ConnectionClosedOK:
            logger.info(f"Worker {self.worker_id} connection closed cleanly (1000 OK)")
        except websockets.ConnectionClosedError as ce:
            logger.warning(f"Worker {self.worker_id} connection closed with code {ce.code}: {ce.reason}")
        except websockets.ConnectionClosed as cc:
            logger.info(f"Worker {self.worker_id} connection closed: {cc.code}")
        finally:
            await self.close(mark_idle=False)

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
                        now = time.perf_counter()
                        if self._last_frame_recv_time > 0:
                            self.last_frame_step_ms = round((now - self._last_frame_recv_time) * 1000, 2)
                        self._last_frame_recv_time = now
                        if self.use_opus and self._opus_reader is not None:
                            if len(msg.data) == 1920 * 4 and not msg.data.startswith(b"OggS"):
                                # Raw PCM from mock server in unit tests
                                yield msg
                            else:
                                try:
                                    self._opus_reader.append_bytes(msg.data)
                                    pcm = self._opus_reader.read_pcm()
                                    if len(pcm) > 0:
                                        yield AudioMessage(data=pcm.astype(np.float32).tobytes())
                                except Exception as opus_err:
                                    logger.debug(f"Opus decode note: {opus_err}")
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

    async def probe_health(self) -> bool:
        """Probe if upstream worker host and port is open and accepting TCP connections."""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port),
                timeout=1.0,
            )
            writer.close()
            await writer.wait_closed()
            return True
        except Exception as e:
            logger.warning(f"Worker {self.worker_id} ({self.host}:{self.port}) health probe failed: {e}")
            return False

    async def close(self, mark_idle: bool = True) -> None:
        """Close connection and reset worker state to IDLE if requested."""
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        if mark_idle:
            self._status = WorkerStatus.IDLE
        self._active_session_id = None
        if self.use_opus:
            try:
                import sphn
                self._opus_reader = sphn.OpusStreamReader(self.sample_rate)
                self._opus_writer = sphn.OpusStreamWriter(self.sample_rate)
            except Exception:
                pass
