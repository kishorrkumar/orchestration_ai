"""
High-Fidelity Mock PersonaPlex Server for local testing, CI, and development.

Emulates:
- The exact PersonaPlex / Moshi server WebSocket endpoint: `GET /api/chat`
- Handshake protocol: sends `0x00` after prompt initialization
- Full-duplex audio stream: 12.5 Hz (80ms per frame, 1,920 samples @ 24kHz)
- Text token generation matching SentencePiece token stream
- Single-concurrency mutual exclusion lock matching upstream `server.py`
- Barge-in / interruption handling
"""

from __future__ import annotations
import asyncio
import logging
import math
import time
from typing import Optional

import numpy as np
import websockets
from websockets.asyncio.server import Server, serve, ServerConnection

from ..protocol.messages import (
    MessageType,
    HandshakeMessage,
    AudioMessage,
    TextMessage,
    ErrorMessage,
    encode_message,
    decode_message,
)
from ..protocol.audio import (
    FRAME_SIZE,
    SAMPLE_RATE,
    compute_rms,
    generate_silence_frame,
)

logger = logging.getLogger(__name__)


class PersonaPlexMockServer:
    """
    Mock PersonaPlex server that mimics NVIDIA's server.py WebSocket behavior.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8998,
        prompt_init_delay: float = 0.05,
        frame_interval_sec: float = 0.08,  # 80ms = 12.5 Hz
    ):
        self.host = host
        self.port = port
        self.prompt_init_delay = prompt_init_delay
        self.frame_interval = frame_interval_sec

        self._server: Optional[Server] = None
        self._lock = asyncio.Lock()  # Replicates `self.lock = asyncio.Lock()` in server.py
        self._is_running = False

        # Metrics
        self.active_connections = 0
        self.total_sessions_served = 0
        self.total_frames_received = 0
        self.total_frames_generated = 0

    async def start(self) -> None:
        """Start the WebSocket mock server."""
        self._is_running = True
        self._server = await serve(
            self._handle_connection,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=10,
        )
        logger.info(f"PersonaPlexMockServer started on ws://{self.host}:{self.port}")

    async def stop(self) -> None:
        """Stop the WebSocket mock server."""
        self._is_running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        logger.info("PersonaPlexMockServer stopped")

    async def _handle_connection(self, websocket: ServerConnection) -> None:
        """Handle incoming WebSocket connection from client or orchestration gateway."""
        path = websocket.request.path if hasattr(websocket, "request") else ""
        self.active_connections += 1

        # Check if another session is already holding the GPU inference lock
        if self._lock.locked():
            logger.warning("Mock worker is busy. Rejecting concurrent session.")
            err_msg = encode_message(ErrorMessage(error="Worker busy: another session is currently active."))
            await websocket.send(err_msg)
            await websocket.close(code=1008, reason="Worker busy")
            self.active_connections -= 1
            return

        async with self._lock:
            self.total_sessions_served += 1
            logger.info(f"Mock server accepted connection on {path}")

            # Simulate system prompt loading (text + voice conditioning)
            if self.prompt_init_delay > 0:
                await asyncio.sleep(self.prompt_init_delay)

            # Send handshake 0x00 to signal readiness
            handshake_bytes = encode_message(HandshakeMessage(version=0, model=0))
            await websocket.send(handshake_bytes)
            logger.info("Mock server sent handshake 0x00")

            # Setup streaming state
            user_speaking = False
            last_user_audio_time = 0.0
            stop_event = asyncio.Event()

            # Pre-computed synthetic speech response tokens
            sample_tokens = [
                " Hello", "!", " I", " am", " your", " voice", " assistant",
                " powered", " by", " Persona", "Plex", ".", " How", " can",
                " I", " assist", " you", " today", "?"
            ]
            token_idx = 0
            audio_phase = 0.0

            async def receiver():
                nonlocal user_speaking, last_user_audio_time
                try:
                    async for raw in websocket:
                        if not isinstance(raw, bytes):
                            continue
                        msg = decode_message(raw)
                        if msg.type == MessageType.AUDIO:
                            self.total_frames_received += 1
                            # Compute audio energy
                            audio_samples = np.frombuffer(msg.data, dtype=np.float32) if len(msg.data) >= FRAME_SIZE * 4 else np.zeros(FRAME_SIZE, dtype=np.float32)
                            rms = compute_rms(audio_samples)
                            if rms > 0.02:
                                user_speaking = True
                                last_user_audio_time = time.time()
                            elif time.time() - last_user_audio_time > 0.4:
                                user_speaking = False
                except websockets.ConnectionClosed:
                    pass
                finally:
                    stop_event.set()

            async def generator():
                nonlocal token_idx, audio_phase
                step = 0
                while not stop_event.is_set():
                    t0 = time.time()
                    step += 1

                    # Check if user is currently speaking (barge-in / interruption)
                    if user_speaking:
                        # User is speaking: Agent listens and emits silence or occasional backchannel
                        out_frame = generate_silence_frame()
                        if step % 25 == 0:  # Every ~2 seconds of user speech, backchannel
                            await websocket.send(encode_message(TextMessage(text=" [backchannel: mm-hmm]")))
                    else:
                        # Agent is speaking: synthesize 24kHz tone/voice carrier
                        t = np.linspace(audio_phase, audio_phase + self.frame_interval, FRAME_SIZE, endpoint=False)
                        audio_phase += self.frame_interval
                        # Modulated harmonious wave at 220Hz (A3) with harmonics
                        carrier = (0.2 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
                        out_frame = carrier

                        # Periodically emit text token (e.g. every 2-3 frames = ~200ms)
                        if step % 3 == 0 and token_idx < len(sample_tokens):
                            token = sample_tokens[token_idx]
                            token_idx += 1
                            await websocket.send(encode_message(TextMessage(text=token)))

                    # Send agent audio frame (0x01)
                    await websocket.send(encode_message(AudioMessage(data=out_frame.tobytes())))
                    self.total_frames_generated += 1

                    # Maintain strict 80ms frame cadence (12.5 Hz)
                    elapsed = time.time() - t0
                    sleep_time = max(0.001, self.frame_interval - elapsed)
                    await asyncio.sleep(sleep_time)

            recv_task = asyncio.create_task(receiver())
            gen_task = asyncio.create_task(generator())

            # Wait until one completes or client disconnects
            done, pending = await asyncio.wait(
                [recv_task, gen_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            logger.info("Mock server session terminated")

        self.active_connections -= 1
