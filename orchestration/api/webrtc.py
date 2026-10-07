"""
WebRTC Transport for PersonaPlex S2S Voice Agent Platform.
Uses open-source aiortc for browser mic (Opus) -> PCM decode -> Orchestrator -> PersonaPlex;
and PersonaPlex 24kHz response -> PCM encode -> Opus -> browser <audio>.
Signaling is handled via POST /v2/webrtc/offer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, Field

from ..audio.resample import AudioResampler, resample_oneshot
from ..db.service import AgentService
from ..db.session import get_session_factory
from ..persona.registry import default_registry
from ..prompts.compiler import compile_prompt
from ..protocol.audio import (
    FRAME_SIZE,
    SAMPLE_RATE,
    float32_to_int16,
    int16_to_float32,
)
from ..protocol.messages import AudioMessage, MessageType, decode_message, encode_message
from ..settings import app_settings
from ..worker.client import PersonaPlexWorkerClient
from ..worker.pool import WorkerPool

logger = logging.getLogger("orchestration.webrtc")

router = APIRouter(prefix="/v2/webrtc", tags=["WebRTC Transport"])

try:
    import av
    from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
    from aiortc.mediastreams import MediaStreamError
    HAS_AIORTC = True
except ImportError:
    HAS_AIORTC = False


class WebRTCOfferRequest(BaseModel):
    sdp: str = Field(..., description="Client SDP offer")
    type: str = Field(default="offer", description="SDP type ('offer')")
    agent_id: str = Field(default="default", description="Agent ID to connect with")
    token: str | None = Field(default=None, description="Optional authentication token")
    caller_name: str | None = Field(default=None, description="Optional caller name for template interpolation")


class WebRTCAnswerResponse(BaseModel):
    sdp: str = Field(..., description="Server SDP answer")
    type: str = Field(default="answer", description="SDP type ('answer')")
    session_id: str = Field(..., description="WebRTC voice session ID")


if HAS_AIORTC:
    class AgentAudioTrack(MediaStreamTrack):
        """
        Output audio track feeding 48 kHz Opus frames back to the browser.
        WebRTC standard browser audio pipeline expects 48 kHz mono/stereo frames.
        """
        kind = "audio"

        def __init__(self, sample_rate: int = 48000):
            super().__init__()
            self.sample_rate = sample_rate
            self.frame_samples = int(sample_rate * 0.02)  # 20ms frame = 960 samples @ 48kHz
            self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=150)
            self._timestamp = 0
            self._pts = 0

        def push_audio(self, pcm_data: np.ndarray) -> None:
            """Push PCM float32 or int16 samples into the playback queue."""
            if pcm_data.dtype != np.int16:
                pcm_i16 = (np.clip(pcm_data, -1.0, 1.0) * 32767.0).astype(np.int16)
            else:
                pcm_i16 = pcm_data

            # Split into 960-sample chunks
            for i in range(0, len(pcm_i16), self.frame_samples):
                chunk = pcm_i16[i : i + self.frame_samples]
                if len(chunk) < self.frame_samples:
                    padded = np.zeros(self.frame_samples, dtype=np.int16)
                    padded[: len(chunk)] = chunk
                    chunk = padded
                try:
                    self._queue.put_nowait(chunk)
                except asyncio.QueueFull:
                    # Drop oldest on severe backpressure to prevent buffer bloat
                    try:
                        self._queue.get_nowait()
                        self._queue.put_nowait(chunk)
                    except Exception:
                        pass

        def flush(self) -> None:
            """Flush queued audio on user interruption."""
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except Exception:
                    break

        async def recv(self) -> av.AudioFrame:
            """Called by aiortc media loop every 20ms to encode and transmit audio."""
            try:
                chunk = await asyncio.wait_for(self._queue.get(), timeout=0.03)
            except (TimeoutError, asyncio.QueueEmpty):
                # Send silence if queue underrun occurs
                chunk = np.zeros(self.frame_samples, dtype=np.int16)

            # Shape for av.AudioFrame: (1, frame_samples)
            data = chunk.reshape(1, -1)
            frame = av.AudioFrame.from_ndarray(data, format="s16", layout="mono")
            frame.sample_rate = self.sample_rate
            frame.pts = self._pts
            self._pts += self.frame_samples
            return frame


# Active WebRTC PeerConnections tracked for clean teardown
ACTIVE_PEER_CONNECTIONS: dict[str, Any] = {}


@router.post("/offer", response_model=WebRTCAnswerResponse)
async def webrtc_offer_endpoint(
    req: WebRTCOfferRequest,
    request: Request,
    authorization: str | None = Header(None),
):
    """
    WebRTC Signaling Offer Endpoint.
    Accepts browser SDP offer, negotiates ICE/Opus media, attaches PersonaPlex worker,
    and returns SDP answer.
    """
    if not HAS_AIORTC:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="WebRTC transport unavailable: aiortc is not installed on this host",
        )

    # 1. Bearer Token Verification
    if app_settings.AUTH_TOKEN:
        token_val = req.token
        if not token_val and authorization:
            token_val = authorization.replace("Bearer ", "").strip()
        if token_val != app_settings.AUTH_TOKEN:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized: Invalid auth token",
            )

    # 2. Worker Pool Availability
    worker_pool: WorkerPool = getattr(request.app.state, "pool", None)
    if not worker_pool:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Worker pool not available on gateway",
        )

    # 3. Resolve Agent Configuration
    session_factory = get_session_factory()
    async with session_factory() as db:
        agent_svc = AgentService(db)
        agent = None
        if req.agent_id and req.agent_id != "default":
            agent = await agent_svc.get_agent(req.agent_id)
        else:
            agents_list, _ = await agent_svc.list_agents()
            if agents_list:
                agent = agents_list[0]

        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent '{req.agent_id}' not found",
            )

        # Resolve prompt and generation parameters
        raw_prompt = agent.draft_system_prompt or "You are a helpful voice assistant."
        voice_prompt = agent.draft_voice_id or "NATF2.pt"

    compiled_prompt = compile_prompt(
        system_prompt=raw_prompt,
        agent_name=agent.name,
        caller_name=req.caller_name or "caller",
        variables={"company": "BrightNet", "customer_name": "customer"},
    )

    session_id = f"rtc_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    pc = RTCPeerConnection()
    ACTIVE_PEER_CONNECTIONS[session_id] = pc

    audio_track = AgentAudioTrack(sample_rate=48000)
    pc.addTrack(audio_track)

    data_channels: list[Any] = []

    @pc.on("datachannel")
    def on_datachannel(channel):
        logger.info(f"[{session_id}] WebRTC DataChannel established: {channel.label}")
        data_channels.append(channel)

        @channel.on("message")
        def on_message(message):
            try:
                data = json.loads(message)
                if data.get("type") == "hangup":
                    asyncio.create_task(pc.close())
            except Exception:
                pass

    def send_dc(msg: dict):
        if data_channels:
            try:
                payload = json.dumps(msg)
                for ch in data_channels:
                    ch.send(payload)
            except Exception:
                pass

    # Background worker bridge task
    async def run_worker_bridge(inbound_audio_queue: asyncio.Queue):
        lease = None
        try:
            send_dc({"type": "status", "status": "priming", "elapsed_ms": 0})
            lease = await worker_pool.acquire_worker(timeout_sec=app_settings.CONNECT_TIMEOUT_SEC)
            worker_client = PersonaPlexWorkerClient(
                worker_id=lease.worker_id,
                host=lease.host,
                port=lease.port,
                connect_timeout=app_settings.CONNECT_TIMEOUT_SEC,
            )

            # Register temporary persona config
            persona_cfg = default_registry.get_or_create_custom(
                persona_id=f"webrtc_{session_id}",
                name=agent.name,
                voice_prompt=voice_prompt,
                text_prompt=compiled_prompt.formatted_prompt,
            )

            await worker_client.connect(session_id=session_id, persona=persona_cfg)
            send_dc({"type": "status", "status": "ready"})
            send_dc({"type": "session_started", "session_id": session_id, "agent_name": agent.name})

            # Inbound feeder loop (browser 48kHz -> resample 24kHz -> 1920-sample chunks)
            inbound_samples_24k: list[np.ndarray] = []

            async def feeder_loop():
                while pc.connectionState not in ("failed", "closed"):
                    try:
                        chunk_24k = await asyncio.wait_for(inbound_audio_queue.get(), timeout=0.08)
                        inbound_samples_24k.append(chunk_24k)
                    except TimeoutError:
                        inbound_samples_24k.append(np.zeros(1920, dtype=np.float32))

                    total_len = sum(len(c) for c in inbound_samples_24k)
                    if total_len >= FRAME_SIZE:
                        combined = np.concatenate(inbound_samples_24k)
                        frame_to_send = combined[:FRAME_SIZE]
                        remainder = combined[FRAME_SIZE:]
                        inbound_samples_24k.clear()
                        if len(remainder) > 0:
                            inbound_samples_24k.append(remainder)

                        await worker_client.send_audio_frame(frame_to_send)

            # Outbound worker loop
            async def worker_recv_loop():
                async for msg in worker_client.messages():
                    if msg.type == MessageType.AUDIO:
                        # Resample 24 kHz -> 48 kHz for WebRTC browser track
                        audio_24k = np.frombuffer(msg.data, dtype=np.float32)
                        resampled_48k = resample_oneshot(audio_24k, in_rate=24000, out_rate=48000)
                        audio_track.push_audio(resampled_48k)
                    elif msg.type == MessageType.TEXT:
                        send_dc({"type": "transcript", "role": "assistant", "text": msg.text})

            feeder_task = asyncio.create_task(feeder_loop())
            worker_task = asyncio.create_task(worker_recv_loop())

            await asyncio.gather(feeder_task, worker_task, return_exceptions=True)

        except Exception as e:
            logger.error(f"[{session_id}] WebRTC worker bridge error: {e}")
            send_dc({"type": "error", "message": str(e)})
        finally:
            if lease:
                await worker_pool.release_worker(lease)
            logger.info(f"[{session_id}] WebRTC worker bridge closed.")

    inbound_audio_queue = asyncio.Queue(maxsize=100)

    @pc.on("track")
    def on_track(track):
        if track.kind == "audio":
            logger.info(f"[{session_id}] Browser audio track received.")

            async def process_inbound():
                try:
                    while pc.connectionState not in ("failed", "closed"):
                        frame = await track.recv()
                        ndarray = frame.to_ndarray()
                        if ndarray.dtype == np.int16:
                            f32 = ndarray.astype(np.float32) / 32768.0
                        else:
                            f32 = ndarray.astype(np.float32)
                        if f32.ndim > 1:
                            f32 = f32.mean(axis=0)

                        # Resample incoming browser audio (usually 48 kHz) to 24 kHz
                        in_rate = frame.sample_rate or 48000
                        resampled_24k = resample_oneshot(f32, in_rate=in_rate, out_rate=24000)
                        try:
                            inbound_audio_queue.put_nowait(resampled_24k)
                        except asyncio.QueueFull:
                            pass
                except (MediaStreamError, asyncio.CancelledError):
                    pass
                except Exception as e:
                    logger.debug(f"[{session_id}] Inbound audio error: {e}")

            asyncio.create_task(process_inbound())

    @pc.on("iceconnectionstatechange")
    def on_ice():
        logger.info(f"[{session_id}] ICE Connection State: {pc.iceConnectionState}")
        if pc.iceConnectionState in ("failed", "closed"):
            asyncio.create_task(pc.close())

    @pc.on("connectionstatechange")
    def on_conn():
        logger.info(f"[{session_id}] Connection State: {pc.connectionState}")
        if pc.connectionState in ("failed", "closed"):
            ACTIVE_PEER_CONNECTIONS.pop(session_id, None)

    # Set Remote Offer and Create Local Answer
    offer = RTCSessionDescription(sdp=req.sdp, type=req.type)
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    # Launch background bridge
    asyncio.create_task(run_worker_bridge(inbound_audio_queue))

    return WebRTCAnswerResponse(
        sdp=pc.localDescription.sdp,
        type=pc.localDescription.type,
        session_id=session_id,
    )
