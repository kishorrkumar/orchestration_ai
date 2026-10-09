"""
Direct Voice Enrollment & Live S2S Streaming Controller.
Implements the target specification:
- POST /voice/enroll  (audio -> voice_id, QA metrics, persistent cache)
- GET /voice/{voice_id}
- DELETE /voice/{voice_id}
- GET /voice/{voice_id}/preview
- GET /voice
- WebSocket /agent/stream (live full-duplex bi-directional audio stream taking voice_id)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import time
import uuid
from typing import Any

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import FileResponse
import numpy as np

from ..audio.similarity import compute_speaker_similarity
from ..protocol.audio import MODEL_FRAME_SAMPLES, MODEL_SAMPLE_RATE, compute_rms
from ..protocol.messages import (
    AudioMessage,
    ControlAction,
    ControlMessage,
    MessageType,
    TextMessage,
    decode_message,
    encode_message,
)
from ..tts.voice_clone import (
    VoiceCloningValidationError,
    default_voice_cloner,
)
from ..worker.client import PersonaConfig, PersonaPlexWorkerClient
from ..worker.pool import WorkerPool

logger = logging.getLogger("orchestration.api.voice_stream")

router = APIRouter(tags=["Voice & Streaming"])


@router.post("/voice/enroll", status_code=status.HTTP_201_CREATED)
async def enroll_voice(
    file: UploadFile | None = File(None),
    audio: UploadFile | None = File(None),
    voice_name: str = Form("My Cloned Voice"),
    name: str | None = Form(None),
    consent: bool = Form(True),
    gender: str | None = Form(None),
) -> dict[str, Any]:
    """
    Enroll a voice from recorded microphone audio or uploaded reference (3-60 seconds).
    Validates quality, removes silence, normalizes loudness (-22 LUFS),
    computes speaker similarity QA metrics, and caches conditioning artifacts.
    """
    if not consent:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Voice cloning requires explicit user consent.",
        )

    upload = file or audio
    if upload is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No audio file provided in 'file' or 'audio' field.",
        )

    audio_bytes = await upload.read()
    if len(audio_bytes) < 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio sample too short or empty (minimum 1024 bytes required).",
        )

    resolved_name = (name or voice_name or "My Cloned Voice").strip()

    try:
        meta = default_voice_cloner.clone_voice(
            audio_bytes=audio_bytes,
            voice_name=resolved_name,
            consent=consent,
            consent_statement="I confirm that this is my own voice recording for AI conversational speech.",
            preferred_gender=gender,
        )
    except VoiceCloningValidationError as v_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(v_err))
    except Exception as err:
        logger.error(f"Error enrolling voice: {err}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process and enroll voice reference: {err}",
        )

    voice_id = meta["id"]
    wav_id = f"{voice_id}.wav"

    return {
        "voice_id": wav_id,
        "id": voice_id,
        "name": meta.get("name", resolved_name),
        "sample_rate": meta.get("sample_rate", 24000),
        "duration_sec": meta.get("duration_sec"),
        "qa_passed": meta.get("qa_passed", True),
        "qa_score": meta.get("qa_score", 1.0),
        "metrics": meta.get("metrics", {}),
        "preview_url": f"/voice/{voice_id}/preview",
        "recommended_engine": meta.get("recommended_engine", "personaplex_s2s"),
        "status": "ready",
    }


@router.get("/voice")
async def list_enrolled_voices() -> list[dict[str, Any]]:
    """List all registered cloned voice profiles with status and metrics."""
    voices = default_voice_cloner.list_cloned_voices()
    return [
        {
            "voice_id": f"{v['id']}.wav",
            "id": v["id"],
            "name": v.get("name", v["id"]),
            "duration_sec": v.get("duration_sec"),
            "qa_passed": v.get("qa_passed", True),
            "qa_score": v.get("qa_score", 1.0),
            "preview_url": f"/voice/{v['id']}/preview",
            "status": "ready",
        }
        for v in voices
        if v.get("id") != "voice_prompt"
    ]


@router.get("/voice/{voice_id}")
async def get_enrolled_voice(voice_id: str) -> dict[str, Any]:
    """Retrieve detailed metadata and QA verification for an enrolled voice."""
    clean = voice_id.replace(".wav", "").replace(".pt", "").strip()
    meta = default_voice_cloner.get_voice_metadata(clean)
    if not meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Enrolled voice '{voice_id}' not found.",
        )
    return {
        "voice_id": f"{clean}.wav",
        "id": clean,
        "name": meta.get("name", clean),
        "sample_rate": meta.get("sample_rate", 24000),
        "duration_sec": meta.get("duration_sec"),
        "qa_passed": meta.get("qa_passed", True),
        "qa_score": meta.get("qa_score", 1.0),
        "metrics": meta.get("metrics", {}),
        "preview_url": f"/voice/{clean}/preview",
        "status": "ready",
    }


@router.delete("/voice/{voice_id}")
async def delete_enrolled_voice(voice_id: str) -> dict[str, Any]:
    """Permanently delete an enrolled voice profile and remove all disk artifacts."""
    clean = voice_id.replace(".wav", "").replace(".pt", "").strip()
    deleted = default_voice_cloner.delete_voice(clean)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Voice profile '{voice_id}' not found.",
        )
    return {"success": True, "id": clean, "voice_id": f"{clean}.wav"}


@router.get("/voice/{voice_id}/preview")
async def preview_enrolled_voice(voice_id: str):
    """Stream audio sample preview for an enrolled voice."""
    clean = voice_id.replace(".wav", "").replace(".pt", "").strip()
    v_path = default_voice_cloner.get_voice_path(clean)
    if v_path and v_path.exists() and v_path.suffix == ".wav":
        return FileResponse(
            str(v_path),
            media_type="audio/wav",
            filename=f"{clean}.wav",
        )

    # Search candidate directories
    candidate_dirs = [
        pathlib.Path("data") / "cloned_voices" / clean,
        pathlib.Path("voices"),
        pathlib.Path("/workspace/voices"),
        pathlib.Path.home() / ".cache" / "huggingface" / "voices",
    ]
    for cdir in candidate_dirs:
        wav_file = cdir / f"{clean}.wav"
        if wav_file.exists():
            return FileResponse(str(wav_file), media_type="audio/wav", filename=f"{clean}.wav")

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Preview audio for voice '{voice_id}' not found.",
    )


@router.websocket("/agent/stream")
async def agent_live_stream_endpoint(
    websocket: WebSocket,
    voice_id: str = Query("kkishorekumar.wav", description="Voice profile ID or preset"),
    text_prompt: str | None = Query(None, description="System persona instructions"),
    sample_rate: int = Query(16000, description="Client PCM sample rate (16000 or 24000)"),
    codec: str = Query("pcm16", description="Client audio codec ('pcm16')"),
):
    """
    Direct Full-Duplex S2S WebSocket endpoint.
    Streams input audio frames to the model as they arrive,
    decodes output audio frame by frame, and streams binary PCM chunks to the client.
    Supports instant barge-in and real-time per-turn latency instrumentation.
    """
    await websocket.accept()
    session_id = f"stream_{int(time.time())}_{uuid.uuid4().hex[:6]}"

    # Resolve voice profile
    clean_voice = voice_id.replace(".wav", "").replace(".pt", "").strip()
    matched_voice = f"{clean_voice}.wav"
    v_path = default_voice_cloner.get_voice_path(clean_voice)
    if v_path:
        matched_voice = v_path.name
    elif (pathlib.Path("voices") / f"{clean_voice}.wav").exists():
        matched_voice = f"{clean_voice}.wav"
    elif (pathlib.Path("voices") / f"{clean_voice}.pt").exists():
        matched_voice = f"{clean_voice}.pt"

    # Default natural human persona prompt if none provided
    prompt_text = text_prompt or (
        "You are a friendly, helpful conversational voice assistant. "
        "You speak naturally and casually in 1-2 short spoken sentences. "
        "Use contractions, sound warm and human, acknowledge what the user said, "
        "and never speak in lists, bullet points, or markdown."
    )

    worker_pool: WorkerPool = getattr(websocket.app.state, "pool", None)
    if not worker_pool:
        await websocket.send_json({"type": "error", "message": "Worker pool not available on gateway"})
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        return

    persona_cfg = PersonaConfig(
        id=session_id,
        name="Agent",
        voice_prompt=matched_voice,
        text_prompt=prompt_text,
        system_prompt=prompt_text,
        audio_temperature=0.8,
        text_temperature=0.7,
        top_k_audio=250,
        top_k_text=25,
    )

    t_start = time.perf_counter()
    worker_client: PersonaPlexWorkerClient | None = None
    try:
        worker_client = await worker_pool.acquire_worker(session_id=session_id, timeout=3.0)
        await worker_client.connect(session_id=session_id, persona=persona_cfg)
    except Exception as e:
        logger.error(f"[AgentStream {session_id}] Failed to acquire/connect worker: {e}")
        await websocket.send_json({"type": "error", "message": f"Worker unavailable: {e}"})
        if worker_client:
            await worker_pool.release_worker(worker_client.worker_id, success=False)
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        return

    ttfa_priming_ms = round((time.perf_counter() - t_start) * 1000.0, 1)

    # Handshake ready notification
    await websocket.send_json({
        "type": "status",
        "status": "ready",
        "session_id": session_id,
        "voice_id": matched_voice,
        "priming_ms": ttfa_priming_ms,
        "message": "Voice agent streaming connected and ready.",
    })

    stop_event = asyncio.Event()
    worker_queue: asyncio.Queue[bytes] = asyncio.Queue()
    audio_in_queue: asyncio.Queue[np.ndarray] = asyncio.Queue()

    # Latency tracking state
    turn_index = 0
    last_user_speech_end = 0.0
    agent_spoke_on_turn = False

    async def worker_receiver():
        try:
            async for raw in worker_client.recv_raw_frames():
                if stop_event.is_set():
                    break
                await worker_queue.put(raw)
        except asyncio.CancelledError:
            pass
        except Exception as err:
            logger.debug(f"[AgentStream {session_id}] worker_receiver note: {err}")

    async def client_reader():
        nonlocal last_user_speech_end, agent_spoke_on_turn
        carryover_bytes = bytearray()
        try:
            while not stop_event.is_set():
                msg = await websocket.receive()
                if msg.get("type") == "websocket.disconnect":
                    stop_event.set()
                    break

                if "bytes" in msg and msg["bytes"]:
                    data = msg["bytes"]
                    carryover_bytes.extend(data)

                    # Ingest PCM16 frames: 640 bytes = 320 samples at 16kHz (20ms)
                    # or 1920 bytes = 960 samples at 24kHz
                    bytes_per_sample = 2
                    target_chunk_bytes = int(sample_rate * 0.02 * bytes_per_sample)  # 20ms chunks

                    while len(carryover_bytes) >= target_chunk_bytes:
                        chunk = bytes(carryover_bytes[:target_chunk_bytes])
                        del carryover_bytes[:target_chunk_bytes]

                        int16 = np.frombuffer(chunk, dtype=np.int16)
                        f32 = int16.astype(np.float32) / 32768.0

                        # Resample to 24 kHz if client is 16 kHz
                        if sample_rate != 24000:
                            num_out = int(round(len(f32) * 24000 / sample_rate))
                            f32_24k = np.interp(
                                np.linspace(0, len(f32), num_out, endpoint=False),
                                np.arange(len(f32)),
                                f32,
                            ).astype(np.float32)
                        else:
                            f32_24k = f32

                        rms = compute_rms(f32_24k)
                        if rms > 0.008:
                            last_user_speech_end = time.time()
                            agent_spoke_on_turn = False

                        await audio_in_queue.put(f32_24k)

                elif "text" in msg and msg["text"]:
                    try:
                        cmd = json.loads(msg["text"])
                        ctype = cmd.get("type")
                        if ctype == "interrupt":
                            # Instant barge-in: flush pending outbound queues
                            while not worker_queue.empty():
                                try:
                                    worker_queue.get_nowait()
                                except asyncio.QueueEmpty:
                                    break
                            logger.info(f"[AgentStream {session_id}] Barge-in interrupt acknowledged")
                        elif ctype == "ping":
                            await websocket.send_json({"type": "pong", "time": time.time()})
                    except Exception:
                        pass
        except WebSocketDisconnect:
            stop_event.set()
        except asyncio.CancelledError:
            pass

    async def pacer_sender():
        """Paces continuous 80ms (1920 samples) frames to worker at 12.5 Hz."""
        frame_interval = 0.080
        next_deadline = time.monotonic()
        pcm_buffer: list[float] = []

        try:
            while not stop_event.is_set():
                next_deadline += frame_interval

                # Drain available input audio into pcm_buffer
                while not audio_in_queue.empty():
                    try:
                        arr = audio_in_queue.get_nowait()
                        pcm_buffer.extend(arr.tolist())
                    except asyncio.QueueEmpty:
                        break

                if len(pcm_buffer) >= MODEL_FRAME_SAMPLES:
                    frame_samples = np.array(pcm_buffer[:MODEL_FRAME_SAMPLES], dtype=np.float32)
                    del pcm_buffer[:MODEL_FRAME_SAMPLES]
                else:
                    if pcm_buffer:
                        pad_len = MODEL_FRAME_SAMPLES - len(pcm_buffer)
                        frame_samples = np.array(pcm_buffer + [0.0] * pad_len, dtype=np.float32)
                        pcm_buffer.clear()
                    else:
                        frame_samples = np.zeros(MODEL_FRAME_SAMPLES, dtype=np.float32)

                if worker_client and worker_client.is_connected:
                    await worker_client.send_audio(frame_samples)

                now = time.monotonic()
                sleep_sec = max(0.0, next_deadline - now)
                await asyncio.sleep(sleep_sec)
        except asyncio.CancelledError:
            pass

    async def worker_to_client_pacer():
        nonlocal turn_index, last_user_speech_end, agent_spoke_on_turn
        try:
            while not stop_event.is_set():
                try:
                    raw = await asyncio.wait_for(worker_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                if not raw:
                    continue

                opcode = raw[0]
                if opcode == 0x01:  # Audio
                    payload = raw[1:]
                    # Convert float32 or raw PCM to client sample rate & PCM16
                    if len(payload) >= 4:
                        if len(payload) % 4 == 0:
                            f32 = np.frombuffer(payload, dtype=np.float32)
                        else:
                            f32 = np.zeros(MODEL_FRAME_SAMPLES, dtype=np.float32)

                        # Check RMS to detect active speech
                        rms = compute_rms(f32)
                        if rms > 0.005 and not agent_spoke_on_turn and last_user_speech_end > 0:
                            agent_spoke_on_turn = True
                            turn_index += 1
                            ttfa_ms = round((time.time() - last_user_speech_end) * 1000.0, 1)
                            await websocket.send_json({
                                "type": "latency",
                                "turn": turn_index,
                                "ttfa_ms": ttfa_ms,
                                "timestamp": time.time(),
                            })

                        # Resample to client rate if needed
                        if sample_rate != 24000:
                            out_len = int(round(len(f32) * sample_rate / 24000))
                            out_f32 = np.interp(
                                np.linspace(0, len(f32), out_len, endpoint=False),
                                np.arange(len(f32)),
                                f32,
                            ).astype(np.float32)
                        else:
                            out_f32 = f32

                        int16 = (np.clip(out_f32, -0.98, 0.98) * 32767.0).astype(np.int16)
                        await websocket.send_bytes(int16.tobytes())

                elif opcode == 0x02:  # Text token
                    token = raw[1:].decode("utf-8", errors="replace").replace("▁", " ")
                    await websocket.send_json({
                        "type": "transcript",
                        "role": "assistant",
                        "text": token,
                    })

                elif opcode == 0x05:  # Error
                    err = raw[1:].decode("utf-8", errors="replace")
                    await websocket.send_json({"type": "error", "message": err})

        except WebSocketDisconnect:
            stop_event.set()
        except asyncio.CancelledError:
            pass

    receiver_task = asyncio.create_task(worker_receiver())
    client_task = asyncio.create_task(client_reader())
    pacer_task = asyncio.create_task(pacer_sender())
    outbound_task = asyncio.create_task(worker_to_client_pacer())

    done, pending = await asyncio.wait(
        [receiver_task, client_task, pacer_task, outbound_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    stop_event.set()
    for task in pending:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    if worker_client:
        await worker_pool.release_worker(worker_client.worker_id, success=True)
