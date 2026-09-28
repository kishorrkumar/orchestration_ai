"""
CLI Interface for NVIDIA PersonaPlex Orchestration Layer.

Commands:
- run-gateway: Run the orchestration gateway server.
- run-mock-worker: Run a standalone high-fidelity PersonaPlex mock worker.
- test-call: Execute a full-duplex voice test call from audio file to agent.
"""

from __future__ import annotations
import argparse
import asyncio
import json
import logging
import sys
import time
from typing import Optional

import numpy as np
import soundfile as sf
import uvicorn
from websockets.asyncio.client import connect as ws_connect

from ..gateway.app import create_app
from ..persona.registry import default_registry
from ..protocol.audio import (
    FRAME_SIZE,
    SAMPLE_RATE,
    float32_to_int16,
    int16_to_float32,
)
from ..protocol.messages import (
    MessageType,
    AudioMessage,
    TextMessage,
    decode_message,
    encode_message,
)
from ..worker.mock_worker import PersonaPlexMockServer
from ..worker.pool import WorkerPool, WorkerNodeConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("orchestration.cli")


async def _run_gateway_cmd(args):
    pool = WorkerPool()
    mock_servers = []

    # If mock workers requested, spin them up on background ports
    if args.mock_workers > 0:
        base_port = args.mock_port_start
        for i in range(args.mock_workers):
            m_port = base_port + i
            m_id = f"mock-worker-{i}"
            server = PersonaPlexMockServer(host="127.0.0.1", port=m_port)
            await server.start()
            mock_servers.append(server)
            pool.register_worker(WorkerNodeConfig(id=m_id, host="127.0.0.1", port=m_port))
            logger.info(f"Attached mock worker {m_id} on port {m_port}")

    # Register any explicit real GPU workers
    if args.worker:
        for w_str in args.worker:
            # format: id:host:port or id:host:port:gpu_id
            parts = w_str.split(":")
            wid = parts[0]
            host = parts[1]
            port = int(parts[2])
            gpu = int(parts[3]) if len(parts) > 3 else None
            pool.register_worker(WorkerNodeConfig(id=wid, host=host, port=port, gpu_id=gpu))
            logger.info(f"Registered real worker node {wid} ({host}:{port})")

    app = create_app(pool=pool)
    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
    uv_server = uvicorn.Server(config)

    logger.info(f"Starting Orchestration Gateway on http://{args.host}:{args.port}")
    logger.info(f"Developer Console available at http://{args.host}:{args.port}/console")

    try:
        await uv_server.serve()
    finally:
        for s in mock_servers:
            await s.stop()


async def _run_mock_worker_cmd(args):
    server = PersonaPlexMockServer(host=args.host, port=args.port)
    await server.start()
    logger.info(f"PersonaPlex Mock Worker running on ws://{args.host}:{args.port}/api/chat")
    try:
        while True:
            await asyncio.sleep(3600)
    except (asyncio.CancelledError, KeyboardInterrupt):
        await server.stop()


async def _test_call_cmd(args):
    url = f"{args.url}?persona_id={args.persona}"
    logger.info(f"Initiating test call to {url} (Persona: {args.persona})")

    # Prepare input audio
    if args.input_wav:
        logger.info(f"Loading input audio from {args.input_wav}")
        data, sr = sf.read(args.input_wav, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)  # Convert to mono
        if sr != SAMPLE_RATE:
            # Simple linear resample if rate differs
            num_samples = int(len(data) * SAMPLE_RATE / sr)
            data = np.interp(
                np.linspace(0, len(data), num_samples, endpoint=False),
                np.arange(len(data)),
                data,
            ).astype(np.float32)
    else:
        logger.info("No input WAV specified; synthesizing 2.0-second 440Hz test audio tone at 24kHz")
        t = np.linspace(0, 2.0, 2 * SAMPLE_RATE, endpoint=False)
        data = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    # Slice into 1920-sample frames (80ms each)
    num_frames = len(data) // FRAME_SIZE
    frames = [data[i * FRAME_SIZE : (i + 1) * FRAME_SIZE] for i in range(num_frames)]
    logger.info(f"Prepared {len(frames)} audio frames ({len(frames) * 0.08:.2f} seconds)")

    agent_audio_chunks = []
    agent_text_tokens = []
    received_handshake = asyncio.Event()

    t_start = time.time()

    async with ws_connect(url) as ws:
        async def receiver():
            try:
                async for raw in ws:
                    if not isinstance(raw, bytes):
                        continue
                    msg = decode_message(raw)
                    if msg.type == MessageType.HANDSHAKE:
                        logger.info("Received Handshake [0x00] from Orchestration Gateway")
                        received_handshake.set()
                    elif msg.type == MessageType.METADATA:
                        logger.info(f"Received Metadata: {msg.data}")
                    elif msg.type == MessageType.AUDIO:
                        samples = np.frombuffer(msg.data, dtype=np.float32)
                        agent_audio_chunks.append(samples)
                    elif msg.type == MessageType.TEXT:
                        agent_text_tokens.append(msg.text)
                        sys.stdout.write(msg.text)
                        sys.stdout.flush()
                    elif msg.type == MessageType.ERROR:
                        logger.error(f"Received Error: {msg.error}")
                        break
            except Exception as e:
                logger.debug(f"Receiver closed: {e}")

        recv_task = asyncio.create_task(receiver())

        # Wait for handshake
        try:
            await asyncio.wait_for(received_handshake.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.error("Timed out waiting for handshake from gateway")
            return

        logger.info("\nStreaming user frames at 12.5 Hz (80ms cadence)...")
        print("Agent Transcript: ", end="", flush=True)

        # Stream user frames at 80ms cadence
        for frame in frames:
            t0 = time.time()
            msg_bytes = encode_message(AudioMessage(data=frame.tobytes()))
            await ws.send(msg_bytes)
            elapsed = time.time() - t0
            await asyncio.sleep(max(0.001, 0.08 - elapsed))

        # Allow extra time to catch trailing response tokens
        await asyncio.sleep(1.0)
        await ws.close()
        recv_task.cancel()
        try:
            await recv_task
        except asyncio.CancelledError:
            pass

    print("\n")
    duration = time.time() - t_start

    # Save output audio
    if agent_audio_chunks:
        full_audio = np.concatenate(agent_audio_chunks)
        sf.write(args.output_wav, full_audio, SAMPLE_RATE)
        logger.info(f"Saved {len(full_audio)/SAMPLE_RATE:.2f}s of agent audio to {args.output_wav}")

    # Save output transcript
    transcript_text = "".join(agent_text_tokens)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"transcript": transcript_text, "tokens": agent_text_tokens, "duration_sec": duration}, f, indent=2)
    logger.info(f"Saved transcript to {args.output_json}")

    print("\n" + "=" * 55)
    print("CALL SESSION SUMMARY")
    print(f"Total Duration:     {duration:.2f} s")
    print(f"User Frames Sent:   {len(frames)}")
    print(f"Agent Frames Recv:  {len(agent_audio_chunks)}")
    print(f"Text Tokens Recv:   {len(agent_text_tokens)}")
    print(f"Transcript:         {transcript_text.strip()}")
    print("=" * 55 + "\n")


def main():
    parser = argparse.ArgumentParser(description="PersonaPlex Orchestration CLI")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # run-gateway
    gw = subparsers.add_parser("run-gateway", help="Run the orchestration gateway")
    gw.add_argument("--host", default="127.0.0.1", help="Gateway bind host")
    gw.add_argument("--port", type=int, default=8000, help="Gateway bind port")
    gw.add_argument("--mock-workers", type=int, default=1, help="Number of local mock workers to auto-spawn")
    gw.add_argument("--mock-port-start", type=int, default=8998, help="Starting port for mock workers")
    gw.add_argument("--worker", action="append", help="Register real worker (id:host:port[:gpu_id])")

    # run-local (convenience command for instant local testing)
    local_p = subparsers.add_parser("run-local", help="One-command local run: gateway + 2 local workers on port 8000")
    local_p.add_argument("--host", default="127.0.0.1", help="Gateway bind host")
    local_p.add_argument("--port", type=int, default=8000, help="Gateway bind port")
    local_p.add_argument("--mock-workers", type=int, default=2, help="Number of local workers")
    local_p.add_argument("--mock-port-start", type=int, default=8998, help="Starting port for workers")

    # run-mock-worker
    mock = subparsers.add_parser("run-mock-worker", help="Run standalone PersonaPlex mock worker")
    mock.add_argument("--host", default="127.0.0.1")
    mock.add_argument("--port", type=int, default=8998)

    # test-call
    call = subparsers.add_parser("test-call", help="Run a test full-duplex voice call")
    call.add_argument("--url", default="ws://127.0.0.1:8000/v1/realtime", help="Gateway WebSocket URL")
    call.add_argument("--persona", default="wise_teacher", help="Persona ID")
    call.add_argument("--input-wav", default=None, help="Input WAV file (optional)")
    call.add_argument("--output-wav", default="output_agent.wav", help="Path to save agent output WAV")
    call.add_argument("--output-json", default="output_transcript.json", help="Path to save transcript JSON")

    args = parser.parse_args()

    if args.subcommand in ("run-gateway", "run-local"):
        if not hasattr(args, "worker"):
            args.worker = None
        asyncio.run(_run_gateway_cmd(args))
    elif args.subcommand == "run-mock-worker":
        asyncio.run(_run_mock_worker_cmd(args))
    elif args.subcommand == "test-call":
        asyncio.run(_test_call_cmd(args))


if __name__ == "__main__":
    main()
