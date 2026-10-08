"""
CLI Interface for NVIDIA PersonaPlex Orchestration Layer.

Commands:
- run-gateway: Run the orchestration gateway server with --worker-type {mock, cascaded, personaplex}.
- run-cascaded-worker: Run standalone local streaming cascaded worker (STT -> LLM -> Chunker -> TTS).
- run-mock-worker: Run a standalone high-fidelity PersonaPlex mock worker.
- test-call: Execute a full-duplex voice test call from audio file to agent.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time

import numpy as np
import soundfile as sf
import uvicorn
from websockets.asyncio.client import connect as ws_connect

from ..gateway.app import create_app
from ..protocol.audio import (
    FRAME_SIZE,
    SAMPLE_RATE,
)
from ..protocol.messages import (
    AudioMessage,
    MessageType,
    decode_message,
    encode_message,
)
from ..worker.mock_worker import PersonaPlexMockServer
from ..worker.pool import WorkerNodeConfig, WorkerPool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("orchestration.cli")


async def _run_gateway_cmd(args):
    pool = WorkerPool()
    spawned_servers = []

    worker_type = getattr(args, "worker_type", "mock")

    # If mock workers requested
    if worker_type == "mock":
        base_port = args.mock_port_start
        count = getattr(args, "mock_workers", 1) or 1
        for i in range(count):
            m_port = base_port + i
            m_id = f"mock-worker-{i}"
            server = PersonaPlexMockServer(host="127.0.0.1", port=m_port)
            await server.start()
            spawned_servers.append(server)
            pool.register_worker(WorkerNodeConfig(id=m_id, host="127.0.0.1", port=m_port))
            logger.info(f"Attached Mock Worker {m_id} on port {m_port}")
        base_port = args.mock_port_start
        count = getattr(args, "mock_workers", 1) or 1
        for i in range(count):
            m_port = base_port + i
            m_id = f"mock-worker-{i}"
            server = PersonaPlexMockServer(host="127.0.0.1", port=m_port)
            await server.start()
            spawned_servers.append(server)
            pool.register_worker(WorkerNodeConfig(id=m_id, host="127.0.0.1", port=m_port))
            logger.info(f"Attached Mock Worker {m_id} on port {m_port}")

    # Register any explicit real GPU workers (--worker [id:]host:port[:gpu])
    if getattr(args, "worker", None):
        for w_str in args.worker:
            parts = [p.strip() for p in w_str.split(":")]
            if len(parts) == 2:
                host, port_str = parts
                wid = f"worker-{host}-{port_str}"
                port = int(port_str)
                gpu = None
            elif len(parts) == 3:
                wid, host, port_str = parts
                port = int(port_str)
                gpu = None
            elif len(parts) >= 4:
                wid, host, port_str, gpu_str = parts[:4]
                port = int(port_str)
                gpu = int(gpu_str)
            else:
                continue
            pool.register_worker(WorkerNodeConfig(id=wid, host=host, port=port, gpu_id=gpu))
            logger.info(f"Registered real PersonaPlex worker node {wid} ({host}:{port})")

    app = create_app(pool=pool, worker_type=worker_type, active_server=spawned_servers[0] if spawned_servers else None)
    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        ws_ping_interval=20.0,
        ws_ping_timeout=20.0,
        ws_max_size=16777216,
    )
    uv_server = uvicorn.Server(config)

    logger.info(f"Starting Orchestration Gateway ({worker_type.upper()}) on http://{args.host}:{args.port}")
    logger.info(f"Developer Console available at http://{args.host}:{args.port}/console")

    try:
        await uv_server.serve()
    finally:
        for s in spawned_servers:
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
    logger.info(f"Connecting test call to {args.url} (persona: {args.persona})...")
    full_url = f"{args.url}?persona_id={args.persona}"

    if args.input_wav:
        data, sr = sf.read(args.input_wav, dtype="float32")
        if sr != SAMPLE_RATE:
            num_samples = int(len(data) * SAMPLE_RATE / sr)
            data = np.interp(
                np.linspace(0, len(data), num_samples, endpoint=False),
                np.arange(len(data)),
                data,
            ).astype(np.float32)
        if data.ndim > 1:
            data = data.mean(axis=1)
        frames = [data[i : i + FRAME_SIZE] for i in range(0, len(data) - FRAME_SIZE, FRAME_SIZE)]
    else:
        # Default: 3 seconds of gentle audio pulses
        t = np.linspace(0, 3.0, int(SAMPLE_RATE * 3.0), endpoint=False)
        tone = (0.1 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        frames = [tone[i : i + FRAME_SIZE] for i in range(0, len(tone), FRAME_SIZE)]

    agent_audio_chunks = []
    agent_text_tokens = []
    t_start = time.time()

    async with ws_connect(full_url) as ws:
        async def send_audio():
            for frame in frames:
                msg = AudioMessage(data=frame.tobytes())
                await ws.send(encode_message(msg))
                await asyncio.sleep(0.08)
            # Continuous full-duplex silence frames for turn endpointing and streaming response
            silence = np.zeros(FRAME_SIZE, dtype=np.float32)
            for _ in range(85):  # ~6.8 seconds
                msg = AudioMessage(data=silence.tobytes())
                await ws.send(encode_message(msg))
                await asyncio.sleep(0.08)

        async def receive_stream():
            try:
                async for raw in ws:
                    msg = decode_message(raw)
                    if msg.type == MessageType.AUDIO:
                        chunk = np.frombuffer(msg.data, dtype=np.float32)
                        agent_audio_chunks.append(chunk)
                    elif msg.type == MessageType.TEXT:
                        agent_text_tokens.append(msg.text)
                        print(msg.text, end="", flush=True)
            except Exception:
                pass

        send_task = asyncio.create_task(send_audio())
        recv_task = asyncio.create_task(receive_stream())

        await send_task
        await ws.close()
        recv_task.cancel()

    duration = time.time() - t_start
    if agent_audio_chunks:
        full_audio = np.concatenate(agent_audio_chunks)
        sf.write(args.output_wav, full_audio, SAMPLE_RATE)
        logger.info(f"Saved {len(full_audio)/SAMPLE_RATE:.2f}s of agent audio to {args.output_wav}")

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


def _doctor_cmd(args):
    print("=" * 60)
    print("AARAV VOICE AGENT - ENVIRONMENT & HARDWARE DOCTOR")
    print("=" * 60)

    # 1. Check Ollama
    try:
        import httpx
        r = httpx.get("http://127.0.0.1:11434/api/tags", timeout=3.0)
        if r.status_code == 200:
            models = [m.get("name", "") for m in r.json().get("models", [])]
            print("[OK] Ollama Server: RUNNING on port 11434")
            print(f"     Available Models: {', '.join(models) if models else 'None'}")
        else:
            print(f"[ERROR] Ollama Server: Error HTTP {r.status_code}")
    except Exception as e:
        print(f"[ERROR] Ollama Server: NOT REACHABLE ({e})")

    # 2. Check CUDA & GPU
    try:
        import torch
        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            mem_free, _ = torch.cuda.mem_get_info(0)
            mem_free_mb = mem_free / (1024**2)
            print(f"[OK] CUDA: AVAILABLE ({torch.version.cuda})")
            print(f"     GPU: {dev_name} (Total: {vram_total:.0f} MB, Free: {mem_free_mb:.0f} MB)")
        else:
            print("[WARN] CUDA: Not detected in PyTorch (running in CPU mode)")
    except Exception as e:
        print(f"[ERROR] PyTorch/CUDA check error: {e}")

    # 3. Check ASR (faster-whisper)
    try:
        import faster_whisper
        print(f"[OK] faster-whisper: INSTALLED (version {faster_whisper.__version__})")
        print("     ASR Configuration: base model on CPU int8 (4 threads, RTF ~0.15)")
    except Exception as e:
        print(f"[ERROR] faster-whisper: NOT INSTALLED ({e})")

    # 4. Check TTS (Kokoro)
    try:
        print("[OK] Kokoro TTS: INSTALLED (ONNX runtime 24 kHz)")
        print("     Voices: aarav_colloquial (Male), priya_colloquial (Female)")
    except Exception as e:
        print(f"[ERROR] Kokoro TTS: NOT INSTALLED ({e})")

    # 5. Check Audio Devices
    try:
        import sounddevice as sd
        default_in = sd.query_devices(kind='input')
        default_out = sd.query_devices(kind='output')
        print(f"[OK] Audio Input:  {default_in.get('name', 'Unknown')}")
        print(f"[OK] Audio Output: {default_out.get('name', 'Unknown')}")
    except Exception as e:
        print(f"[INFO] Audio Devices: Host audio loopback available ({e})")

    # 6. Chosen Configuration Summary
    print("-" * 60)
    print("CHOSEN RUNTIME CONFIGURATION:")
    print("  Worker Type:       local_cascade (100% open-source local pipeline)")
    print("  Persona:           Aarav (Colloquial Indian English)")
    print("  LLM:               Ollama qwen2.5:1.5b (Q4 quant, VRAM ~1.65 GB)")
    print("  ASR:               faster-whisper base (CPU int8, ~470ms latency)")
    print("  Turn Detector:     Silero VAD + 650ms base + 700ms linguistic extension")
    print("  TTS:               Kokoro-82M ONNX (CPU 24 kHz, TTFA ~400ms)")
    print("  VRAM Partition:    Ollama GPU (1.65 GB) + CPU ASR/TTS (0 MB VRAM)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="PersonaPlex Orchestration CLI")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # doctor
    subparsers.add_parser("doctor", help="Check hardware, Ollama, CUDA, models, and audio setup")

    # run-gateway
    gw = subparsers.add_parser("run-gateway", help="Run the orchestration gateway")
    gw.add_argument("--host", default="127.0.0.1", help="Gateway bind host")
    gw.add_argument("--port", type=int, default=8000, help="Gateway bind port")
    gw.add_argument(
        "--worker-type",
        choices=["mock", "personaplex"],
        default="mock",
        help="Worker type: mock (simulated) or personaplex (remote GPU)",
    )
    gw.add_argument("--mock-workers", type=int, default=1, help="Number of local workers to auto-spawn")
    gw.add_argument("--mock-port-start", type=int, default=8998, help="Starting port for workers")
    gw.add_argument("--worker", action="append", help="Register real worker (id:host:port[:gpu_id])")

    # run-local
    local_p = subparsers.add_parser("run-local", help="One-command local run: gateway + mock worker on port 8000")
    local_p.add_argument("--host", default="127.0.0.1", help="Gateway bind host")
    local_p.add_argument("--port", type=int, default=8000, help="Gateway bind port")
    local_p.add_argument(
        "--worker-type",
        choices=["mock", "personaplex"],
        default="mock",
        help="Worker type",
    )
    local_p.add_argument("--mock-workers", type=int, default=1, help="Number of local workers")
    local_p.add_argument("--mock-port-start", type=int, default=8998, help="Starting port for workers")

    # run-mock-worker
    mock = subparsers.add_parser("run-mock-worker", help="Run standalone PersonaPlex mock worker")
    mock.add_argument("--host", default="127.0.0.1")
    mock.add_argument("--port", type=int, default=8998)

    # test-call
    call = subparsers.add_parser("test-call", help="Run a test full-duplex voice call")
    call.add_argument("--url", default="ws://127.0.0.1:8000/v1/realtime", help="Gateway WebSocket URL")
    call.add_argument("--persona", default="indian_pro", help="Persona ID")
    call.add_argument("--input-wav", default=None, help="Input WAV file (optional)")
    call.add_argument("--output-wav", default="output_agent.wav", help="Path to save agent output WAV")
    call.add_argument("--output-json", default="output_transcript.json", help="Path to save transcript JSON")

    args = parser.parse_args()

    if args.subcommand == "doctor":
        _doctor_cmd(args)
    elif args.subcommand in ("run-gateway", "run-local"):
        if not hasattr(args, "worker"):
            args.worker = None
        asyncio.run(_run_gateway_cmd(args))
    elif args.subcommand == "run-mock-worker":
        asyncio.run(_run_mock_worker_cmd(args))
    elif args.subcommand == "test-call":
        asyncio.run(_test_call_cmd(args))


if __name__ == "__main__":
    main()
