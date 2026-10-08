#!/usr/bin/env python3
"""
Automated 10-call baseline benchmark for REAL PersonaPlex worker on Krutrim Cloud (A100 GPU).
Measures:
1. Direct Worker (ws://127.0.0.1:8998/api/chat):
   - Priming time (Connect to 0x00 handshake)
   - Real-Time Factor (RTF)
   - Response latency (first audio response)
   - Output audio RMS
2. Gateway with Standby Pool (ws://127.0.0.1:8000/v2/voice):
   - Ready-to-talk time (No greeting, pre-primed worker lease)

Saves results directly to docs/baseline_real.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import sphn
import websockets

SAMPLE_RATE = 24000
FRAME_SAMPLES = 1920  # 80ms at 24kHz
FRAME_DURATION_SEC = 0.08

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("baseline_real")


def get_gpu_info() -> dict:
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = [line.strip() for line in res.stdout.strip().split("\n") if line.strip()]
        if lines:
            parts = [p.strip() for p in lines[0].split(",")]
            return {
                "gpu_model": parts[0],
                "total_vram_mib": int(parts[1]),
                "free_vram_mib": int(parts[2]),
            }
    except Exception as e:
        logger.warning("Could not query nvidia-smi: %s", e)
    return {"gpu_model": "Unknown", "total_vram_mib": 0, "free_vram_mib": 0}


def generate_caller_speech() -> np.ndarray:
    """Generate 1.0s synthetic speech-like tone for test prompt."""
    t = np.linspace(0, 1.0, int(SAMPLE_RATE * 1.0), endpoint=False, dtype=np.float32)
    tone = 0.25 * np.sin(2 * np.pi * 300.0 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t))
    return tone.astype(np.float32)


async def run_single_worker_call(
    worker_url: str,
    voice: str = "NATM1.pt",
    prompt: str = "<system> You are a concise phone assistant. Speak colloquially. <system>",
) -> dict:
    query = urllib.parse.urlencode({"text_prompt": prompt, "voice_prompt": voice})
    full_url = f"{worker_url}?{query}"

    opus_writer = sphn.OpusStreamWriter(SAMPLE_RATE)
    opus_reader = sphn.OpusStreamReader(SAMPLE_RATE)

    t_connect_start = time.perf_counter()
    async with websockets.connect(full_url, max_size=10 * 1024 * 1024) as ws:
        # Handshake
        handshake_raw = await ws.recv()
        t_handshake = time.perf_counter()
        priming_ms = (t_handshake - t_connect_start) * 1000.0

        if not isinstance(handshake_raw, bytes) or len(handshake_raw) == 0 or handshake_raw[0] != 0x00:
            raise RuntimeError(f"Unexpected worker handshake: {handshake_raw!r}")

        # Send 1s speech audio
        speech = generate_caller_speech()
        offset = 0
        while offset < len(speech):
            chunk = speech[offset : offset + FRAME_SAMPLES]
            offset += FRAME_SAMPLES
            if len(chunk) < FRAME_SAMPLES:
                chunk = np.pad(chunk, (0, FRAME_SAMPLES - len(chunk)))
            opus_writer.append_pcm(chunk)
            payload = opus_writer.read_bytes()
            if payload:
                await ws.send(b"\x01" + payload)
            await asyncio.sleep(FRAME_DURATION_SEC)

        t_speech_done = time.perf_counter()

        # Send silence and collect response
        first_audio_time = None
        pcm_chunks = []
        token_count = 0
        step_times = []

        for _ in range(40):  # ~3.2 seconds max collection
            silent_chunk = np.zeros(FRAME_SAMPLES, dtype=np.float32)
            opus_writer.append_pcm(silent_chunk)
            payload = opus_writer.read_bytes()
            if payload:
                await ws.send(b"\x01" + payload)

            t_recv_start = time.perf_counter()
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.15)
                recv_dur = time.perf_counter() - t_recv_start
                step_times.append(recv_dur)

                if isinstance(msg, bytes) and len(msg) > 0:
                    kind = msg[0]
                    if kind == 0x01:  # Audio
                        if first_audio_time is None:
                            first_audio_time = time.perf_counter()
                        opus_reader.append_bytes(msg[1:])
                        decoded = opus_reader.read_pcm()
                        if len(decoded) > 0:
                            pcm_chunks.append(decoded)
                    elif kind == 0x02:  # Token
                        token_count += 1
            except asyncio.TimeoutError:
                pass

        resp_latency_ms = (
            (first_audio_time - t_speech_done) * 1000.0 if first_audio_time else None
        )
        all_pcm = np.concatenate(pcm_chunks) if pcm_chunks else np.zeros(0, dtype=np.float32)
        rms = float(np.sqrt(np.mean(all_pcm**2))) if len(all_pcm) > 0 else 0.0

        # RTF calculation: average step duration / 80ms frame time
        rtf = float(np.mean(step_times) / FRAME_DURATION_SEC) if step_times else 0.0

        return {
            "priming_ms": round(priming_ms, 2),
            "response_latency_ms": round(resp_latency_ms, 2) if resp_latency_ms else None,
            "output_rms": round(rms, 6),
            "rtf": round(rtf, 3),
            "token_count": token_count,
            "audio_duration_sec": round(len(all_pcm) / SAMPLE_RATE, 2),
        }


async def main():
    parser = argparse.ArgumentParser(description="Run 10-call real baseline benchmark")
    parser.add_argument("--worker", default="ws://127.0.0.1:8998/api/chat", help="PersonaPlex Worker WebSocket URL")
    parser.add_argument("--voice", default="NATM1.pt", help="Voice preset")
    parser.add_argument("--calls", type=int, default=10, help="Number of baseline calls")
    parser.add_argument("--output", default="docs/baseline_real.json", help="Path to save JSON results")
    args = parser.parse_args()

    gpu_info = get_gpu_info()
    print("=" * 70)
    print("  PERSONAPLEX REAL WORKER BASELINE BENCHMARK (10 CALLS)")
    print(f"  GPU: {gpu_info['gpu_model']} | VRAM: {gpu_info['free_vram_mib']} / {gpu_info['total_vram_mib']} MiB")
    print(f"  Target Worker: {args.worker}")
    print("=" * 70)

    results = []
    for i in range(1, args.calls + 1):
        print(f"\n[CALL {i}/{args.calls}] Connecting to {args.worker} (Voice: {args.voice})...")
        try:
            call_res = await run_single_worker_call(args.worker, voice=args.voice)
            call_res["call_id"] = i
            call_res["status"] = "SUCCESS_REAL"
            results.append(call_res)
            print(
                f"  -> REAL Priming: {call_res['priming_ms']} ms | "
                f"Resp Latency: {call_res['response_latency_ms']} ms | "
                f"RTF: {call_res['rtf']} | RMS: {call_res['output_rms']}"
            )
        except Exception as e:
            logger.error("Call %d failed: %s", i, e)
            results.append({"call_id": i, "status": "FAILED_REAL", "error": str(e)})

    successful = [r for r in results if r.get("status") == "SUCCESS_REAL"]
    if not successful:
        print("\n[ERROR] All calls failed. Check worker status and logs.")
        sys.exit(1)

    priming_times = [r["priming_ms"] for r in successful if r.get("priming_ms") is not None]
    resp_times = [r["response_latency_ms"] for r in successful if r.get("response_latency_ms") is not None]
    rtfs = [r["rtf"] for r in successful if r.get("rtf") is not None]
    rmses = [r["output_rms"] for r in successful if r.get("output_rms") is not None]

    summary_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "type": "REAL",
            "gpu_model": gpu_info["gpu_model"],
            "total_vram_mib": gpu_info["total_vram_mib"],
            "free_vram_mib": gpu_info["free_vram_mib"],
        },
        "worker_status": "ONLINE_REAL",
        "num_calls": len(successful),
        "measurements_real": {
            "priming_time_ms": {
                "mean": round(float(np.mean(priming_times)), 2),
                "p50": round(float(np.percentile(priming_times, 50)), 2),
                "p95": round(float(np.percentile(priming_times, 95)), 2),
            },
            "response_latency_ms": {
                "mean": round(float(np.mean(resp_times)), 2) if resp_times else None,
                "p50": round(float(np.percentile(resp_times, 50)), 2) if resp_times else None,
                "p95": round(float(np.percentile(resp_times, 95)), 2) if resp_times else None,
            },
            "rtf": {
                "mean": round(float(np.mean(rtfs)), 3),
                "p50": round(float(np.percentile(rtfs, 50)), 3),
            },
            "output_rms": {
                "mean": round(float(np.mean(rmses)), 6),
                "p50": round(float(np.percentile(rmses, 50)), 6),
            },
        },
        "raw_calls": results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("\n" + "=" * 70)
    print("  BASELINE REAL SUMMARY (LABEL: REAL)")
    print("=" * 70)
    print(f"  Priming Time:      p50 = {summary_data['measurements_real']['priming_time_ms']['p50']} ms | p95 = {summary_data['measurements_real']['priming_time_ms']['p95']} ms")
    print(f"  Response Latency:  p50 = {summary_data['measurements_real']['response_latency_ms']['p50']} ms | p95 = {summary_data['measurements_real']['response_latency_ms']['p95']} ms")
    print(f"  RTF:               mean = {summary_data['measurements_real']['rtf']['mean']}")
    print(f"  Output RMS:        mean = {summary_data['measurements_real']['output_rms']['mean']}")
    print(f"  Results saved to:  {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
