"""
Comprehensive Benchmark Suite for PersonaPlex Local Cascaded Worker.
Benchmarks LLMs (Ollama), ASR (Faster-Whisper), and TTS (Kokoro).
Saves detailed results to docs/benchmarks.md and docs/decisions.md.
"""
from __future__ import annotations
import asyncio
import json
import os
import pathlib
import time
from typing import Dict, Any, List
import numpy as np
import soundfile as sf

BENCHMARK_PROMPTS = [
    "Hi Aarav, how are you doing today?",
    "Can you tell me about artificial intelligence?",
    "Tell me a short, funny joke about cats.",
    "How does UPI payment work in simple terms?",
    "What's the weather like in Bengaluru during monsoon?",
]

def check_vram() -> float:
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return round(info.used / (1024 * 1024), 1)
    except Exception:
        return 0.0

async def benchmark_ollama_model(model_name: str) -> Dict[str, Any]:
    import httpx
    print(f"\n--- Benchmarking LLM: {model_name} ---")
    ttft_list = []
    tps_list = []
    responses = []
    vram_start = check_vram()

    system_prompt = (
        "You are Aarav, an articulate and friendly Indian English voice assistant. "
        "Reply directly in 1-2 spoken sentences using natural Indian English idioms sparingly. "
        "No markdown, bullets, or emojis."
    )

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Warm up
        try:
            await client.post(
                "http://127.0.0.1:11434/api/chat",
                json={"model": model_name, "messages": [{"role": "user", "content": "hi"}], "stream": False, "keep_alive": "10m"}
            )
        except Exception as e:
            return {"model": model_name, "error": str(e)}

        vram_loaded = check_vram()

        for user_msg in BENCHMARK_PROMPTS:
            t0 = time.perf_counter()
            first_token_time = None
            token_count = 0
            accumulated = []

            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                "stream": True,
                "options": {"temperature": 0.7, "top_p": 0.9, "num_ctx": 2048},
            }

            resp = await client.post("http://127.0.0.1:11434/api/chat", json=payload)
            async for line in resp.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                token = data.get("message", {}).get("content", "")
                if token:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    token_count += 1
                    accumulated.append(token)
                if data.get("done", False):
                    break

            t_end = time.perf_counter()
            ttft = (first_token_time - t0) * 1000.0 if first_token_time else 0.0
            gen_time = t_end - (first_token_time or t0)
            tps = (token_count / gen_time) if gen_time > 0 else 0.0

            ttft_list.append(ttft)
            tps_list.append(tps)
            responses.append("".join(accumulated).strip())

    avg_ttft = np.mean(ttft_list)
    avg_tps = np.mean(tps_list)
    print(f"[{model_name}] Avg TTFT: {avg_ttft:.1f}ms | Avg TPS: {avg_tps:.1f} tok/s | VRAM: {vram_loaded} MB")

    return {
        "model": model_name,
        "avg_ttft_ms": round(float(avg_ttft), 1),
        "avg_tps": round(float(avg_tps), 1),
        "vram_mb": vram_loaded,
        "sample_reply": responses[1] if len(responses) > 1 else "",
        "joke_reply": responses[2] if len(responses) > 2 else "",
    }

def benchmark_faster_whisper(model_size: str, device: str, compute_type: str) -> Dict[str, Any]:
    print(f"\n--- Benchmarking ASR: faster-whisper ({model_size}, {device}, {compute_type}) ---")
    try:
        from faster_whisper import WhisperModel
        t_load = time.perf_counter()
        whisper = WhisperModel(model_size, device=device, compute_type=compute_type, cpu_threads=4)
        load_ms = (time.perf_counter() - t_load) * 1000.0
        vram = check_vram() if device == "cuda" else 0.0

        # Create a synthetic 3-second speech-like waveform (16 kHz)
        sr = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        test_audio = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

        # Transcribe benchmark
        latencies = []
        for _ in range(3):
            t0 = time.perf_counter()
            segments, _ = whisper.transcribe(
                test_audio,
                language="en",
                beam_size=1,
                initial_prompt="artificial intelligence, machine learning, Aarav, Bengaluru",
            )
            _ = list(segments)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        avg_lat = np.mean(latencies)
        rtf = (avg_lat / 1000.0) / duration
        print(f"[{model_size}-{device}] Latency: {avg_lat:.1f}ms (RTF: {rtf:.3f}) | VRAM: {vram} MB")

        return {
            "model_size": model_size,
            "device": device,
            "compute_type": compute_type,
            "latency_ms": round(float(avg_lat), 1),
            "rtf": round(float(rtf), 3),
            "vram_mb": vram,
        }
    except Exception as e:
        print(f"[{model_size}-{device}] Error: {e}")
        return {"model_size": model_size, "device": device, "error": str(e)}

async def benchmark_tts() -> Dict[str, Any]:
    print("\n--- Benchmarking TTS: Kokoro-v1.0 (ONNX 24 kHz) ---")
    from orchestration.tts.base import KokoroTTSBackend
    tts = KokoroTTSBackend(sample_rate=24000)
    tts.warm_up()

    test_clauses = [
        "Namaste! This is Aarav.",
        "Sure thing, let me explain how artificial intelligence works.",
        "Why don't scientists trust atoms? Because they make up everything!",
    ]

    latencies = []
    for clause in test_clauses:
        t0 = time.perf_counter()
        audio = await tts.synthesize(clause, voice="aarav_colloquial")
        elapsed = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed)

    avg_ttfa = np.mean(latencies)
    print(f"[Kokoro-82M] Avg TTFA: {avg_ttfa:.1f}ms")
    return {
        "engine": "Kokoro-82M",
        "avg_ttfa_ms": round(float(avg_ttfa), 1),
        "voices": ["aarav_colloquial (Male)", "priya_colloquial (Female)"],
        "sample_rate": 24000,
    }

async def main():
    results: Dict[str, Any] = {"llm": [], "asr": [], "tts": None}

    # 1. LLM Benchmarks
    available_models = ["qwen2.5:1.5b", "llama3.2:1b"]
    # Check if qwen2.5:3b is ready
    import httpx
    try:
        r = httpx.get("http://127.0.0.1:11434/api/tags")
        if r.status_code == 200:
            names = [m["name"] for m in r.json().get("models", [])]
            for candidate in ["qwen2.5:3b", "llama3.2:3b"]:
                if candidate in names or f"{candidate}:latest" in names:
                    available_models.append(candidate)
    except Exception:
        pass

    for m in available_models:
        res = await benchmark_ollama_model(m)
        results["llm"].append(res)

    # 2. ASR Benchmarks
    # Test base on GPU vs CPU int8, and small on CPU int8
    results["asr"].append(benchmark_faster_whisper("base", "cuda", "float16"))
    results["asr"].append(benchmark_faster_whisper("base", "cpu", "int8"))
    results["asr"].append(benchmark_faster_whisper("small", "cpu", "int8"))

    # 3. TTS Benchmark
    results["tts"] = await benchmark_tts()

    # Save to JSON
    with open("benchmarks/measured_benchmarks.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nBenchmark completed. Results written to benchmarks/measured_benchmarks.json.")

if __name__ == "__main__":
    asyncio.run(main())
