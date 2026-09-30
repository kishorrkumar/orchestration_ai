"""
Benchmark script for Indian-English TTS Backends.
Evaluates:
- First-chunk latency (TTFA)
- Real-Time Factor (RTF = compute_time / audio_duration)
- VRAM usage (if GPU available)
- Saves audio WAV files to benchmarks/ directory for listening comparison.

Usage:
    .venv/bin/python scripts/tts_bench.py
"""

import asyncio
import os
import pathlib
import sys
import time
from typing import Dict, List

import numpy as np
import soundfile as sf

# Add parent directory to sys.path
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from orchestration.tts.base import (
    KokoroTTSBackend,
    EdgeTTSBackend,
    IndicTTSBackend,
    XTTSBackend,
    PiperTTSBackend,
    FallbackTTSBackend,
)
from orchestration.tts.text_norm import normalize_indian_english_text

BENCHMARK_SENTENCES = [
    "Namaste! This is Aarav from customer support, how may I assist you today?",
    "Your account balance is ₹15,450, and the pending transfer of ₹2.5 lakh has been initiated.",
    "I have booked your appointment for tomorrow at 4:30 pm with Doctor Sharma.",
    "Please verify your mobile number ending in 9876 before we proceed with the payment.",
    "Understood, I can definitely help you with that. Would you like me to walk you through the details step by step?",
]

OUTPUT_DIR = BASE_DIR / "benchmarks"
OUTPUT_DIR.mkdir(exist_ok=True)


def get_vram_mb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    return 0.0


async def benchmark_backend(backend, name: str) -> List[Dict]:
    results = []
    print(f"\n=======================================================")
    print(f"Benchmarking Backend: {name} (Available: {backend.is_available()})")
    print(f"=======================================================")

    if not backend.is_available():
        print(f"Skipping {name}: dependencies or model weights not detected.")
        return results

    # Warmup
    backend.warm_up()

    for idx, sentence in enumerate(BENCHMARK_SENTENCES, 1):
        norm_text = normalize_indian_english_text(sentence)
        vram_before = get_vram_mb()

        t0 = time.perf_counter()
        audio = await backend.synthesize(norm_text)
        compute_time = time.perf_counter() - t0

        vram_after = get_vram_mb()
        audio_dur = len(audio) / backend.sample_rate if len(audio) > 0 else 0.0
        rtf = (compute_time / audio_dur) if audio_dur > 0 else 0.0
        ttfa_ms = compute_time * 1000.0

        out_path = OUTPUT_DIR / f"{name}_sent_{idx}.wav"
        if len(audio) > 0:
            sf.write(str(out_path), audio, backend.sample_rate)

        results.append({
            "sentence_idx": idx,
            "ttfa_ms": ttfa_ms,
            "compute_sec": compute_time,
            "audio_dur_sec": audio_dur,
            "rtf": rtf,
            "vram_mb": vram_after,
            "file": str(out_path.name),
        })

        print(
            f"  [{idx}/5] TTFA: {ttfa_ms:6.1f}ms | "
            f"Audio: {audio_dur:4.2f}s | "
            f"RTF: {rtf:5.3f}x | "
            f"Saved: {out_path.name}"
        )

    return results


async def main():
    print("Initializing TTS Backends...")
    backends = [
        ("Kokoro-v1.0 (ONNX)", KokoroTTSBackend(default_voice="hm_omega")),
        ("Edge-TTS (en-IN-Prabhat)", EdgeTTSBackend(default_voice="en-IN-PrabhatNeural")),
        ("Indic-TTS (AI4Bharat)", IndicTTSBackend()),
        ("XTTS-v2 (Voice Cloning)", XTTSBackend()),
        ("Piper TTS", PiperTTSBackend()),
        ("Native Fallback", FallbackTTSBackend()),
    ]

    all_results = {}
    for name, b in backends:
        res = await benchmark_backend(b, name)
        if res:
            all_results[name] = res

    # Summary Table
    print("\n" + "=" * 78)
    print(f"{'Backend':<26} | {'Avg TTFA (ms)':<14} | {'Avg RTF':<10} | {'VRAM (MB)':<10} | Status")
    print("-" * 78)
    for name, res in all_results.items():
        avg_ttfa = np.mean([r["ttfa_ms"] for r in res])
        avg_rtf = np.mean([r["rtf"] for r in res])
        vram = res[-1]["vram_mb"]
        status = "PASSED" if avg_rtf < 1.0 else "SLOW (>1.0x)"
        print(f"{name:<26} | {avg_ttfa:14.1f} | {avg_rtf:10.3f} | {vram:10.1f} | {status}")
    print("=" * 78)
    print(f"Benchmark audio files saved to: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    asyncio.run(main())
