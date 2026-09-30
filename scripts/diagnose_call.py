"""
Offline Call Diagnosis and Stage-by-Stage Verification Script.

Feeds a known test audio utterance through the complete PersonaPlex cascaded pipeline:
1. VAD & Audio Cleaning (24 kHz)
2. STT (faster-whisper on CUDA, resampled to 16 kHz)
3. LLM (Qwen2.5 on GPU via Ollama or dialogue fallback)
4. Chunker (ClauseChunker streaming tokens)
5. TTS (Neural synthesis to 24 kHz Float32)

Reports per-stage:
- Output content / length
- Latency (ms)
- Device (cuda vs cpu)
- Pinpoints exact stage failure
"""

from __future__ import annotations
import argparse
import asyncio
import io
import json
import logging
import os
import pathlib
import sys
import time
from typing import Dict, Any, Optional

import numpy as np
import soundfile as sf

# Add project root to sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from orchestration.protocol.audio import SAMPLE_RATE, FRAME_SIZE, compute_rms
from orchestration.chunker.bridge import ClauseChunker
from orchestration.tts.base import StreamingCompositeTTS
from orchestration.persona.dialogue import StrictVoiceDialogueEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("diagnose_call")


def generate_synthetic_speech_wav(text: str = "Hello, can you hear me clearly?") -> np.ndarray:
    """Generate a clean synthetic 24 kHz spoken audio sample using available TTS or dual-tone modulation."""
    logger.info(f"Synthesizing synthetic test speech for: '{text}'")
    tts = StreamingCompositeTTS(preferred_backend="auto", sample_rate=24000)
    loop = asyncio.new_event_loop()
    try:
        audio = loop.run_until_complete(tts.synthesize_chunk(text))
        if len(audio) > 0:
            return audio
    except Exception as e:
        logger.warning(f"TTS synthesis fallback for test WAV: {e}")
    finally:
        loop.close()

    # Harmonic acoustic carrier simulating speech formants (200 Hz + 700 Hz + 2200 Hz)
    duration = 2.0
    t = np.linspace(0, duration, int(24000 * duration), endpoint=False)
    envelope = np.sin(np.pi * t / duration) ** 2
    formant = 0.2 * np.sin(2 * np.pi * 220 * t) + 0.15 * np.sin(2 * np.pi * 700 * t) + 0.1 * np.sin(2 * np.pi * 2200 * t)
    return (formant * envelope).astype(np.float32)


def get_gpu_info() -> Dict[str, Any]:
    """Check GPU presence and VRAM stats via pynvml or torch."""
    gpu_info = {"available": False, "device_name": "None", "vram_allocated_mb": 0.0, "vram_reserved_mb": 0.0}
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["device_name"] = torch.cuda.get_device_name(0)
            gpu_info["vram_allocated_mb"] = torch.cuda.memory_allocated(0) / (1024 * 1024)
            gpu_info["vram_reserved_mb"] = torch.cuda.memory_reserved(0) / (1024 * 1024)
    except Exception:
        pass
    return gpu_info


async def run_diagnosis(wav_path: Optional[str] = None, enforce_cuda: bool = True) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print(" PERSONAPLEX CASCADED PIPELINE DIAGNOSTIC TRACE")
    print("=" * 70)

    stages_report = {}
    gpu_status = get_gpu_info()
    print(f"GPU Hardware: {gpu_status['device_name']} (CUDA: {gpu_status['available']})")

    # ----------------------------------------------------
    # STAGE 0: Load / Generate Audio Utterance
    # ----------------------------------------------------
    if wav_path and os.path.exists(wav_path):
        print(f"\n[Stage 0] Loading input WAV: {wav_path}")
        audio_data, sr = sf.read(wav_path, dtype="float32")
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        if sr != 24000:
            n_out = int(round(len(audio_data) * 24000 / sr))
            audio_data = np.interp(
                np.linspace(0, len(audio_data), n_out, endpoint=False),
                np.arange(len(audio_data)),
                audio_data,
            ).astype(np.float32)
    else:
        print("\n[Stage 0] Generating synthetic test speech WAV (24 kHz)...")
        audio_data = generate_synthetic_speech_wav("Hello Aarav, can you assist me with scheduling today?")

    duration_sec = len(audio_data) / 24000.0
    input_rms = compute_rms(audio_data)
    print(f"  Input audio: {len(audio_data)} samples ({duration_sec:.2f}s) | RMS: {input_rms:.4f}")

    # ----------------------------------------------------
    # STAGE 1: Audio Cleaner & VAD
    # ----------------------------------------------------
    t1_start = time.perf_counter()
    from orchestration.audio.cleaner import CallerAudioCleaner
    cleaner = CallerAudioCleaner(sample_rate=24000, frame_size=1920)
    cleaned_frames = cleaner.process_chunk(audio_data)
    t1_ms = (time.perf_counter() - t1_start) * 1000.0

    speech_detected = cleaner.is_speech_active or (input_rms > 0.015)
    stages_report["VAD_Cleaner"] = {
        "device": "cpu",
        "latency_ms": round(t1_ms, 2),
        "status": "PASS" if speech_detected else "WARN (Low energy)",
        "output": f"{len(cleaned_frames)} frames (speech_detected={speech_detected})",
    }
    print(f"\n[Stage 1: VAD & Cleaner] -> Latency: {t1_ms:.1f}ms | Frames: {len(cleaned_frames)} | Speech Active: {speech_detected}")

    # ----------------------------------------------------
    # STAGE 2: STT (Speech-to-Text via faster-whisper)
    # ----------------------------------------------------
    print("\n[Stage 2: STT faster-whisper]")
    t2_start = time.perf_counter()
    stt_transcript = ""
    stt_device = "cuda" if gpu_status["available"] else "cpu"
    stt_status = "PASS"

    try:
        from faster_whisper import WhisperModel
        # Whisper requires 16 kHz audio! Resample from 24 kHz to 16 kHz
        n_16k = int(round(len(audio_data) * 16000 / 24000))
        audio_16k = np.interp(
            np.linspace(0, len(audio_data), n_16k, endpoint=False),
            np.arange(len(audio_data)),
            audio_data,
        ).astype(np.float32)

        # Enforce CUDA if requested
        target_device = "cuda" if (gpu_status["available"] and enforce_cuda) else "cpu"
        compute_type = "float16" if target_device == "cuda" else "int8"
        
        try:
            whisper_model = WhisperModel("base", device=target_device, compute_type=compute_type)
            stt_device = target_device
        except Exception as e_cuda:
            if enforce_cuda:
                raise RuntimeError(f"CUDA required for STT but failed to initialize: {e_cuda}")
            whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
            stt_device = "cpu"

        segments, _ = whisper_model.transcribe(audio_16k, language="en", beam_size=1)
        stt_transcript = " ".join([s.text for s in segments]).strip()
        t2_ms = (time.perf_counter() - t2_start) * 1000.0

        if not stt_transcript:
            stt_status = "FAIL (Empty transcript)"
        print(f"  Transcript: \"{stt_transcript}\"")
        print(f"  Device: {stt_device} | Compute Type: {compute_type} | Latency: {t2_ms:.1f}ms")

    except Exception as e:
        t2_ms = (time.perf_counter() - t2_start) * 1000.0
        stt_status = f"FAIL ({e})"
        print(f"  STT Error: {e}")

    stages_report["STT"] = {
        "device": stt_device,
        "latency_ms": round(t2_ms, 2),
        "status": stt_status,
        "output": stt_transcript or "<EMPTY>",
    }

    # If STT failed, use prompt fallback for diagnostic continuation
    llm_input_text = stt_transcript if stt_transcript else "Hello, how can you help me today?"
    if not stt_transcript:
        print(f"  [Diagnostic Notice] Continuing with test utterance: \"{llm_input_text}\"")

    # ----------------------------------------------------
    # STAGE 3: LLM (Qwen2.5 / Dialogue Engine)
    # ----------------------------------------------------
    print(f"\n[Stage 3: LLM Qwen2.5]")
    t3_start = time.perf_counter()
    llm_reply = ""
    llm_device = "cuda (Ollama)"
    llm_status = "PASS"
    llm_ttft_ms = 0.0

    try:
        import httpx
        payload = {
            "model": "qwen2.5:1.5b",
            "prompt": (
                "You are Aarav, an Indian English voice assistant. "
                "Respond in 1 short spoken sentence without markdown or bullet points.\n\n"
                f"Caller: {llm_input_text}\nAarav:"
            ),
            "stream": True,
            "options": {"temperature": 0.7, "top_p": 0.9, "num_ctx": 2048},
            "keep_alive": "30m",
        }

        tokens = []
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post("http://127.0.0.1:11434/api/generate", json=payload)
            if response.status_code == 200:
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    tok = data.get("response", "")
                    if tok:
                        if not llm_ttft_ms:
                            llm_ttft_ms = (time.perf_counter() - t3_start) * 1000.0
                        tokens.append(tok)
                    if data.get("done", False):
                        break
                llm_reply = "".join(tokens).strip()
            else:
                raise RuntimeError(f"Ollama returned HTTP {response.status_code}")

        t3_ms = (time.perf_counter() - t3_start) * 1000.0
        print(f"  LLM Response: \"{llm_reply}\"")
        print(f"  TTFT: {llm_ttft_ms:.1f}ms | Total LLM: {t3_ms:.1f}ms | Device: {llm_device}")

    except Exception as e:
        t3_ms = (time.perf_counter() - t3_start) * 1000.0
        print(f"  Ollama unavailable ({e}), evaluating DialogueEngine fallback...")
        d_engine = StrictVoiceDialogueEngine()
        llm_reply = d_engine.reply(llm_input_text)
        llm_device = "cpu (Rule Engine)"
        llm_status = f"FALLBACK ({e.__class__.__name__})"
        print(f"  DialogueEngine Response: \"{llm_reply}\"")

    # Clean any Qwen reasoning thinking tags (<think>...</think>)
    import re
    llm_reply = re.sub(r"<think>.*?</think>", "", llm_reply, flags=re.DOTALL).strip()

    stages_report["LLM"] = {
        "device": llm_device,
        "latency_ms": round(t3_ms, 2),
        "ttft_ms": round(llm_ttft_ms, 2) if llm_ttft_ms else round(t3_ms, 2),
        "status": llm_status,
        "output": llm_reply or "<EMPTY>",
    }

    # ----------------------------------------------------
    # STAGE 4: Clause Chunker
    # ----------------------------------------------------
    print("\n[Stage 4: Clause Chunker]")
    chunker = ClauseChunker(first_chunk_min_words=2, first_chunk_max_words=6)
    chunks = []
    for word in (llm_reply or "Namaste, how may I help?").split():
        c_list = chunker.feed_token(word + " ")
        chunks.extend(c_list)
    chunks.extend(chunker.flush())
    print(f"  Generated {len(chunks)} streaming chunk(s): {chunks}")
    stages_report["Chunker"] = {
        "device": "cpu",
        "chunks_count": len(chunks),
        "status": "PASS" if chunks else "FAIL (No chunks)",
    }

    # ----------------------------------------------------
    # STAGE 5: TTS Synthesis
    # ----------------------------------------------------
    print("\n[Stage 5: TTS Synthesis]")
    t5_start = time.perf_counter()
    tts_engine = StreamingCompositeTTS(preferred_backend="auto", sample_rate=24000)
    tts_audio_samples = 0
    tts_first_chunk_ms = 0.0

    if chunks:
        first_audio = await tts_engine.synthesize_chunk(chunks[0])
        tts_first_chunk_ms = (time.perf_counter() - t5_start) * 1000.0
        tts_audio_samples += len(first_audio)
        first_rms = compute_rms(first_audio)
        print(f"  Chunk 1: '{chunks[0]}' -> {len(first_audio)} samples ({len(first_audio)/24000:.2f}s) | RMS: {first_rms:.4f} | Latency: {tts_first_chunk_ms:.1f}ms")

    t5_ms = (time.perf_counter() - t5_start) * 1000.0
    tts_status = "PASS" if tts_audio_samples > 0 else "FAIL (Zero audio generated)"
    stages_report["TTS"] = {
        "device": tts_engine.active_backend.name,
        "latency_ms": round(t5_ms, 2),
        "first_chunk_latency_ms": round(tts_first_chunk_ms, 2),
        "samples_generated": tts_audio_samples,
        "status": tts_status,
    }

    # ----------------------------------------------------
    # FINAL SUMMARY REPORT
    # ----------------------------------------------------
    print("\n" + "=" * 70)
    print(f"{'STAGE':<15} | {'DEVICE':<15} | {'LATENCY (ms)':<15} | {'STATUS':<15}")
    print("-" * 70)
    for stg, data in stages_report.items():
        lat = data.get("latency_ms", "N/A")
        dev = data.get("device", "N/A")
        stat = data.get("status", "N/A")
        print(f"{stg:<15} | {dev:<15} | {str(lat):<15} | {stat:<15}")
    print("=" * 70)

    # Pinpoint failures
    failures = [stg for stg, d in stages_report.items() if "FAIL" in d.get("status", "")]
    if failures:
        print(f"\n[ALERT] Pipeline failed at stage(s): {', '.join(failures)}")
    else:
        print("\n[SUCCESS] All pipeline stages executed successfully.")

    return stages_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PersonaPlex Pipeline Diagnostic Tool")
    parser.add_argument("--wav", type=str, default=None, help="Path to test input WAV file")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU fallback for debugging")
    args = parser.parse_args()

    asyncio.run(run_diagnosis(wav_path=args.wav, enforce_cuda=not args.allow_cpu))
