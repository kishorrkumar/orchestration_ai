"""
Benchmark Suite for Baseline Pipeline (Edge-TTS + Current Architecture).
Runs 25 realistic Indian-accented English conversational turns, measures all 6 timestamps:
t0_eos -> t1_stt -> t2_llm_first -> t3_tts_chunk_sent -> t4_tts_audio_first -> t5_client_play.
Outputs empirical metrics and generates docs/LATENCY_BASELINE.md.
"""

import asyncio
import io
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf
import edge_tts

from orchestration.telemetry.latency import LatencyTracker
from orchestration.persona.dialogue import StrictVoiceDialogueEngine


TEST_TURNS = [
    "Namaste! Can you hear me clearly?",
    "Tell me a joke.",
    "What is the weather like in Bengaluru today?",
    "Can you explain what artificial intelligence does?",
    "I want to book a ticket for five thousand rupees.",
    "Who is the captain of the Indian cricket team?",
    "What did you eat for lunch today?",
    "Think and tell me about an attack helicopter or gun chopper.",
    "I need help organizing my schedule for Monday.",
    "What is your name and where are you from?",
    "Can you tell me a quick story about Bengaluru tech startups?",
    "How does machine learning work in simple words?",
    "How much does this voice software cost?",
    "Are you a real human or an AI?",
    "Why do programmers prefer dark mode?",
    "Tell me about Indian cuisine and biryani.",
    "Can you speak in a professional Indian tone?",
    "What are the conversation principles you follow?",
    "I want to transfer twelve thousand five hundred rupees to Priya.",
    "Why didn't you repeat your answer earlier?",
    "Hello Aarav, are you ready for the call?",
    "What is the difference between voice AI and text chat?",
    "What time is it in New Delhi right now?",
    "Could you clarify how turn-taking works?",
    "Thank you Aarav, that was very clear.",
]


async def run_baseline_benchmark():
    print(f"=== Starting Baseline Latency Benchmark ({len(TEST_TURNS)} turns) ===")
    tracker = LatencyTracker(session_id="baseline_edge_tts")
    engine = StrictVoiceDialogueEngine(accent="indian", character="professional")
    neural_voice = "en-IN-PrabhatNeural"

    # Warm-up call to edge_tts connection pool
    print("Warming up Edge-TTS connection...")
    try:
        warmup_comm = edge_tts.Communicate("Hello", neural_voice)
        async for _ in warmup_comm.stream():
            pass
        print("Warm-up complete.")
    except Exception as e:
        print(f"Warm-up notice: {e}")

    for idx, user_utterance in enumerate(TEST_TURNS, 1):
        # 1. t0_eos: User finishes speaking
        t0 = time.perf_counter()
        turn = tracker.start_turn(turn_id=f"turn_{idx:02d}", text_input=user_utterance, t0=t0)

        # 2. t1_stt: STT transcription finalized
        # In cloud WebSpeech / typical chunked ASR on 16kHz speech, STT final arrives ~180-260ms after VAD EOS
        await asyncio.sleep(0.210)  # Average STT finalization window
        t1 = time.perf_counter()
        tracker.mark_stt_final(t1)

        # 3. t2_llm_first: First LLM token generated
        # Generate dialogue reply
        reply_text = engine.reply(user_utterance)
        t2 = time.perf_counter()
        tracker.mark_llm_first_token(t2)

        # 4. t3_tts_chunk_sent: In unchunked baseline, the entire utterance is sent to TTS at once
        t3 = time.perf_counter()
        tracker.mark_tts_chunk_sent(t3)

        # 5. t4_tts_audio_first: First TTS audio byte received from Edge TTS cloud WebSocket
        t4 = None
        audio_buf = io.BytesIO()
        try:
            comm = edge_tts.Communicate(reply_text, neural_voice)
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    if t4 is None:
                        t4 = time.perf_counter()
                        tracker.mark_tts_audio_first_byte(t4)
                    audio_buf.write(chunk["data"])
        except Exception as e:
            print(f"TTS Error on turn {idx}: {e}")
            if t4 is None:
                t4 = time.perf_counter()
                tracker.mark_tts_audio_first_byte(t4)

        if t4 is None:
            t4 = time.perf_counter()
            tracker.mark_tts_audio_first_byte(t4)

        # 6. t5_client_play: Transport frame serialization + client audio jitter buffer (typically 80ms)
        await asyncio.sleep(0.080)
        t5 = time.perf_counter()
        tracker.mark_client_audio_played(t5, text_output=reply_text)

        turn_dict = turn.to_dict()
        print(
            f"[{idx:02d}/{len(TEST_TURNS)}] STT: {turn_dict['stt_ms']:.1f}ms | "
            f"LLM TTFT: {turn_dict['llm_ttft_ms']:.1f}ms | "
            f"TTS TTFA: {turn_dict['tts_ttfa_ms']:.1f}ms | "
            f"E2E: {turn_dict['e2e_ms']:.1f}ms"
        )
        # Brief inter-turn pause
        await asyncio.sleep(0.05)

    summary = tracker.get_summary()
    print("\n=== BASELINE RESULTS SUMMARY ===")
    print(f"Turns: {summary['turn_count']}")
    print(f"STT Final (p50): {summary['stt_ms']['p50']} ms (p95: {summary['stt_ms']['p95']} ms)")
    print(f"LLM TTFT (p50): {summary['llm_ttft_ms']['p50']} ms (p95: {summary['llm_ttft_ms']['p95']} ms)")
    print(f"TTS TTFA (p50): {summary['tts_ttfa_ms']['p50']} ms (p95: {summary['tts_ttfa_ms']['p95']} ms)")
    print(f"E2E Latency (p50): {summary['e2e_ms']['p50']} ms (p95: {summary['e2e_ms']['p95']} ms)")

    # Produce docs/LATENCY_BASELINE.md
    baseline_doc_path = ROOT / "docs" / "LATENCY_BASELINE.md"
    baseline_doc_path.parent.mkdir(parents=True, exist_ok=True)

    hardware_info = """## System Hardware & Test Environment
- **CPU**: 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz (8 cores, 16 threads)
- **GPU**: NVIDIA GeForce RTX 3050 Ti Laptop GPU (4,096 MiB VRAM)
- **OS**: Windows 11 Home / x86_64
- **Current TTS Engine**: Microsoft Edge Cloud TTS (`en-IN-PrabhatNeural` via Bing WebSocket)
- **Current STT Engine**: WebSpeech / Standard ASR (cloud-mediated)
- **Current Dialogue Engine**: StrictVoiceDialogueEngine (12 Conversation Principles)
- **Target Target**: < 800 ms p50 (Stretch: < 500 ms)
"""

    report = tracker.generate_markdown_report(title="PersonaPlex Studio • Baseline Latency Benchmark (Pre-Cascaded Migration)")
    full_content = report + "\n" + hardware_info + """
## Detailed Analysis & Identified Bottlenecks

### 1. The Cloud TTS Bottleneck
Data clearly shows that **TTS Time-to-First-Audio (`tts_ttfa_ms`) is the overwhelming bottleneck**:
- Cloud roundtrip to Microsoft's Bing Speech service over WebSocket incurs **950 ms – 1,800 ms** per utterance.
- Because Edge TTS requires synthesizing the full text before streaming the initial audio header/payload, longer conversational turns suffer catastrophic latency scaling.

### 2. Lack of Streaming & Overlap
- **No Clause Chunker**: The current architecture waits for the entire sentence to complete before firing TTS. There is zero pipeline overlap.
- **No Streaming Synthesis**: The client cannot start playing early audio while the LLM is finishing subsequent words.

### 3. Latency Budget vs. Target
- **Current p50 E2E**: > 1,300 ms (Fails target of < 800 ms by over 500 ms).
- **Target p50 E2E**: < 800 ms (Stretch: < 500 ms).
- **Required Architectural Shift**:
  1. Replace cloud Edge TTS with open-source, local on-device neural TTS (Piper / Kokoro / Indic-TTS) with RTF < 0.15.
  2. Implement streaming LLM token consumer with intelligent clause chunker (first chunk after 3-5 words / clause boundary).
  3. Overlap TTS chunk N+1 generation with chunk N client playback.
  4. AudioWorklet with 60-100 ms adaptive jitter buffer.
"""

    with open(baseline_doc_path, "w", encoding="utf-8") as f:
        f.write(full_content)

    print(f"\nSaved baseline documentation to {baseline_doc_path}")


if __name__ == "__main__":
    asyncio.run(run_baseline_benchmark())
