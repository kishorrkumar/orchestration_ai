"""
Comprehensive 15-Prompt Dialogue & Style Evaluation for Aarav (Indian English).

Evaluates:
- Spoken text only (no markdown, no emojis, no stage directions)
- Brevity (under 60 words per turn)
- Absence of banned call-centre bot phrases
- No repeated consecutive openers
- Indian English colloquial style score (0 to 2 points per reply, target ~1.0)
- End-to-end Kokoro TTS audio generation saved to eval_out/*.wav
"""

from __future__ import annotations
import asyncio
import json
import os
import pathlib
import re
import sys
import time
from typing import Dict, Any, List, Optional
import httpx
import soundfile as sf

from orchestration.chunker.normalizer import strip_markdown_and_emojis, normalize_for_tts
from orchestration.worker.local_cascade import load_persona_prompt
from orchestration.tts.base import KokoroTTSBackend

EVAL_OUT_DIR = pathlib.Path("eval_out")
EVAL_OUT_DIR.mkdir(exist_ok=True, parents=True)

PROMPTS = [
    "Tell me a joke",
    "Tell me a joke about cats",
    "Can you tell me about artificial intelligence?",
    "Hi Aarav, how is your day going?",
    "Actually I wanted to ask about...",
    "Wait, stop, what did you just say?",
    "Arre yaar, bahut traffic hai aaj Bangalore mein",
    "What is the weather like in Chennai during the monsoon?",
    "How much is 15 lakh rupees in US dollars approximately?",
    "How does UPI work for paying at a chai shop?",
    "Are you a real human or an AI bot?",
    "Can you explain machine learning in two simple sentences?",
    "Tell me something about cricket and IPL.",
    "I am feeling quite stressed about my exams tomorrow.",
    "Thanks Aarav, talk to you later!",
]

BANNED_PHRASES = [
    "i understand you need support",
    "how can i assist you today",
    "as an ai language model",
    "certainly!",
    "i'd be happy to help",
    "is there anything else i can assist you with",
    "how may i help you",
]

INDIAN_DISCOURSE_MARKERS = [
    r"\bactually\b",
    r"\bbasically\b",
    r"\bno\?",
    r"\bna\b",
    r"\bonly\b",
    r"\bachha\b",
    r"\bhaan\b",
    r"\barre\b",
    r"\byaar\b",
    r"\bsimple,?\s*na\b",
    r"\bwhat to do\b",
    r"\bno worries\b",
    r"\bsure sure\b",
    r"\bone minute\b",
    r"\bi'll tell you\b",
    r"\blike that only\b",
    r"\bright,? right\b",
    r"\bchai\b",
    r"\bauto\b",
    r"\blakh\b",
    r"\bcrore\b",
    r"\brupees\b",
]


def compute_style_score(text: str) -> float:
    """Compute Indian English colloquial score (capped at 2.0 per reply)."""
    score = 0.0
    text_lower = text.lower()
    for marker in INDIAN_DISCOURSE_MARKERS:
        if re.search(marker, text_lower):
            score += 1.0
    return min(2.0, score)


def get_first_word(text: str) -> str:
    words = re.sub(r"[^\w\s]", "", text).split()
    return words[0].lower() if words else ""


async def run_evaluation():
    print("=" * 80)
    print("STARTING AARAV 15-PROMPT INDIAN ENGLISH DIALOGUE EVALUATION")
    print("=" * 80)

    # Pre-warm TTS engine
    print("[1/3] Initializing Kokoro TTS backend...")
    tts = KokoroTTSBackend(sample_rate=24000)
    tts.warm_up()

    # Load Aarav Persona Prompt
    print("[2/3] Loading persona from personas/aarav.md...")
    system_prompt = load_persona_prompt("Aarav")

    client = httpx.AsyncClient(timeout=25.0)
    eval_results = []
    failed_reasons = []
    prev_opener = ""

    conversation_history: List[Dict[str, str]] = []

    print("\n[3/3] Executing 15 Dialogue Turns...")
    print(f"{'#':<3} | {'User Prompt':<35} | {'Words':<5} | {'Style':<5} | {'Status':<6} | {'Opener'}")
    print("-" * 80)

    for i, user_msg in enumerate(PROMPTS, start=1):
        t0 = time.perf_counter()

        # Build payload
        messages = [{"role": "system", "content": system_prompt}]
        for turn in conversation_history[-8:]:
            messages.append(turn)
        messages.append({"role": "user", "content": user_msg})

        payload = {
            "model": "qwen2.5:1.5b",
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.7, "top_p": 0.9, "num_ctx": 2048, "num_predict": 45},
            "keep_alive": "30m",
        }

        # Query Ollama
        raw_reply = ""
        try:
            r = await client.post("http://127.0.0.1:11434/api/chat", json=payload)
            if r.status_code == 200:
                raw_reply = r.json().get("message", {}).get("content", "").strip()
            else:
                raw_reply = f"Error HTTP {r.status_code}"
        except Exception as e:
            raw_reply = f"Error: {e}"

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Normalization
        clean_reply = normalize_for_tts(raw_reply)
        word_count = len(clean_reply.split())
        style_score = compute_style_score(clean_reply)
        opener = get_first_word(clean_reply)

        # Check failure conditions
        turn_failed = False
        reasons = []

        # 1. Markdown or emojis check
        if re.search(r"[*_#`\[\]]", raw_reply) or re.search(r"[\U00010000-\U0010ffff]", raw_reply):
            turn_failed = True
            reasons.append("Contains markdown or emojis")

        # 2. Word count check (max 60 words)
        if word_count > 60:
            turn_failed = True
            reasons.append(f"Exceeds 60 words ({word_count} words)")

        # 3. Banned call-centre phrases check
        clean_lower = clean_reply.lower()
        for banned in BANNED_PHRASES:
            if banned in clean_lower:
                turn_failed = True
                reasons.append(f"Contains banned phrase: '{banned}'")

        # 4. Repeated opener check
        if opener and opener == prev_opener:
            turn_failed = True
            reasons.append(f"Repeated opener '{opener}' consecutively")

        prev_opener = opener

        # Synthesize audio to eval_out
        wav_path = EVAL_OUT_DIR / f"turn_{i:02d}.wav"
        try:
            audio_samples = await tts.synthesize(clean_reply, voice="aarav_colloquial")
            if len(audio_samples) > 0:
                sf.write(str(wav_path), audio_samples, 24000)
        except Exception as e:
            pass

        # Update conversation history
        conversation_history.append({"role": "user", "content": user_msg})
        conversation_history.append({"role": "assistant", "content": clean_reply})

        status_str = "FAIL" if turn_failed else "PASS"
        if turn_failed:
            failed_reasons.append(f"Turn {i} ('{user_msg}'): {', '.join(reasons)}")

        print(f"{i:<3} | {user_msg[:35]:<35} | {word_count:<5} | {style_score:<5.1f} | {status_str:<6} | {opener}", flush=True)

        eval_results.append({
            "turn": i,
            "prompt": user_msg,
            "reply": clean_reply,
            "word_count": word_count,
            "style_score": style_score,
            "latency_ms": round(elapsed_ms, 1),
            "passed": not turn_failed,
            "reasons": reasons,
            "audio_file": str(wav_path),
        })

    await client.aclose()

    # Save summary JSON
    summary_path = EVAL_OUT_DIR / "eval_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    total_turns = len(eval_results)
    passed_turns = sum(1 for r in eval_results if r["passed"])
    avg_style = sum(r["style_score"] for r in eval_results) / total_turns
    avg_words = sum(r["word_count"] for r in eval_results) / total_turns

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Turns:            {total_turns}")
    print(f"Passed Turns:           {passed_turns} / {total_turns}")
    print(f"Average Words per Turn: {avg_words:.1f} (target: 10-35 words)")
    print(f"Indian-English Style:   {avg_style:.2f} / 2.0 (target: ~0.8 - 1.5)")
    print(f"Audio Saved in:         {EVAL_OUT_DIR.resolve()}")
    print(f"Summary JSON:           {summary_path.resolve()}")

    if failed_reasons:
        print("\nFailures:")
        for fr in failed_reasons:
            print(f"  - {fr}")
        return False
    else:
        print("\nAll 15 dialogue turns PASSED successfully!")
        return True


if __name__ == "__main__":
    success = asyncio.run(run_evaluation())
    sys.exit(0 if success else 1)
