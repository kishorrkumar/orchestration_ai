# Technical Decisions: Local Cascade Real-Time Voice Agent

Based on the audit ([docs/audit.md](file:///docs/audit.md)) and measured benchmarks on the Intel i7-11800H + NVIDIA RTX 3050 Ti Laptop GPU ([docs/benchmarks.md](file:///docs/benchmarks.md)), the following architectural and model decisions are implemented.

---

## 1. Compute & VRAM Budget Partitioning
- **Hardware constraints:** 4,096 MiB VRAM (RTX 3050 Ti), 15.77 GB RAM, ~7.7 GB free disk on C:.
- **GPU Assignment:** Dedicated exclusively to the Ollama LLM (`qwen2.5:1.5b` or `qwen2.5:3b`). With `num_ctx 2048`, memory consumption is 1.65 GB to 2.60 GB, well within the 4 GB VRAM limit.
- **CPU Assignment:**
  - Silero VAD (ONNX runtime on CPU, < 2% CPU utilization).
  - `faster-whisper` (base model, int8 quantization, 4 CPU threads): Benchmark shows 470.2 ms for 3s audio (RTF 0.157).
  - Kokoro-82M TTS (ONNX runtime on CPU): First clause synthesizes in ~400 ms.
- **Outcome:** Zero GPU VRAM thrashing or memory contention between models.

---

## 2. ASR Engine & Bias Configuration
- **Model Choice:** `faster-whisper` `base` (int8) on CPU.
- **Language & Biasing:** `language="en"`, `beam_size=1`, with an authoritative vocabulary prompt:
  ```python
  initial_prompt = (
      "artificial intelligence, machine learning, Aarav, Bengaluru, Chennai, "
      "Tamil, Hindi, cricket, Bollywood, UPI, Swiggy, Zomato, IPL, tech, phone"
  )
  ```
- **Fixes:** Resolves the phonetic misrecognition of "artificial intelligence" as "artist intelligence".
- **Execution Threading:** Whisper `transcribe` must run in `asyncio.to_thread` to prevent blocking the asyncio event loop and WebSocket pings.

---

## 3. Intelligent Turn Detection (Silero VAD + Linguistic Heuristics)
- **Problem:** Fixed 450 ms silence timer was cutting callers off mid-thought (e.g. "Tell me a joke about... [460ms pause] ...cats").
- **Solution:**
  - Base silence hangover threshold: **650 ms**.
  - **Linguistic Trailing Word Extension:** If the recognized transcript ends with a connective, preposition, or conjunction (`about`, `and`, `the`, `to`, `of`, `with`, `because`, `but`, `so`, `like`, `or`, `that`, `for`), or lacks a verb in a multi-word fragment:
    - Grant an additional **700 ms** silence window.
    - If the caller speaks again, prepend/concatenate the audio into a single unified turn.

---

## 4. LLM Selection & Conversational Memory
- **Model Choice:** `qwen2.5:1.5b` (default for sub-second latency) with `qwen2.5:3b` as high-quality selectable alternative.
- **Endpoint:** Ollama `/api/chat` streaming HTTP API with `keep_alive: "30m"`.
- **Context Management:** Rolling window of the last 10 conversational turns (`conversation_history[-10:]`). Discards raw string completion in favor of structured roles (`system`, `user`, `assistant`).
- **Prompt:** `personas/aarav.md` defining natural Indian English phrasing, contractions, max 15-20 words per sentence, and banned call-center robotic phrases.

---

## 5. Streaming Clause Chunker & TTS Pipelining
- **Chunking Strategy:**
  - Chunk 1: Flushes at 4 to 8 words or first punctuation mark (`.`, `,`, `?`, `!`, `;`).
  - Chunk N > 1: Flushes at 8 to 16 words.
- **Concurrency:** Chunk N+1 is synthesized asynchronously while Chunk N is streaming over the WebSocket frame buffer.
- **Audio Framing:** 1,920 float32 samples per frame (80 ms @ 24,000 Hz) paced at 12.5 Hz to match the PersonaPlex binary wire protocol.

---

## 6. Barge-in & Latency Masking
- **Barge-in:** Sustained user speech (> 200 ms with energy above adaptive threshold) cancels in-flight LLM HTTP streaming and active TTS tasks within < 150 ms, flushes the audio frame buffer, and registers `"[interrupted]"` in conversational history.
- **Thinking Filler:** If LLM TTFT exceeds 700 ms, play a natural short acoustic filler ("Hmm...", "One second...") to mask latency.
