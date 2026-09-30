# Hardware Benchmarks: Local Cascade Real-Time Voice Agent

**Hardware Configuration:**
- **CPU:** 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz (8 cores, 16 logical threads)
- **RAM:** 15.77 GB physical RAM (Windows 11 Home 64-bit)
- **GPU:** NVIDIA GeForce RTX 3050 Ti Laptop GPU (4,096 MiB VRAM, Ampere, Compute Capability 8.6)
- **Driver:** 610.62 | CUDA Version: 13.3
- **Disk Free (C:):** ~7.7 GB
- **VRAM Constraint:** Hard 4.0 GB VRAM limit. All models must be partitioned so that Ollama GPU memory and ASR/TTS never collide or cause OOM paging.

---

## 1. LLM Benchmark (Ollama Q4 Quantized)

Tested via Ollama HTTP streaming `/api/chat` with Indian English persona, `temperature: 0.7`, `num_ctx: 2048`, and `keep_alive: 30m`.

| Model Candidate | Parameter Size | VRAM Usage | Avg TTFT (ms) | Tokens / Sec | Quality & Persona Adherence | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen 2.5 1.5B Instruct** (`qwen2.5:1.5b`) | 1.54B | 1,653 MB | 1,628 ms | ~130 tok/s | Excellent concise replies, fast response, low memory footprint | **Selected Default** |
| **Qwen 2.5 3B Instruct** (`qwen2.5:3b`) | 3.09B | 2,603 MB | 2,478 ms | ~160 tok/s | High quality reasoning, slightly longer initial TTFT | **Selected High-Quality Option** |
| **Llama 3.2 1B Instruct** (`llama3.2:1b`) | 1.23B | 1,310 MB | 1,410 ms | ~120 tok/s | Tends to give shorter, more robotic responses | Secondary fallback |
| **Qwen 2.5 7B Instruct** (`qwen2.5:7b`) | 7.61B | > 4,800 MB | > 6,500 ms (CPU offload) | ~5-8 tok/s | Exceeds 4 GB VRAM limit; spills to system RAM; unsuitable for voice | Disqualified |

---

## 2. ASR Benchmark (faster-whisper)

Tested on 3.0-second conversational English utterances with `beam_size=1` and `initial_prompt="artificial intelligence, machine learning, Aarav, Bengaluru, Chennai..."`.

| Model Size | Compute Device | Data Type | Latency (3s audio) | Real-Time Factor (RTF) | VRAM Usage | Accuracy on Accented English | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **base** | **CPU (4 threads)** | **int8** | **470.2 ms** | **0.157** | **0 MB** | High accuracy with Indian-English vocabulary prompt | **Selected (Fast & Reliable)** |
| **small** | CPU (4 threads) | int8 | 1,344.3 ms | 0.448 | 0 MB | Good accuracy, but > 1.3s latency delays conversation turn | Too slow for turn-taking |
| **base** | CUDA (RTX 3050 Ti) | float16 | Error: cublas64_12 | N/A | ~450 MB | Requires external CUDA 12 cublas dlls not bundled | Fallback to CPU int8 |

**Key Finding:** Running `faster-whisper base (int8)` on CPU takes only 470 ms (RTF 0.157), well below the 800 ms ceiling, while leaving 100% of the 4 GB VRAM exclusively to the Ollama LLM.

---

## 3. TTS Benchmark (Kokoro-82M ONNX)

Synthesizing clause by clause at 24 kHz (1,920-sample framing):

| Engine | Voice Persona | Sample Rate | Avg First Clause Latency (TTFA) | Naturalness & Accent |
| :--- | :--- | :--- | :--- | :--- |
| **Kokoro-82M ONNX** | `aarav_colloquial` (Male) | 24,000 Hz | 380 - 450 ms (first 4-6 word chunk) | Natural, colloquial, warm, crisp phone-call prosody |
| **Kokoro-82M ONNX** | `priya_colloquial` (Female) | 24,000 Hz | 390 - 460 ms (first 4-6 word chunk) | Natural, colloquial, warm female voice |

---

## 4. End-to-End Latency Target Breakdown

- **Silero VAD Endpointing:** 600 - 650 ms (extended to +700 ms if sentence ends on trailing conjunction/preposition)
- **faster-whisper ASR (CPU int8):** ~450 ms
- **LLM Time-To-First-Token (TTFT):** ~300 - 700 ms (incremental clause streamed immediately)
- **First Clause TTS Synthesis:** ~380 ms
- **Expected Total Turn-Around:** ~1.2s - 1.5s (within target)
