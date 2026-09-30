# Codebase Audit: Architecture Flaws & Limitations

**Audit Date**: September 29, 2026  
**System Target**: Windows 11 Home • Intel Core i7-11800H • NVIDIA GeForce RTX 3050 Ti Laptop GPU (4,096 MiB VRAM) • 16 GB RAM • Free Disk: ~7.7 GB  
**Primary Goal**: Transform Aarav into a genuinely conversational, natural Indian English voice agent running 100% locally with open-source models, zero cloud APIs, and zero canned scripts.

---

## 1. System & Environment Profile

| Metric | Measured Value | Constraint Implication |
| :--- | :--- | :--- |
| **OS** | Windows 11 Home (x86_64, build 26100) | PowerShell execution, Windows path conventions, no POSIX assumptions |
| **CPU** | Intel Core i7-11800H @ 2.30 GHz (8 cores / 16 threads) | High multithreaded capacity for CPU inference (ASR, TTS, VAD) |
| **System RAM** | 16 GB physical (15.77 GB visible, ~2.0 GB currently free) | Comfortable for host memory, but must avoid models > 8 GB RAM footprint |
| **GPU Model** | NVIDIA GeForce RTX 3050 Ti Laptop GPU | Ampere architecture (Compute Capability 8.6), FP16 tensor cores |
| **GPU VRAM** | **4,096 MiB (4.0 GB) GDDR6** | **Strict Hard Bottleneck**: Full PersonaPlex (~7B) or large models crash with CUDA OOM |
| **GPU Driver / CUDA** | Driver 610.62 • CUDA 13.3 driver / PyTorch CUDA 12.x | Full CUDA acceleration supported in PyTorch and CTranslate2 |
| **Free Disk Space (C:)** | **7.76 GB Free** | **Strict Storage Limit**: Models must be compact (Q4_K_M quantizations, small/distil ASR) |
| **Ollama Service** | Installed & running at `http://127.0.0.1:11434` | Available locally. Loaded models: `qwen2.5:1.5b` (986 MB), `llama3.2:1b` (1.3 GB) |
| **Python Runtimes** | `.venv` (Python 3.14) & `.venv-gpu` (Python 3.11.15) | Python 3.11 in `.venv-gpu` contains valid compiled wheels for PyTorch CUDA & ONNX |

---

## 2. In-Depth Flaw Analysis with File & Line References

### Flaw 1: Canned, Scripted Replies Replacing Real Dialogue
* **Location 1**: [orchestration/worker/mock_worker.py:64-70](file:///c:/Users/kisho/Desktop/orchestration_ai/orchestration/worker/mock_worker.py#L64-L70) & [orchestration/worker/mock_worker.py:230-235](file:///c:/Users/kisho/Desktop/orchestration_ai/orchestration/worker/mock_worker.py#L230-L235)
  * `DEFAULT_RESPONSES` cycles through generic scripted phrases:
    * `"Namaste, this is Aarav. How can I assist you with your query today?"`
    * `"I understand you need support. How can I assist you today?"`
    * `"I am processing your request. Could you please specify the details?"`
    * `"Thank you for reaching out. Let me look into that for you right away."`
* **Location 2**: [orchestration/persona/dialogue.py:51-58](file:///c:/Users/kisho/Desktop/orchestration_ai/orchestration/persona/dialogue.py#L51-L58), [dialogue.py:129-150](file:///c:/Users/kisho/Desktop/orchestration_ai/orchestration/persona/dialogue.py#L129-L150), [dialogue.py:220-223](file:///c:/Users/kisho/Desktop/orchestration_ai/orchestration/persona/dialogue.py#L220-L223)
  * `StrictVoiceDialogueEngine` relies heavily on static regex heuristics and canned template fallbacks:
    * `reply("can you hear me")` $\rightarrow$ hardcoded `"Yes, I hear you loud and clear! How can I help you today?"`
    * Generic fallback: `f"Understood regarding '{clean_stmt}'. How would you like us to proceed on this?"`
  * When the upstream LLM is bypassed, unconfigured, or encounters errors, the agent falls back to rigid robotic templates instead of genuine human conversation.

---

### Flaw 2: The Worker Interface & Concurrency Model
* **Location**: [orchestration/worker/client.py:94-125](file:///c:/Users/kisho/Desktop/orchestration_ai/orchestration/worker/client.py#L94-L125) & [orchestration/worker/pool.py:30-80](file:///c:/Users/kisho/Desktop/orchestration_ai/orchestration/worker/pool.py#L30-L80)
  * `PersonaPlexWorkerClient` assumes upstream workers speak the Moshi/PersonaPlex WebSocket chat protocol at `/api/chat?text_prompt=...&voice_prompt=...`.
  * The worker pool treats each node as an exclusive 1:1 concurrency slot (`WorkerStatus.IDLE` $\rightarrow$ `CONNECTING` $\rightarrow$ `BUSY`).
  * The system previously only supported `--worker-type mock`, `--worker-type cascaded`, or `--worker host:port:gpu`.
  * **Requirement**: Introduce a dedicated, first-class `local_cascade` worker adhering strictly to the `PersonaPlexWorkerClient` WebSocket protocol (0x00 Handshake, 0x01 Audio, 0x02 Text, 0x04 Metadata).

---

### Flaw 3: Premature Turn Termination & Mid-Sentence Cutoff ("Tell me a joke about.")
* **Location**: [orchestration/worker/cascaded_worker.py:453-475](file:///c:/Users/kisho/Desktop/orchestration_ai/orchestration/worker/cascaded_worker.py#L453-L475)
  * End-of-turn detection used a naive raw RMS threshold (`rms > 0.032` or `0.048`) and an aggressive silence hangover of only **450 ms** (`time.time() - last_speech_time > 0.45`).
  * **Why it cuts off**:
    1. Humans naturally pause for 500–800 ms mid-utterance to formulate thoughts (e.g., *"Tell me a joke... [600ms pause] ...about cats"*).
    2. The worker had zero linguistic awareness: it immediately triggered ASR when silence reached 450 ms, transcribing only the first half (*"Tell me a joke about."*).
    3. The transcript ended with a dangling preposition (`about`), but the code had no check for trailing continuation words (`about`, `and`, `the`, `to`, `of`, `with`, `because`, `but`, `so`, `like`, `or`) or incomplete grammatical clauses.
    4. There was no pre-roll buffer or resumption merger: when the user spoke the second half (*"about cats"*), it was treated as an entirely separate turn rather than being merged.

---

### Flaw 4: Phonetic ASR Confusion ("artificial intelligence" -> "artist intelligence")
* **Location**: [orchestration/worker/cascaded_worker.py:481-496](file:///c:/Users/kisho/Desktop/orchestration_ai/orchestration/worker/cascaded_worker.py#L481-L496)
  * Faster-Whisper was invoked with bare parameters:
    ```python
    segments, _ = self._whisper_model.transcribe(audio_16k, language="en", beam_size=1)
    ```
  * **Why it misheard**:
    1. **No Initial Prompt / Domain Context**: Whisper heavily relies on `initial_prompt` to bias its language model toward expected vocabulary, Indian accents, and technical terminology. Without an `initial_prompt`, Indian English pronunciation of "artificial" (`/ɑːrˈtɪfɪʃəl/`) easily collapses to the higher-frequency token "artist".
    2. **No Pre-Roll Buffer**: Audio slicing began precisely when RMS crossed the threshold, frequently clipping the initial 50–100 ms of the opening consonant/vowel.
    3. **No Hot-word / VAD Filtering**: Standard Whisper VAD filter was disabled, allowing ambient noise bursts and mouth clicks to pollute recognition tokens.

---

### Flaw 5: Discarded System Prompts & Missing Multi-Turn Conversation Memory
* **Location 1**: [orchestration/gateway/studio_ui.py:1807-1813](file:///c:/Users/kisho/Desktop/orchestration_ai/orchestration/gateway/studio_ui.py#L1807-L1813)
  * The frontend WebSocket connection (`startCall()`) previously omitted `text_prompt` from the query parameters, discarding whatever prompt the user configured in the UI.
* **Location 2**: [orchestration/worker/cascaded_worker.py:328-341](file:///c:/Users/kisho/Desktop/orchestration_ai/orchestration/worker/cascaded_worker.py#L328-L341)
  * The backend worker hardcoded its own system prompt and ignored `query.get("text_prompt")`.
  * It invoked Ollama via raw text completion (`/api/generate`) with single-turn prompt concatenation (`Caller: ... Aarav: ...`), storing zero conversational history.
  * When the user repeated or clarified a question, the model had no context from preceding turns, causing bizarre replies like *"You're welcome, man. Need something?"*.

---

### Flaw 6: Event-Loop-Blocking Calls Inside Async Coroutines
* **Location 1**: [orchestration/worker/cascaded_worker.py:491-496](file:///c:/Users/kisho/Desktop/orchestration_ai/orchestration/worker/cascaded_worker.py#L491-L496)
  * `self._whisper_model.transcribe(audio_16k, ...)` runs CTranslate2 native C++ inference synchronously on the main asyncio thread. While running (200–700 ms), the event loop is blocked; WebSocket heartbeats and ping frames cannot be handled.
* **Location 2**: [orchestration/tts/base.py:210](file:///c:/Users/kisho/Desktop/orchestration_ai/orchestration/tts/base.py#L210)
  * `self._kokoro.create(text, ...)` runs ONNX runtime neural inference synchronously inside `_synthesize_sync`, freezing the event loop unless explicitly dispatched to a worker thread via `asyncio.to_thread` or a `ThreadPoolExecutor`.
* **Location 3**: [orchestration/worker/mock_worker.py:180-181](file:///c:/Users/kisho/Desktop/orchestration_ai/orchestration/worker/mock_worker.py#L180-L181)
  * `pyttsx3` offline synthesis called `engine.runAndWait()` directly on the calling thread.

---

### Flaw 7: Tests Passing Only Due to the Mock Worker
* **Location 1**: [tests/test_mock_worker.py:13-56](file:///c:/Users/kisho/Desktop/orchestration_ai/tests/test_mock_worker.py#L13-L56)
  * Tests that `PersonaPlexMockServer` accepts connections and streams dummy frames. It verifies protocol framing, but validates zero ASR, zero LLM reasoning, and zero real acoustic synthesis.
* **Location 2**: [tests/test_e2e.py:25-84](file:///c:/Users/kisho/Desktop/orchestration_ai/tests/test_e2e.py#L25-L84)
  * Generates an artificial 300 Hz sine wave and feeds it to the mock worker on port 9911. The test passes because the mock server blindly outputs pre-computed audio frames regardless of input.
  * Real-world edge cases (barge-in latency, acoustic noise, Indian English phonetic variance, streaming clause boundaries) had no test coverage.

---

### Flaw 8: Console UX and Transcript Glitches
* **Location**: [orchestration/gateway/studio_ui.py:1715-1760](file:///c:/Users/kisho/Desktop/orchestration_ai/orchestration/gateway/studio_ui.py#L1715-L1760) & [studio_ui.py:1840-1865](file:///c:/Users/kisho/Desktop/orchestration_ai/orchestration/gateway/studio_ui.py#L1840-L1865)
  * WebSpeech browser transcriptions and server-side Whisper transcripts competed for the transcript window, creating duplicate or overwriting user bubbles.
  * Token appending lacked word-boundary normalization, occasionally gluing adjacent words together.
  * The UI lacked distinct visual indicators for conversational states (`Listening` vs `Thinking (LLM TTFT)` vs `Speaking (TTS TTFA)`).
  * No push-to-talk mode existed for noisy environments.
  * No diagnostic text-input pathway existed to debug the LLM/TTS pipeline without requiring live microphone speech.

---

## 3. Action Plan for Step 1 through Step 6

1. **Step 0 Benchmarks**: Measure Ollama LLM candidates (Qwen2.5-1.5B vs 3B vs 7B vs Llama-3.2-3B), Faster-Whisper models (base vs small on CPU int8 vs GPU), and Kokoro-82M on this exact hardware. Record in `docs/benchmarks.md` and `docs/decisions.md`.
2. **Step 1 Architecture (`local_cascade`)**: Implement `LocalCascadeWorkerServer` with Silero VAD, intelligent turn detection (trailing preposition hangover extension), `initial_prompt` biased ASR, streaming LLM chat with conversation history, clause chunking, non-blocking executor dispatch, and sub-200ms barge-in.
3. **Step 2 Aarav Persona**: Create `personas/aarav.md`, few-shot examples, Indian English speech normalization, thinking fillers, and full configuration via `config.yaml` and `/v1/agents`.
4. **Step 3 Latency & Doctor**: Measure end-to-end latency (< 1.5s target) and implement `python -m orchestration.cli doctor`.
5. **Step 4 Console UI**: Real-time stage indicators, push-to-talk, diagnostic text input.
6. **Step 5 Testing & Evaluation**: Implement `scripts/eval_dialogue.py` (15 prompts) and comprehensive automated tests.
7. **Step 6 Packaging**: `setup_local.ps1`, `run_local.ps1`, and documentation.
