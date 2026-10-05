# PersonaPlex Orchestration AI: Comprehensive Architecture & System Specification

> **Target Audience:** LLMs (specifically Anthropic Claude), systems architects, and engineers requiring an exhaustive, complete ground-truth understanding of this codebase.

---

## 1. Executive Summary & Purpose

**PersonaPlex Orchestration AI** is a production-grade conversational orchestration platform built for **NVIDIA PersonaPlex 7B** (a full-duplex speech-to-speech foundation model built on Kyutai Moshi architecture and the Mimi neural audio codec).

Unlike traditional cascaded voice bots (ASR -> LLM -> TTS), PersonaPlex processes audio tokens end-to-end at 24 kHz in real time (80 ms frames, 12.5 Hz), allowing simultaneous listening and speaking, instantaneous barge-in interruptions, and human-like backchanneling.

Because native PersonaPlex requires an NVIDIA GPU with 18–20 GB VRAM and is strictly English-only, this orchestration framework provides:
1. **GPU-Accelerated Speech-to-Speech Core:** Full-duplex WebSocket gateway managing worker pools, session lifecycle, and binary framing (`0x00`–`0x06`).
2. **Local Cascaded Fallback Pipeline:** Modular ASR (Faster-Whisper), Local LLM (Ollama / Qwen 2.5), and TTS (Kokoro / EdgeTTS) for multilingual (specifically Indian English / Indic languages) and low-resource CPU operation.
3. **Auto Mock Worker:** Automatic local loopback worker (`127.0.0.1:8998`) for development and testing without any GPU or external cloud dependencies.
4. **Interactive Developer Web Console:** Apple Design System glassmorphism UI with live Siri Orb visualization, real-time telemetry, 18-voice catalog, and zero-shot voice cloning.
5. **Acoustic Audio Pipeline:** Client-side AEC (`echoCancellation: true`), headphone loopback detection, server-side 80 Hz high-pass filtering, RNNoise neural denoise, and Silero VAD v6 voice isolation.

---

## 2. Verified Ground Truths & Hardware Constraints

These specifications are derived directly from upstream inspection (`NVIDIA/personaplex` and `kyutai-labs/moshi`):

| Specification | Ground Truth | Engineering Implication |
| :--- | :--- | :--- |
| **Language Support** | **Strictly English-Only** | Trained on Fisher English (`LDC2004T19`) with a 32,000-token SentencePiece English model (`tokenizer_spm_32k_3.model`). Non-English input causes phonetic hallucinations. Non-English or Indic tasks must route to the cascaded fallback. |
| **Acoustic Codec** | **Mimi Neural Codec (Kyutai)** | Operates at **24,000 Hz mono**, 12.5 frames per second. Each frame is exactly **1,920 audio samples** (80.0 ms). Quantized into 8 codebooks. |
| **Worker Concurrency** | **1 Stream per Worker Process** | Upstream `moshi.server` enforces `async with self.lock:`. A single worker process can serve exactly one active conversation stream. Concurrency requires multi-worker clustering on separate loopback ports (`8998`, `8999`, etc.). |
| **VRAM Consumption** | **~18 to 20 GB VRAM per Worker** | Model weights: ~14.5 GB (BF16); KV-cache & activations: ~3.5 GB. 24 GB GPU (A10G, RTX 3090/4090) hosts **1 worker**; 48 GB GPU (A40, L40) hosts **2 workers**; 80 GB GPU (A100, H100) hosts **3–4 workers**. |
| **System Prompt Delimiter** | `<system> {prompt} <system>` | Exact delimiters: both open and close tags are `<system>` (literally `<system>`, **not** `</system>`). Stepped autoregressively one token per 80ms frame during initialization. |
| **System Prompt Length** | **Max 350 tokens (ideal < 150)** | Each prompt token takes ~15–25ms to autoregressively step before handshake. Prompts > 350 tokens delay session startup by >7 seconds, risking browser WebSocket timeouts. |
| **Voice Conditioning** | **18 Presets (`.pt`) or 5–12s `.wav`** | 18 built-in `.pt` embeddings. Custom `.wav` references must be 24 kHz mono normalized to -24 LUFS. Recommended 5–12 seconds (~8s optimal). References > 30s cause timeout during handshake. |

---

## 3. High-Level System Architecture

```mermaid
flowchart TD
    User["Web Browser Client (Mic + Speaker)"] -->|WSS / HTTPS| ReverseProxy["Caddy / Nginx Reverse Proxy (Port 443)"]
    ReverseProxy -->|HTTP / WS| Gateway["FastAPI Orchestration Gateway (Port 8000)"]
    
    subgraph GatewayComponents ["FastAPI Gateway Core (orchestration/gateway)"]
        Gateway --> Auth["API Key & Token Bucket Rate Limiting"]
        Gateway --> Router["REST Endpoints (/v1/agents, /v1/voices, etc.)"]
        Gateway --> WSHandler["WebSocket Wire Handler (/v1/realtime)"]
        Gateway --> DevConsole["Built-in Developer Console (/console)"]
    end
    
    subgraph SessionAndAudio ["Session & Audio Processing (orchestration/session & audio)"]
        WSHandler --> SM["Session State Machine (7 Quality Dimensions)"]
        WSHandler --> Chunker["Audio Chunker (1,920 samples @ 24 kHz)"]
        WSHandler --> Cleaner["CallerAudioCleaner (80Hz HPF + RNNoise + Silero VAD)"]
        WSHandler --> TurnDet["Turn Detector & Barge-in Monitor (<200ms cutoff)"]
    end

    subgraph WorkerLayer ["Inference Worker Pool (orchestration/worker)"]
        WSHandler --> Pool["Worker Pool Manager"]
        Pool -->|Binary Opcode Stream (ws://127.0.0.1:8998)| MoshiWorker["PersonaPlex 7B Worker (GPU A100/A10G)"]
        Pool -->|Binary Opcode Stream (ws://127.0.0.1:8998)| MockWorker["PersonaPlexMockServer (Zero-GPU Fallback)"]
        Pool -->|Internal Orchestration| LocalCascade["Local Cascaded Worker (ASR + Ollama + TTS)"]
    end
```

---

## 4. Repository & File Structure

```
orchestration_ai/
├── config.yaml                    # System configuration (gateway, personaplex, audio, fallback)
├── pyproject.toml                 # Single source of truth for build, tools, ruff, mypy, pytest
├── Dockerfile                     # Multi-stage production container build
├── docker-compose.yml             # Local orchestrated stack (Gateway + Mock Worker)
├── scripts/
│   ├── run.ps1                    # Universal PowerShell task runner (setup, dev, test, check, seed)
│   └── fetch_upstream.py          # Pinned upstream checkout (commit 3428dfd95309a7f3c84fd93259ded0f810d1ff91)
│
├── frontend/                      # Modern React 18 + Vite + Tailwind v4 Web Application
│   ├── src/
│   │   ├── components/ui/         # Apple HIG / Claude UI primitives (Button, Badge, AudioIndicator, etc.)
│   │   ├── views/                 # AgentsListView, AgentEditorView, CallsHistoryView, DesignSystemView, TestCallModal
│   │   ├── lib/                   # API client, types, classname utilities
│   │   └── index.css              # Design tokens (--color-bg-base, terracotta, tabular numbers)
│   └── dist/                      # Pre-compiled static assets served by FastAPI gateway
│
├── orchestration/                 # Layered Core Python Architecture
│   ├── domain/                    # Pure Domain Layer (Agent, Version, Session, Delimiter, Protocols)
│   │   ├── agent.py               # 6-field lean agent aggregate & 18 presets validation
│   │   ├── session.py             # CallSession and CallTurn entities
│   │   ├── prompt.py              # <system> {prompt} <system> compiler & voice linter
│   │   ├── detector.py            # EndOfCallDetector state machine & quiet-window drain
│   │   └── protocols.py           # Abstract structural contracts
│   │
│   ├── application/               # Use Cases & Application Orchestration
│   │   ├── agents/                # CreateAgent, UpdateAgent, PublishVersion, RevertVersion
│   │   ├── calls/                 # StartSession, RecordTurn, EndSession
│   │   └── prompts/               # PromptCompilerUseCase with token limit enforcement
│   │
│   ├── infrastructure/            # Adapters & Persistence
│   │   ├── db/repositories/       # SQLAlchemy 2.0 async repositories (SQLite & Neon Postgres)
│   │   ├── clock/                 # SystemClock & FrozenClock for deterministic testing
│   │   └── tokenizer/             # SentencePieceTokenizerAdapter
│   │
│   ├── interfaces/                # HTTP Controllers & Wire Handlers
│   │   └── http/                  # REST routers (/v2/agents, /v2/calls, /v2/prompts, /healthz, /readyz)
│   │                              # RFC 9457 Problem Details error handlers
│   │
│   ├── shared/                    # Cross-Cutting Shared Modules
│   │   ├── errors.py              # DomainError hierarchy + ProblemDetail RFC 9457 model
│   │   ├── logging.py             # Structlog with correlation IDs & credential redaction
│   │   ├── settings.py            # Pydantic-settings 12-factor configuration
│   │   └── ids.py                 # Prefixed ID generators (agt_, ses_, ver_, trn_, err_)
│   │
│   ├── audio/                     # 16 kHz Linear PCM, 24 kHz polyphase resampling, G.711 μ-law
│   ├── db/                        # Models, session engine, seeding
│   ├── gateway/                   # FastAPI application & server assembly
│   ├── worker/                    # WorkerPool, MockWorker, Upstream Client
│   └── dormant/                   # Preserved dormant modules (RAG, Voice Cloning, Cascaded Fallback)
│
├── deploy/                        # Deployment Artifacts
│   ├── cloud/                     # Automated VM provisioning scripts (deploy_krutrim.sh, start_services.sh)
│   ├── proxy/                     # Caddyfile and Nginx reverse proxy configurations
│   └── systemd/                   # Systemd service units
│
└── docs/                          # Architecture Documentation & Specifications
    ├── ARCHITECTURE.md            # System design and audio DSP specifications
    ├── HOW_THE_MODEL_WORKS.md     # PersonaPlex 7B dual-decoder architecture
    ├── VOICE_PRESETS.md           # 18 official upstream voice conditioning embeddings
    ├── adr/                       # Architecture Decision Records (0001 - 0005)
    └── verified_facts.md          # Upstream empirical ground truths
│
└── tests/                         # Comprehensive Automated Test Suite (75+ tests)
    ├── test_gateway.py            # Gateway REST API & WebSocket tests
    ├── test_e2e.py                # End-to-end full duplex conversation simulation
    ├── test_chunker.py            # Audio framing and chunking tests
    └── test_audio_cleaner.py      # RNNoise & VAD noise reduction tests
```

---

## 5. Binary WebSocket Wire Protocol

The WebSocket connection operates over `/v1/realtime` (Gateway) and `/api/chat` (Upstream Worker). Every binary packet begins with a **1-byte opcode**:

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  Opcode (1B)  |               Payload Data ...               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

| Opcode | Hex | Name | Direction | Payload Specification |
| :---: | :---: | :--- | :--- | :--- |
| `0` | `0x00` | **Handshake** | Server -> Client | 2 bytes: Protocol version (1 byte) + Model ID (1 byte). Dispatched once voice & system prompts finish autoregressive stepping. |
| `1` | `0x01` | **Audio Frame** | Bidirectional | Linear 16-bit signed PCM, 24,000 Hz, mono. Exactly 1,920 samples = 3,840 bytes per packet (or Opus-compressed stream packets). |
| `2` | `0x02` | **Text Token** | Bidirectional | UTF-8 encoded text token string. Upstream converts special prefix `▁` to a whitespace character. |
| `3` | `0x03` | **Control Signal** | Bidirectional | 1 byte control code: `0=Start`, `1=EndTurn`, `2=Pause`, `3=Restart`, `4=BargeInInterrupt`. |
| `4` | `0x04` | **Metadata** | Bidirectional | UTF-8 encoded JSON object (session info, RAG context, latency metrics). |
| `5` | `0x05` | **Error** | Server -> Client | UTF-8 encoded human-readable error string. Server immediately closes connection afterwards. |
| `6` | `0x06` | **Ping / Heartbeat**| Bidirectional | 4-byte big-endian timestamp or sequence counter for keep-alive and RTT calculation. |

---

## 6. The 7 Conversation Quality Dimensions

The orchestration layer continuously instruments and enforces seven quality dimensions:

1. **Listening (Input Ingestion):**
   - Full-duplex 24 kHz capture.
   - Continuous bit-exact 1,920-sample framing (zero dropped frames).
   - Silero VAD v6 energy and speech probability estimation.
2. **Understanding (Real-Time Transcription):**
   - Real-time text token emission via `0x02`.
   - Dual-channel transcript recording (User turn vs Agent turn).
   - Exportable transcript REST endpoint (`GET /v1/sessions/{id}/transcript`).
3. **Reasoning (Persona Conditioning & RAG):**
   - System prompts wrapped in `<system> {prompt} <system>` delimiters.
   - Injected RAG context documents formatted compactly to fit token budgets (<350 tokens).
4. **Speaking (Acoustic Quality):**
   - 24 kHz acoustic delivery, normalized to -24.0 LUFS.
   - Click-free crossfading and 25 ms jitter buffer to eliminate audio underruns.
5. **Latency (Responsiveness Target):**
   - Target Time-To-First-Audio (TTFA) < 300 ms p50 after user turn completion.
   - Frame processing latency target < 80 ms.
   - Real-time telemetry exposed via Prometheus format at `/metrics`.
6. **Conversation (Barge-In & Turn Taking):**
   - User speech detection cuts off agent audio playback within <200 ms.
   - Control packet `0x03 (BargeInInterrupt)` flushes queued audio buffers.
   - Tolerant of conversational backchannels ("uh-huh", "yeah", "hmm").
7. **Task Success (Outcome Classification):**
   - Per-session outcome tagging: `in_progress`, `completed`, `interrupted`, `failed`.
   - Stored in session database with call duration and turn count.

---

## 7. Acoustic Cleaning & Echo Cancellation

### 1. Browser-Side Acoustic Echo Cancellation (AEC)
Because full-duplex agents talk while listening, speaker playback leaking into the microphone creates an acoustic feedback loop.
- The web console (`studio_ui.py` / `frontend.html`) strictly requests browser AEC:
  ```javascript
  navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      sampleRate: 24000,
      echoCancellation: true,    // Enforces OS/WebRTC hardware acoustic echo cancellation
      noiseSuppression: true,    // Enforces hardware background noise suppression
      autoGainControl: true
    }
  });
  ```
- **Headphone Detection:** The UI scans `navigator.mediaDevices.enumerateDevices()` for output devices. If open laptop speakers are detected without headphones, a warning banner appears (`#headphone-warning-banner`).

### 2. Server-Side Audio Cleaner (`orchestration/audio/cleaner.py`)
- **80 Hz High-Pass Filter:** Strips low-frequency mechanical rumble, AC hum, and table thumps.
- **RNNoise (48 kHz C RNN):** Deep recurrent neural network denoiser removing stationary room noise and fan hiss (achieves ~6.17 dB noise floor reduction with 92.7% speech preservation in <2.3 ms).
- **Silero VAD Voice Isolation:** Evaluates speech probability. Emits pure zero frames during non-speech intervals to prevent room echo or background voices from triggering the model.
- **350 ms Hangover Gating:** Prevents clipping natural inter-syllable pauses while delivering an immediate sharp cutoff at turn boundaries.

---

## 8. Built-in Personas & Voice Presets

### The 18 PersonaPlex Voice Presets
PersonaPlex contains 18 official pre-computed acoustic conditioning vectors (`.pt`):
- **Natural Female (4):** `NATF0.pt` (Warm & Calm), `NATF1.pt` (Crisp & Professional), `NATF2.pt` (Expressive Teacher), `NATF3.pt` (Bright & Articulate)
- **Natural Male (4):** `NATM0.pt` (Deep & Authoritative), `NATM1.pt` (Warm Consultative), `NATM2.pt` (Technical & Energetic), `NATM3.pt` (Casual & Direct)
- **Variety Female (5):** `VARF0.pt` (Storyteller), `VARF1.pt` (Presenter), `VARF2.pt` (Analyst), `VARF3.pt` (Serene), `VARF4.pt` (Actor)
- **Variety Male (5):** `VARM0.pt` (Radio Announcer), `VARM1.pt` (Astronaut), `VARM2.pt` (Empathetic), `VARM3.pt` (Tech Host), `VARM4.pt` (Baritone)

### Built-in Agent Personas (`orchestration/persona/registry.py`)
| Persona ID | Name | Gender | Default Voice | Description | Backend Engine |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `support_agent` | Alex | Female | `NATF1.pt` | Concise, empathetic tier-1 technical support engineer. | PersonaPlex 7B |
| `wise_teacher` | Dr. Elena | Female | `NATF2.pt` | Patient academic mentor using vivid analogies. | PersonaPlex 7B |
| `sales_caller` | Marcus | Male | `NATM1.pt` | Energetic, consultative enterprise sales specialist. | PersonaPlex 7B |
| `casual_friend` | Sam | Male | `NATM0.pt` | Relaxed peer using informal language and natural reactions. | PersonaPlex 7B |
| `indian_pro` | Aarav | Male | `NATM0.pt` | Articulate Indian English corporate professional. | Local Cascaded Fallback |
| `indian_priya` | Priya | Female | `NATF0.pt` | Warm, colloquial Indian English customer relationship manager. | Local Cascaded Fallback |

---

## 9. Zero-Shot Voice Cloning Pipeline

Located in `orchestration/tts/voice_clone.py`:
1. **Audio Ingestion:** Accepts WAV, MP3, M4A, or OGG (5 to 12 seconds recommended; minimum 3.0s).
2. **Signal Conditioning:** Decodes via `ffmpeg`, resamples to 24,000 Hz mono, normalizes to -24.0 LUFS, trims trailing silence.
3. **Signal Quality Gate:** Checks clipping ratio (<5%) and non-silent frames (>5%).
4. **Speaker Similarity:** Measures cosine similarity using acoustic mel-filterbanks / Resemblyzer against the reference audio (minimum threshold `0.65`).
5. **Mandatory Consent Gate:** Requires explicit `consent=true` in the API call; unauthorized cloning attempts fail with HTTP 400.

---

## 10. Frontends & User Interfaces

This repository includes two frontend solutions:

### 1. Developer Studio UI (Primary)
- **File:** [`orchestration/gateway/studio_ui.py`](file:///c:/Users/kisho/Desktop/orchestration_ai/orchestration/gateway/studio_ui.py) (served directly by FastAPI at `/console` and `/`)
- **Offline HTML:** [`frontend.html`](file:///c:/Users/kisho/Desktop/orchestration_ai/frontend.html)
- **Features:**
  - Siri Apple Intelligence fluid multi-color glowing canvas orb.
  - Live audio waveform and volume visualizer.
  - Persona switching cards and 18-voice preset drawer.
  - Zero-shot voice cloning studio with mic recording and consent checkbox.
  - Full-duplex conversational chat transcript bubbles with barge-in interruption badges.
  - Telemetry HUD: TTFA latency, packet counters, and GPU stats.
  - RAG document drag-and-drop uploader.

### 2. Upstream React/Vite Client
- **Directory:** [`_personaplex_upstream/client`](file:///c:/Users/kisho/Desktop/orchestration_ai/_personaplex_upstream/client)
- **Tech Stack:** React 18, Vite, TypeScript, TailwindCSS, DaisyUI.
- **Run command:** `cd _personaplex_upstream/client && npm run dev` (Runs on `http://localhost:5173`).

---

## 11. REST API Specification

### Authentication & Rate Limiting
- Pass API Key via header `X-API-Key: <token>` (Configured in `config.yaml` or `API_KEY` env var).
- Token-bucket rate limiting default: 60 requests/minute per IP.

### Core Endpoints

#### Gateway & System
- `GET /health` — Health status of gateway and connected inference workers.
- `GET /metrics` — Prometheus metrics (sessions, latency TTFA p50/p95, frame drops).
- `GET /console` — Interactive Developer Web Console HTML page.

#### Agents & Personas
- `GET /v1/agents` — List all registered personas.
- `GET /v1/agents/{id}` — Get persona configuration and system prompt.
- `PUT /v1/agents/{id}` — Update persona prompt or default voice.

#### Voices & Cloning
- `GET /v1/voices` — List all 18 official presets and cloned custom voices.
- `GET /v1/voices/{voice_id}/preview` — Download WAV preview audio sample.
- `POST /v1/voices/clone` — Multipart upload for zero-shot voice cloning:
  - `audio`: File (WAV/MP3/M4A)
  - `voice_name`: string
  - `gender`: "male" | "female"
  - `consent`: "true"

#### Sessions & Telemetry
- `POST /v1/sessions` — Initialize a new conversation session.
- `GET /v1/sessions/{id}` — Session status, quality metrics, and duration.
- `GET /v1/sessions/{id}/transcript` — Full dual-channel conversational transcript.
- `DELETE /v1/sessions/{id}` — Force terminate a session and release worker lock.

#### RAG (Retrieval-Augmented Generation)
- `POST /v1/rag/upload` — Upload document (PDF, TXT, MD, CSV) to current session.
- `GET /v1/rag/query` — Query semantic embeddings for top-$k$ context chunks.

---

## 12. Deployment & Execution Guide

### Local Development (Zero GPU Required)
On Windows or macOS without an NVIDIA GPU, run the gateway with the auto-mock worker:
```powershell
# In PowerShell (using .venv-gpu which contains all dependencies):
.\.venv-gpu\Scripts\python.exe -m uvicorn orchestration.gateway.app:create_app --factory --host 127.0.0.1 --port 8000
```
Open **`http://localhost:8000/console`** in your browser. The mock worker (`127.0.0.1:8998`) will automatically handle the WebSocket stream.

### Cloud GPU Deployment (Krutrim Cloud / Ubuntu 22.04 LTS)
Automated VM deployment with an NVIDIA A100 SXM4 (40GB/80GB):
```bash
# 1. Clone repository and set Hugging Face token
export HF_TOKEN="your_huggingface_token"

# 2. Run automated provisioning
chmod +x deploy_krutrim.sh
./deploy_krutrim.sh

# 3. Start services
chmod +x start_services.sh
./start_services.sh
```

### Multi-Worker Cluster (Large GPUs)
On 80 GB GPUs (A100 / H100), launch 3 concurrent worker processes:
```bash
chmod +x scripts/run_cluster.sh
./scripts/run_cluster.sh --workers 3 --base-port 8998 --gpu 0
```

---

## 13. Testing & Validation

The test suite validates the binary protocol, session state machine, rate limits, audio cleaners, and voice cloning:
```bash
# Run full automated test suite (75+ tests):
pytest tests/ -v
```

---

## 14. Key Architectural Decisions (ADR Summary)

1. **Why Moshi / PersonaPlex over Cascaded Pipelines for English?**
   - Cascaded pipelines suffer from turn-taking latency (ASR: 200ms + LLM TTFT: 300ms + TTS: 250ms = ~750ms+). PersonaPlex achieves <300ms latency with natural barge-in interruptions.
2. **Why Local Cascade Fallback for Indic/Multilingual?**
   - PersonaPlex has zero training on non-English tokens. Hallucination cannot be fixed by prompt engineering. The honest architecture routes Indic/Hindi to Faster-Whisper + Ollama + Kokoro/EdgeTTS.
3. **Why Loopback Worker Pool?**
   - Since `moshi.server` holds a global lock per process, scaling concurrency requires running multiple Python processes bound to `127.0.0.1:8998`, `8999`, etc., with the FastAPI gateway acting as the reverse-proxy dispatcher.
4. **Why Client-Side AEC?**
   - The browser has access to the physical microphone hardware and OS audio graph, allowing zero-latency acoustic echo subtraction before transmitting audio packets over WebSocket.

---

## 15. Lean S2S Voice Agent Platform (2026 Upgrade)

The platform was upgraded to a focused, production-grade **Lean S2S Voice Agent Platform** for **NVIDIA PersonaPlex 7B**, adhering to strict audio engineering principles and an Apple/Linear design system.

### 15.1 Six-Field Lean Agent Architecture
Each agent is defined by exactly 6 essential fields (no unnecessary RAG, tools, or complex webhooks in the primary path):
1. **Name**: Display name, dynamically injected into prompts as `{{agent_name}}`.
2. **Voice**: One of the 18 official PersonaPlex `.pt` embeddings (`orchestration/persona/presets.py`).
3. **Greeting**: First spoken words with `agent_speaks_first` toggle.
4. **System Prompt**: Core instructions formatted with `<system> {prompt} <system>`, supporting dynamic variables (`{{time}}`, `{{date}}`, `{{timezone}}`).
5. **Ending**: Final phrase spoken upon call completion, paired with `end_call_timeout_sec` and `silence_timeout_sec`.
6. **Timezone**: Canonical IANA timezone identifier (e.g., `UTC`, `America/New_York`, `Asia/Kolkata`).

### 15.2 Database & Immutable Versioning
- **Models** (`orchestration/db/models.py`):
  - `agents`: Primary record with UUID primary key.
  - `agent_versions`: Immutable version snapshots (`v1`, `v2`, ...). Updating an agent automatically clones a new version, ensuring active calls never mutate in-flight.
  - `call_sessions`: High-level call records with status (`in_progress`, `completed`, `failed`), end reason, and duration.
  - `call_turns`: Chronological turn-by-turn logs with speaker tag, text transcript, and latency telemetry.
- **Dual-Engine Persistence** (`orchestration/db/session.py`):
  - Local development: SQLite via `aiosqlite` at `data/platform.db`.
  - Cloud production: Neon Serverless PostgreSQL via `asyncpg` (`DATABASE_URL=postgresql+asyncpg://...`).

### 15.3 Prompt Compilation & SentencePiece Token Meter
- **Delimiters**: `<system> {prompt} <system>` (both tags are `<system>`, verified from upstream `moshi.server`).
- **Token Counter**: Uses `models/tokenizer_spm_32k_3.model` to compute exact BPE token counts. Hard threshold: 350 tokens; recommended: $\le 150$ tokens.
- **Voice Linter**: Flags written conventions that sound unnatural when spoken by PersonaPlex (URLs, markdown formatting, parentheticals, ALL-CAPS acronyms, bullet points).

### 15.4 16 kHz Audio Core & Telephony Readiness
- **Bit-Exact G.711 μ-law** (`orchestration/audio/codecs.py`): Pure NumPy ITU-T Recommendation G.711 encoder and decoder with $O(1)$ table lookups.
- **Anti-Aliasing Resampling** (`orchestration/audio/resample.py`): `AudioResampler` and `StreamingResampleBuffer` backed by libsoxr (`soxr.ResampleStream`) providing >82 dB SNR and <0.2 ms delay. Output is framed into exact 480-sample 20ms frames matching Opus/PersonaPlex requirements.
- **DSP Soft Limiter** (`orchestration/audio/dsp.py`): Non-linear hyperbolic tangent (`tanh`) compressor preventing harsh clipping or DAC distortion.

### 15.5 S2S Runtime (`/v2/voice`)
- **Protocol**: Binary WebSocket stream supporting 16 kHz PCM (Web) and 8 kHz G.711 μ-law (Telephony).
- **Turn Logging**: Automatically captures and records user speech and agent responses to `call_turns`.
- **End-of-Call Detection** (`orchestration/pipeline/end_detector.py`): Detects goodbye phrases, silence timeouts, and maximum duration, triggering graceful hangup.

### 15.6 React 18 Lean Studio UI
- Accessible at `http://localhost:8000/` and `http://localhost:8000/studio`.
- Features dark glassmorphism, 18-voice catalog with audio preview players, live SentencePiece token meter, real-time voice linter alerts, and an interactive voice call overlay with microphone capture and a Siri-inspired glowing audio orb.
- Legacy developer console preserved at `/console/legacy`.

