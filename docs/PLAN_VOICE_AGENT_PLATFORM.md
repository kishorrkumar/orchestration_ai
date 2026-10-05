# Voice Agent Platform Upgrade: Comprehensive Implementation Plan (Phase 0)

> **Document Type:** Master Implementation Plan & Architecture Specification  
> **Status:** Approved by User  
> **Project Scope:** Upgrading `orchestration_ai` into an enterprise-grade multi-agent Voice Agent Platform with 16 kHz core audio, dual engine support (PersonaPlex S2S + Cascaded STT-LLM-TTS), PostgreSQL/SQLite database with immutable versioning, Real-time Protocol v2, React 18 Studio UI, and 8 kHz telephony readiness.

---

## 1. Current State Assessment & Gap Analysis

### 1.1 Existing Assets to Preserve (Ground Truths)
1. **PersonaPlex 7B Adapter & Engine:**
   - English-only constraint, 24 kHz mono Mimi codec, 1,920 samples/frame (80 ms), ~18–20 GB VRAM requirement, system prompt limit (≤350 tokens, ideal <150), and strict `<system> {prompt} <system>` delimiter.
   - Retained as a dedicated `SpeechToSpeechProvider` adapter.
2. **Legacy Binary WebSocket Protocol:**
   - `/v1/realtime` with opcodes `0x00`–`0x06` remains functional and tested for existing clients.
3. **Audio Signal Processing:**
   - High-pass filter (80 Hz), RNNoise neural denoiser, and Silero VAD v6 voice isolation in `orchestration/audio/cleaner.py`.
4. **Mock Worker:**
   - `orchestration/worker/mock_worker.py` listening on `127.0.0.1:8998` for local zero-GPU development.
5. **Zero-Shot Voice Cloning & Consent Gate:**
   - `orchestration/tts/voice_clone.py` with duration checking, LUFS normalization, acoustic cosine similarity (threshold 0.65), and explicit `consent=true` enforcement.
6. **RAG & Telemetry:**
   - Document chunking/embedding in `orchestration/rag/` and TTFA / latency tracking in `orchestration/telemetry/`.
7. **Existing Test Suite:**
   - 99 tests in `tests/` covering protocol, chunker, voice clone, VAD, dialogue, and gateway.

### 1.2 Architecture Gaps Addressed by this Upgrade

| Area | Current Implementation | Upgraded Platform Target |
| :--- | :--- | :--- |
| **Audio Core** | 24 kHz fixed in PersonaPlex frames (80 ms / 1,920 samples) | **16 kHz canonical internal format** (20 ms / 320 samples / 640 B) with anti-aliased streaming resamplers (`16k ↔ 24k`, `16k ↔ 8k`) and pure NumPy G.711 μ-law codec. |
| **Audio Transport** | Browser ScriptProcessorNode (deprecated, GC pauses) | **Browser AudioWorklet** for glitch-free 16 kHz capture; modular transports (`BrowserPCM16Transport`, `TwilioMediaStreamsTransport`, `RawMulawTransport`). |
| **Real-Time Protocol** | Binary opcode prefix (`0x00`–`0x06`) | **Protocol v2 (`/v2/voice`)**: JSON control events for text/signals + raw PCM16 binary frames for audio. |
| **Persistence** | In-memory persona registry (`orchestration/persona/registry.py`) | **SQLAlchemy 2.0 + Alembic** with immutable `agent_versions`, session turns, tool traces, recordings, and analytics. Dual DB support: **SQLite** (local dev) and **Neon PostgreSQL** (cloud production). |
| **Pipeline Architecture** | PersonaPlex WebSocket proxy + basic cascaded worker | **Streaming Frame Processor Pipeline** (Pipecat/LiveKit style) with swappable STT, LLM, TTS, and S2S providers, streaming sentence chunker, and dynamic context aggregator. |
| **Turn-Taking & Interruption** | Basic VAD energy threshold | Pluggable **EndOfTurnDetector**, <200 ms cancellation, **spoken text history truncation**, backchannel filtering (e.g. "uh-huh" < 600 ms), and filler speech for slow tools. |
| **Frontend** | Monolithic HTML string (`studio_ui.py`, 121 KB) | **Modern React 18 + Vite + TypeScript + Tailwind + shadcn/ui** with 2-pane Agent Builder, dedicated Greeting & Instructions sections, prompt linter, and real-time Test Panel. Legacy console preserved at `/console/legacy`. |
| **Telephony Readiness** | None | **8 kHz Telephony Preview mode** (real-time 16k → 8k μ-law → 16k loopback) and carrier serialization stubs (Twilio / Exotel / Plivo). |

---

## 2. Target System Architecture & Module Tree

```
orchestration_ai/
├── config.yaml                    # System configuration & provider settings
├── requirements.txt / pyproject.toml
├── deploy_krutrim.sh / start_services.sh
├── neon.ts                        # Neon configuration
│
├── orchestration/                 # Core Python Backend
│   ├── api/                       # Fast-API Routers (v1 REST API)
│   │   ├── __init__.py
│   │   ├── auth.py                # JWT auth, API key hashing & scoped verification
│   │   ├── agents.py              # Agent CRUD, duplicate, publish, rollback, diff
│   │   ├── prompts.py             # Prompt linter, AI improver, token-counter, templates
│   │   ├── voices.py              # Voice preset catalog & consent-gated cloning
│   │   ├── knowledge.py           # RAG document upload, indexing, chunking, testing
│   │   ├── tools.py               # Builtin & webhook tool definitions + test sandbox
│   │   ├── sessions.py            # Session history, dual-channel transcripts, recordings
│   │   └── analytics.py           # Call metrics, latency percentiles, cost, interruptions
│   │
│   ├── db/                        # Database Layer (SQLAlchemy 2.0 + Alembic)
│   │   ├── base.py                # DeclarativeBase, UUID generators, TimestampMixin
│   │   ├── session.py             # Async engine, sessionmaker, SQLite/Postgres factory
│   │   ├── models.py              # Complete relational schema (Agents, Versions, Turns, etc.)
│   │   ├── seed.py                # Database seeder (6 legacy personas + 7 industry templates)
│   │   └── migrations/            # Alembic environment and migration scripts
│   │
│   ├── audio/                     # 16 kHz Audio Core & DSP
│   │   ├── resample.py            # Anti-aliased stateful resampler (soxr / scipy resample_poly)
│   │   ├── codecs.py              # Pure NumPy G.711 μ-law encoder/decoder (no audioop)
│   │   ├── cleaner.py             # 16 kHz 80Hz HPF + RNNoise + Silero VAD v6 isolation
│   │   ├── silero_vad.py          # Native 16 kHz Silero speech probability estimator
│   │   └── turn_detector.py       # EndOfTurnDetector & barge-in monitor (<200 ms latency)
│   │
│   ├── protocol/                  # Wire Protocols & Canonical Constants
│   │   ├── audio.py               # Canonical 16kHz constants (320 samples, 20ms, 640B)
│   │   ├── messages.py            # Legacy binary opcodes (0x00-0x06)
│   │   └── v2.py                  # Protocol v2 JSON event schemas (session.start, metrics.turn, etc.)
│   │
│   ├── pipeline/                  # Pipecat-Style Streaming Frame Pipeline
│   │   ├── frames.py              # Frame types: AudioFrame, TextFrame, InterruptionFrame, etc.
│   │   ├── context.py             # ContextAggregator & history truncation engine
│   │   ├── normalizer.py          # Spoken text normalization (numbers, currency ₹/$, dates, emails)
│   │   ├── chunker.py             # Streaming sentence/clause chunker (4-12 word initial chunk)
│   │   ├── turn_taking.py         # Semantic endpointing, backchannel filter, idle re-prompter
│   │   └── runner.py              # Async pipeline execution coordinator
│   │
│   ├── providers/                 # Swappable, Interface-Driven Providers
│   │   ├── base.py                # STTProvider, LLMProvider, TTSProvider, S2SProvider ABCs
│   │   ├── registry.py            # Provider registry, health tracker, and failover chains
│   │   ├── stt/                   # Faster-Whisper (local), Deepgram, Google stubs
│   │   ├── llm/                   # Ollama (local), OpenAI-compatible, Anthropic, Gemini stubs
│   │   ├── tts/                   # Kokoro (local), Edge-TTS, Piper, ElevenLabs stub
│   │   └── s2s/                   # PersonaPlexAdapter (24k resampler, <350 token budget)
│   │
│   ├── transports/                # Transport & Serialization Adapters
│   │   ├── base.py                # BaseTransport interface
│   │   ├── browser_pcm16.py       # WebSocket v2 (JSON events + raw 16kHz PCM16)
│   │   ├── twilio_mulaw.py        # Twilio Media Streams (JSON + base64 8kHz μ-law) stub
│   │   └── raw_mulaw.py           # Generic 8kHz G.711 μ-law UDP/TCP stub
│   │
│   ├── prompts/                   # Prompt Engineering & Safety
│   │   ├── preamble.py            # Platform voice rules preamble (read-only in UI, opt-outable)
│   │   ├── compiler.py            # Guided sections -> final prompt compiler (with token budgeting)
│   │   ├── linter.py              # Voice-specific prompt linter (markdown, length, questions)
│   │   ├── improver.py            # LLM prompt optimizer with diff generator
│   │   └── templates/             # 7 production-grade industry templates
│   │
│   └── gateway/                   # Application Entrypoint
│       ├── app.py                 # FastAPI factory, v1 REST + v2 WS + legacy v1 WS
│       ├── legacy_console.py      # Preserved legacy studio UI at /console/legacy
│       └── security.py            # Security middlewares and token-bucket rate limiting
│
├── frontend/                      # Modern React 18 + Vite + TypeScript Studio
│   ├── src/
│   │   ├── audio/                 # AudioWorkletProcessor (glitch-free 16kHz capture & playback)
│   │   ├── components/            # shadcn/ui components, Siri Orb, Waveform, Latency Waterfall
│   │   ├── pages/                 # AgentsList, AgentBuilder, Voices, Knowledge, Tools, Calls, Analytics
│   │   ├── stores/                # Zustand state stores (agentStore, testCallStore, settingsStore)
│   │   └── api/                   # TanStack Query hooks for REST and WebSocket v2 client
│   ├── package.json
│   └── vite.config.ts
│
├── docs/                          # Architecture & Operational Documentation
│   ├── TELEPHONY.md               # Complete 8 kHz carrier integration guide (Twilio/Exotel/Plivo)
│   ├── PERF.md                    # Real measured latencies & benchmark metrics on local machine
│   ├── PROMPTING_GUIDE.md         # Spoken prompt design guide with good/bad patterns
│   └── VERIFIED_FACTS.md          # Ground truths documentation
│
└── tests/                         # Comprehensive Unit, Integration & Regression Suite
    ├── test_audio_core.py         # 16kHz math, resamplers SNR, NumPy μ-law golden vectors
    ├── test_db_models.py          # SQLAlchemy models, migrations, versioning, rollback
    ├── test_pipeline_v2.py        # Frame pipeline, barge-in, history truncation, sentence chunker
    ├── test_v2_protocol.py        # /v2/voice WebSocket and transport serializers
    ├── test_prompt_tools.py       # Prompt linter, compiler, token budgeting
    └── ... (retains all existing 99 test cases)
```
