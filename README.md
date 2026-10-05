# PersonaPlex Voice Agent Platform

> A production-grade, multi-agent speech-to-speech voice platform designed for **NVIDIA PersonaPlex 7B**.  
> Features an interface designed with **Apple HIG precision** and the **warmth and calm of Anthropic's Claude app** — zero AI tropes.

[![CI](https://github.com/kishorrkumar/orchestration_ai/actions/workflows/ci.yml/badge.svg)](https://github.com/kishorrkumar/orchestration_ai/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-Strict-blue.svg)](https://www.typescriptlang.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-v2-green.svg)](https://fastapi.tiangolo.com)
[![Vite](https://img.shields.io/badge/Vite-React_18-purple.svg)](https://vitejs.dev)

---

## 1. Product & Design Philosophy

The platform provides a calm, focused environment where users can create, configure, version, and talk to speech-to-speech agents in real time.

- **Apple HIG Deference:** Clean 1px structural hairlines, crisp tactile controls, predictable tab ordering, and zero decorative noise.
- **Claude Editorial Warmth:** Warm stone and linen canvas (`#FAF9F5` light, `#1A1918` dark), generous typography line height, and honest microcopy.
- **Anti-AI Design Discipline:** Zero glowing neon orbs, zero purple-blue gradient soup, zero sparkle emojis, and zero cartoon chat bubbles. Spoken audio is represented by a calm, monochrome concentric circle scaling subtly with live speech RMS energy.

Visit the **Living Style Guide** at [`http://localhost:8000/design-system`](http://localhost:8000/design-system).

---

## 2. Core Architecture

The codebase enforces strict boundary separation across five decoupled layers:

```
orchestration/
├── domain/                  # 1. Pure Domain Layer (Entities, Rules, Protocols)
│   ├── agent.py             # Agent aggregate (6 lean fields + 18 presets)
│   ├── session.py           # CallSession and CallTurn entities
│   ├── prompt.py            # System prompt compilation & voice linter
│   └── detector.py          # EndOfCallDetector state machine & quiet-window drain
│
├── application/             # 2. Use Cases Layer
│   ├── agents/              # Create, Update, Publish, Revert use cases
│   ├── calls/               # Call session persistence & turn logging
│   └── prompts/             # PromptCompilerUseCase with token limit enforcement
│
├── infrastructure/          # 3. Adapters & Persistence
│   ├── db/repositories/     # SQLAlchemy 2.0 async repositories (SQLite & Neon Postgres)
│   ├── clock/               # SystemClock and FrozenClock (deterministic time testing)
│   └── tokenizer/           # SentencePieceTokenizerAdapter
│
├── interfaces/              # 4. HTTP & WebSocket Delivery
│   └── http/                # REST endpoints, Pydantic v2 DTOs, RFC 9457 Problem Details
│
└── shared/                  # 5. Cross-Cutting Utilities
    ├── errors.py            # DomainError hierarchy + RFC 9457 ProblemDetail
    ├── logging.py           # Structlog with correlation IDs & credential redaction
    ├── settings.py          # Pydantic-settings 12-factor configuration
    └── ids.py               # Type-safe prefixed ID generators (agt_, ses_, ver_, trn_)
```

For complete technical specifications, see:
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — System design and audio DSP pipelines.
- [`docs/HOW_THE_MODEL_WORKS.md`](docs/HOW_THE_MODEL_WORKS.md) — PersonaPlex 7B dual-decoder architecture.
- [`docs/PROMPTING_GUIDE.md`](docs/PROMPTING_GUIDE.md) — S2S prompting rules, token budgets, and 5 copy-paste templates.
- [`docs/VOICE_PRESETS.md`](docs/VOICE_PRESETS.md) — Catalog of all 18 official upstream voice conditioning embeddings.
- [`docs/adr/`](docs/adr/) — Architecture Decision Records.

---

## 3. The 6-Field Lean Agent Model

Every agent is defined by six focused, intentional fields:

| Field | Description | Invariant & Validation |
| :--- | :--- | :--- |
| **Name** | Display name | Also interpolates into `{{agent_name}}`. |
| **Voice** | Voice conditioning preset | One of 18 official PersonaPlex embeddings (`NATF0.pt` – `NATM8.pt`). |
| **Greeting** | Spoken opening line | Delivered automatically when `agent_speaks_first` is enabled. |
| **System Prompt** | Spoken persona instructions | Wrapped as `<system> {prompt} <system>`. Enforces `<135` token budget for fast start. |
| **Ending** | Call concluding phrase | Triggers the `EndOfCallDetector` state machine with 1.5s quiet window drain. |
| **Timezone** | Caller timezone | Calculates dynamic local time string (supports half-hour offsets like `Asia/Kolkata`). |

---

## 4. Quickstart Guide

### 4.1 Prerequisites
- Python 3.11+
- Node.js 18+ and npm
- Windows PowerShell, macOS, or Linux bash

### 4.2 Using the Universal Task Runner (`run.ps1`)

The repository includes a PowerShell-native task runner for common developer workflows:

```powershell
# 1. Install dependencies
.\scripts\run.ps1 setup

# 2. Seed database with starter agents
.\scripts\run.ps1 seed

# 3. Start local platform (Gateway + React UI + Auto-Mock Worker)
.\scripts\run.ps1 dev
```

Open [`http://127.0.0.1:8000/`](http://127.0.0.1:8000/) in your browser.

### 4.3 Running Quality Gates

```powershell
# Run all gates (Ruff linter + Mypy strict + Pytest suite)
.\scripts\run.ps1 check

# Or individual gates:
.\scripts\run.ps1 lint
.\scripts\run.ps1 typecheck
.\scripts\run.ps1 test
```

---

## 5. Docker Deployment

Deploy the entire platform with Docker Compose:

```bash
# Build and launch gateway and mock inference worker
docker compose up --build -d

# View live gateway logs
docker compose logs -f gateway
```

---

## 6. Real-Time Audio DSP Pipeline

- **Web Clients (16 kHz):** Browser microphone streams 16 kHz Linear PCM 16-bit mono over WebSocket (`/v2/voice`).
- **Model Invariant (24 kHz):** Polyphase FIR resampling via `scipy.signal.resample_poly` with exact 3:2 rational ratio into 480-sample frames (20ms) for Moshi Mimi.
- **Telephony Invariant (8 kHz):** Hardware-accelerated G.711 μ-law companding with algebraic soft-clipping to prevent speaker distortion.
- **Acoustic Barge-in:** When caller speech energy exceeds $RMS > 0.012$, the gateway dispatches `ControlAction.PAUSE` to immediately silence assistant playback.

---

## 7. License

Licensed under the Apache License, Version 2.0. See [`LICENSE`](LICENSE) for details.
