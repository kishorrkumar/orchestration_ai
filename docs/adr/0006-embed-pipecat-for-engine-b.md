# ADR 0006: Embed Pipecat as Framework Core for Engine B

## Context
Engine A (NVIDIA PersonaPlex 7B) provides a full-duplex speech-to-speech foundation model at 24 kHz, but is English-only, requires 18–20 GB VRAM per concurrent stream, and cannot speak deterministic verbatim greetings or endings.

To support multilingual agents (including Tamil and Hindi), CPU/cloud operation, exact scripted greetings/endings, and unlimited concurrency with third-party providers, we require **Engine B: a streaming STT → LLM → TTS cascaded pipeline**.

A production-grade voice loop requires solving complex real-time audio mechanics:
1. Bidirectional WebSocket streaming with frame-level chunking.
2. Low-latency turn detection (native STT end-of-turn vs. VAD + Smart Turn).
3. Immediate barge-in cancellation and audio buffer flushing.
4. Truncation of conversation history to what was actually spoken before an interruption.
5. Sentence and clause aggregation for low-latency streaming TTS.

We evaluated three architectural paths:
1. **Hand-rolled loop** (extending `orchestration/dormant/worker/local_cascade.py`): High ongoing maintenance burden for provider protocol churn, audio jitter buffers, and corner cases in cancellation synchronization.
2. **LiveKit Agents**: High production readiness, but architectural coupling to WebRTC rooms and dedicated media servers that do not match our lightweight FastAPI WebSocket client architecture (`/v2/voice`).
3. **Pipecat (BSD-2)**: Modular, transport-agnostic frame-processor pipeline in Python. Matches our FastAPI WebSocket architecture, supports our target providers natively (Deepgram Flux, Sarvam, Cartesia, ElevenLabs, OpenAI/Anthropic), and handles sentence aggregation and interruption mechanics.

## Decision
We embed **Pipecat** (`pipecat-ai>=1.12.0`) as the internal pipeline execution engine for Engine B, placed strictly inside the infrastructure layer (`orchestration/infrastructure/engines/cascaded/`).

Pipecat will be hidden behind our domain-defined `VoiceEngine` interface so that:
1. The domain and application layers remain completely framework-free.
2. Both Engine A and Engine B emit identical unified events over the existing `/v2/voice` WebSocket protocol.
3. Engine B can be replaced or modified without affecting the rest of the application.

## Consequences
- **Positive:** Reuses well-tested interruption management, TTS clause aggregation, and streaming provider adapters (Deepgram Flux, Sarvam, Cartesia, ElevenLabs, OpenAI, Anthropic).
- **Positive:** Extremely low overhead: measured at 36 µs (0.036 ms) per 20 ms audio frame on Windows 11 with Python 3.11, with 28 ms interruption propagation.
- **Positive:** Zero PyTorch CUDA conflicts with the existing PersonaPlex GPU environment in `.venv-gpu`.
- **Trade-off:** Adds `pipecat-ai` and provider SDK dependencies. Isolated into an optional dependency extra (`cloud-engine`) in `pyproject.toml`.
