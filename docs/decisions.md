# Architectural & Technical Decisions

This document records the architectural and engineering decisions made for the **PersonaPlex 7B Full-Duplex Real-Time Voice Agent Platform** on Krutrim Cloud (NVIDIA A100 SXM4 40GB).

---

## 1. Single-Agent Architecture (`agent.yaml`)
- **Context:** The repository previously seeded 3 hardcoded starter agents and offered a multi-agent selector UI, which violated product clarity and caused DB sync drift.
- **Decision:** Enforce **ONE agent only**.
- **Implementation:**
  - Defined declaratively in [agent.yaml](file:///agent.yaml) (with sample in [agent.example.yaml](file:///agent.example.yaml)).
  - Contains name, system prompt, variable defaults, voice prompt ID, and generation hyperparameters.
  - Startup database synchronization in [orchestration/db/seed.py](file:///orchestration/db/seed.py) purges legacy starter records and synchronizes the single agent.
  - Single-agent editor in the console allows live editing and publishing.

---

## 2. Audio Pipeline & Wire Protocol Alignment
- **Context:** PersonaPlex uses the Kyutai Mimi neural audio codec operating at:
  - Sample Rate: **24,000 Hz** (mono)
  - Frame Rate: **12.5 Hz** (1 frame every 80 ms)
  - Frame Size: **1,920 float32 samples** (80 ms $\times$ 24 kHz)
- **Decision:**
  - Upstream worker communication uses binary framing over WebSocket (`0x00` handshake, `0x01` Ogg-Opus audio chunks via `sphn`, `0x02` UTF-8 text tokens).
  - Browser transport communicates via 16 kHz PCM16 or WebRTC Opus.
  - Gateway converts between 16 kHz and 24 kHz using anti-aliased polyphase/Soxr streaming resamplers ([orchestration/audio/resample.py](file:///orchestration/audio/resample.py)).

---

## 3. Worker Inference Clocking: Continuous Silence Pacing
- **Context:** In early tests, callers heard NO audio and only saw transcripts. Because the gateway blocked waiting for microphone packets while callers waited in silence for the agent to greet them, the worker received 0 audio frames, stalling its auto-regressive inference loop.
- **Decision:** Implement a decoupled 12.5 Hz pacer task in [orchestration/api/voice_v2.py](file:///orchestration/api/voice_v2.py).
- **Mechanism:** When the user is silent, the pacer continuously transmits 1,920-sample zero frames at 12.5 Hz to keep the model's audio clock advancing so the agent can vocalize.

---

## 4. Elimination of Fake Synthetic Greetings
- **Context:** Previous versions injected a synthetic JSON text transcript (`"Hello, thank you for calling..."`) directly into the WebSocket on connection before receiving any audio from the worker. This caused:
  1. The user seeing a transcript while hearing no audio.
  2. The agent greeting twice (once via fake text, and once when the neural model vocalized).
- **Decision:** Completely removed synthetic greeting text injection. All greeting transcripts and audio originate purely from the PersonaPlex worker.

---

## 5. Strict Template Variable Resolution
- **Context:** Unresolved placeholders like `{{company}}` were previously sent directly to the model, causing the agent to speak literal template syntax over the phone.
- **Decision:** Strict pre-compilation with typed defaults.
- **Implementation:**
  - [orchestration/prompts/compiler.py](file:///orchestration/prompts/compiler.py) loads variable defaults from `agent.yaml`.
  - Raises `TemplateResolutionError` (mapped to HTTP 422 Problem Details) if any `{{variable}}` cannot be resolved.
  - Unit tests guarantee that `{{` or `}}` can never reach the model worker.

---

## 6. Priming Wall & Pre-Warming Architecture
- **Context:** On an NVIDIA A100 40GB, PersonaPlex worker priming takes **~9.4 seconds** ($\approx 362$ auto-regressive steps $\times$ 26 ms per step) to ingest the voice prompt, system prompt, and context tokens. If users click "Start Call" without notice, browsers timed out or users hung up.
- **Decisions:**
  1. **UI Transparency:** Instant status messages (`connecting` $\to$ `priming` with live elapsed milliseconds $\to$ `ready`).
  2. **Keepalive Pulses:** Gateway streams WebSocket status keepalives during priming to prevent proxy idle timeouts.
  3. **Pre-Warming (Standby Lease):** When the browser console page loads, the gateway leases and primes an idle worker in the background. When the user clicks "Start", the session attaches instantly ($T_{\text{ready}} \le 20$ ms, TTFA $\le 280$ ms).

---

## 7. Deprecation of ScriptProcessorNode $\to$ AudioWorklet
- **Context:** Browsers emitted console deprecation warnings for `ScriptProcessorNode`, which ran on the main UI thread and suffered from audio dropouts during DOM renders.
- **Decision:** Replaced with an inline `AudioWorklet` processor (`AudioCaptureProcessor`) registered via a Blob URL in [frontend/src/views/TestCallModal.tsx](file:///frontend/src/views/TestCallModal.tsx).
- **Benefits:** Runs on a dedicated Web Audio rendering thread, enforces continuous 20 ms (320-sample) chunking at 16 kHz, and computes live input RMS without UI thread jitter.

---

## 8. Honest Interruption & Barge-In Architecture
- **Context:** PersonaPlex is an end-to-end full-duplex speech-to-speech foundation model that handles turn-taking, backchanneling, and barge-in internally within its neural weights. An external orchestrator cannot cancel neural generation mid-stream without resetting model state.
- **Decision:**
  - Client-side VAD (energy threshold + AudioWorklet) detects user speech onset.
  - When the caller speaks over the agent, the client immediately flushes its local playback jitter buffer ($< 30$ ms time-to-silence).
  - Transcript marks the turn as `interrupted`.
  - Server metrics record interruption events and time-to-silence.

---

## 9. WebRTC Transport (aiortc) & Network Fallback
- **Context:** Krutrim Cloud pods sit behind an HTTPS reverse proxy mapping external port 443 to internal port 8000. UDP media ports (WebRTC RTP/RTCP) are typically blocked or unreachable through HTTP-only ingress proxies.
- **Decision:**
  - Built WebRTC transport ([orchestration/api/webrtc.py](file:///orchestration/api/webrtc.py)) powered by `aiortc` behind the same worker interface.
  - Implemented runtime ICE connection detection in the browser.
  - If WebRTC ICE negotiation fails (due to UDP ingress restrictions), the frontend automatically alerts the user and falls back seamlessly to the resilient WebSocket transport.
  - Documented production TURN (coturn) over TCP/TLS (port 443) for strict corporate and pod firewalls.

---

## 10. Security & Operations
- **Single Bearer Token:** Optional `AUTH_TOKEN` environment variable enforced across both REST endpoints and WebSocket handshakes (`?token=` query param or `Authorization: Bearer` header).
- **Structured Telemetry:** Per-hop latency, RMS energy, text token counters, and Prometheus-compatible metrics exposed at `/metrics`.
- **Health Checks:** `/healthz` provides worker pool availability and readiness status.
