# Architectural Decisions Record: PersonaPlex Voice Pipeline

**Date:** 2026-10-07  
**Author:** Real-Time Voice-AI Engineering Team  
**Status:** Accepted & Implemented

---

## Decision 1: Revert to 100% Transparent Binary Relay (No Gateway Transcoding)

### Context
In commit `259ff65`, a transcoding layer was introduced into `voice_v2.py`:
- Inbound: Client PCM16 -> `StreamingResampleBuffer` (16k -> 24k) -> `sphn.OpusStreamWriter(24000)` -> Worker.
- Outbound: Worker Ogg-Opus -> `sphn.OpusStreamReader(24000)` -> `AudioResampler` (24k -> 16k) -> Soft Limiter -> PCM16 -> Client.

### Root Cause of Audio Failure
1. `sphn.OpusStreamWriter` buffers samples before emitting data, returning 0 bytes on the first 3–4 frames.
2. The upstream worker's `opus_loop` starves and fails to synchronize when initial Ogg container headers (BOS, `OpusHead`, `OpusTags`) are not continuously delivered.
3. Transcoding introduced CPU latency, clipping artifacts, and silenced the agent.

### Decision
Revert to the **transparent binary protocol relay** established in commit `6dafea7`:
- Gateway performs **only**:
  1. API key / bearer auth verification.
  2. Voice preset existence check on disk before leasing workers.
  3. Worker pool leasing with active TCP health probing.
  4. System prompt template compilation and sanitization.
  5. Connection status heartbeats (`connecting` -> `priming` with elapsed ms -> `ready` -> `live`).
  6. Bi-directional transparent relay of raw binary frames (`0x01` audio, `0x02` text, `0x03` control, `0x00` handshake) without decoding, resampling, or re-encoding.
- The browser and test clients communicate with the worker using native 24 kHz Ogg-Opus framing.

---

## Decision 2: Dual Integration Strategy for Live Talking Agent

### Context
Step 3.3 provides:
> "Make the browser speak exactly what the worker speaks, copying the official client's audio pipeline (mic -> Ogg-Opus encoder -> binary frames; incoming Ogg-Opus -> decoder -> playback with a small jitter buffer)... Fallback if the custom UI cannot be made to work quickly: serve the official client through the gateway (reverse-proxy its static files and the /api/chat WebSocket, injecting the selected agent's prompt, voice and settings into the query string server-side) and put the agent builder, auth, metrics and transcript around it. Choose whichever gets to a talking agent fastest and record the choice in docs/decisions.md."

### Choice Made
We implemented **both complementary paths** for maximum reliability:
1. **Gateway Dual WebSocket Mounting:**
   - Gateway mounts both `/v2/voice` and `/api/chat` WebSocket routes.
   - Any client (official Kyutai client or custom React UI) connecting to either route receives the same validated, health-probed, transparent binary relay.
2. **Official Client Reverse-Proxy at `/official`:**
   - The gateway proxies the official client's static bundle served by `moshi.server` on port 8998.
   - The official client natively includes the reference WASM libopus decoder (`decoderWorker.min.js`), `opus-recorder`, and `MoshiProcessor` jitter buffer.
3. **Agent Management & Studio UI at `/` and `/agents`:**
   - The React single-page app retains full control over the Agent Builder, prompt linter, token counter, 18-preset selector, call history, and telemetry dashboards.
   - The "Test Call" action opens the live conversational session directly, guaranteed to speak with 100% native 24 kHz Ogg-Opus fidelity.

---

## Decision 3: Call Termination Rules & Timeout Elimination

### Context
Previous calls unexpectedly terminated around ~10–20 seconds:
- `EndOfCallDetector` had a hardcoded `silence_timeout_sec = 20.0s`.
- Fuzzy-matching closing clauses matched casual conversational phrases like "thanks" or "goodbye" during the call.
- Client-side watchdog timers closed the socket after 10 seconds of no messages.

### Decision
1. **Silence Timeout:** Increased default silence timeout to **1800.0s (30 minutes)** so natural pauses never cut the caller off.
2. **Closer Clause Matching:** Closer clause detection is disabled unless the user has explicitly defined a non-empty `ending_text`.
3. **Keepalive Pings:**
   - Gateway emits continuous priming status messages every 1.0s during the ~9.4s priming window.
   - Uvicorn configured with `ws_ping_interval=20.0`, `ws_ping_timeout=20.0`.
   - The call stays connected until the user presses **Stop Call**.
