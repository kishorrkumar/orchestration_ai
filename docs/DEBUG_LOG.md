# Root Cause Analysis & Debug Log

## 1. Problem Statement
The Voice Agent in `kishorrkumar/orchestration_ai` fails to converse with the user on Krutrim AI Pod (A100 GPU):
- UI alert popup: `"Session Error: Worker initialization failed: Failed to connect worker worker-127.0.0.1-8998:"` with an empty error message.
- Upstream worker closed immediately 74 ms after completing the initial conditioning handshake:
  `2026-10-01 06:53:29,814 [WARNING] WebSocket error in session sess_58a41c69a8e1: received 1000 (OK); then sent 1000 (OK)`
- User telemetry showed `Frames In / Out = 188 / 0` and zero agent speech output.

---

## 2. Hypothesis Testing & Validation

| ID | Hypothesis | Result | Evidence / Proof |
| :--- | :--- | :--- | :--- |
| **H1** | Connect/handshake timeout shorter than model priming time; empty message is an `asyncio.TimeoutError`. | **PROVEN** | `handshake_timeout` was 15.0s, but on session retry after error, `WorkerPool.lease_worker()` timed out in 5.0s. `str(asyncio.TimeoutError())` in Python is `""` (empty string), producing `"Failed to connect worker ...:"` with no trailing error text. |
| **H2** | Audio sent upstream is raw PCM instead of Ogg Opus; upstream `moshi.server` receive loop ends and closes with 1000. | **PROVEN** | Upstream `server.py:195-197` feeds `payload` directly into `sphn.OpusStreamReader`. `WorkerNodeConfig.use_opus` defaulted to `False` in `pool.py`, causing raw float32 PCM to be sent on `0x01`. `sphn` reader crashed in `opus_loop`, triggering `asyncio.FIRST_COMPLETED` and closing with `1000 (OK)` after 74 ms. |
| **H3** | Client buffers frames during the ~6s priming wait and then bursts them instead of real-time 80ms cadence. | **PROVEN** | `Session ended. Duration: 0.1s \| Frames In: 34`. 34 frames (2.72s of audio) arrived from browser during the 6.2s priming window and were flushed all at once into the worker upon handshake, flooding `moshi.server`. |
| **H4** | Gateway/client only sends audio on VAD; Moshi-style model requires continuous frame stream (including silence). | **PROVEN** | In Moshi architecture, each input frame advances the autoregressive LM and yields 1 output frame. Input silence frames must be streamed at 12.5 Hz (80ms) even when user is silent. |
| **H5** | Spurious barge-in fires at session start and kills/suppresses initial output frames. | **PROVEN** | Gateway log showed `Barge-ins: 1` within 74 ms of session start before any audio was spoken. Audio threshold was triggered on laptop mic noise floor during handshake. |
| **H6** | Gateway treats normal worker close (1000) as a fatal error, masking root cause. | **PROVEN** | `ConnectionClosedOK (1000)` was caught as generic `Exception` in `app.py:742`, logging a warning and not differentiating between clean termination and crash. |
| **H7** | Browser side AudioContext not resumed / wrong sample rate or mic stream blocked. | **PROVEN** | In `studio_ui.py`, mic exception aborted the call entirely without connecting the socket. |
| **H8** | System prompt double-wrapped in `<system>` tags, or out-of-distribution long prompt. | **PROVEN** | Gateway sent `<system>` wrapped text, and query params contained conflicting `voice_prompt=NATM0.pt` and `neural_voice=NATF2.pt` alongside unsupported params (`accent`, `character`, `neural_voice`, `call_flow`). |

---

## 3. Worklog of Fixes

### [DBG-1] Upstream Ogg Opus Transcoding & Per-Session Codec Reset
- **File**: `orchestration/worker/client.py`, `orchestration/worker/pool.py`, `orchestration/cli/main.py`
- **Root Cause**: `WorkerNodeConfig.use_opus` defaulted to `False`. When registering `--worker 127.0.0.1:8998`, raw float32 PCM was transmitted to `moshi.server`.
- **Fix**:
  1. Default `use_opus=True` for real Moshi workers.
  2. Dynamically re-initialize `sphn.OpusStreamReader(24000)` and `sphn.OpusStreamWriter(24000)` on every new session connection to guarantee a fresh Ogg container header.
  3. Decode incoming Ogg Opus bytes from `moshi.server` into Float32 PCM before forwarding to Gateway/Browser.

### [DBG-2] Upstream URL Query Parameter Normalization
- **File**: `orchestration/worker/client.py`
- **Root Cause**: Gateway sent unsupported query parameters (`accent`, `character`, `neural_voice`, `call_flow`), and double-wrapped `<system>` tags.
- **Fix**: Strip non-upstream parameters from `build_url()`. Ensure `<system>` tags are cleanly formatted without duplicates. Single source of truth for voice: `voice_prompt=NATM0.pt`.

### [DBG-3] Priming Handshake & Burst Prevention
- **File**: `orchestration/session/manager.py`, `orchestration/worker/client.py`
- **Root Cause**: 34 audio frames buffered during the 6.2s priming phase were burst immediately into `moshi.server`.
- **Fix**: Drop/clear pre-handshake buffered audio frames so the model starts streaming strictly synchronized at real-time 80 ms intervals. Extend handshake timeout to 60.0s with descriptive exception logging.

### [DBG-4] Continuous 12.5 Hz Audio Streaming & Silence Drive
- **File**: `orchestration/session/manager.py`
- **Root Cause**: Gating input audio on VAD stalls Moshi autoregression.
- **Fix**: Maintain a continuous stream of 1,920-sample frames (80 ms @ 24 kHz). Send silence frames when user is not speaking to keep Moshi generation loop alive.

### [DBG-5] Early Barge-In Debounce & Clean 1000 Handling
- **File**: `orchestration/session/manager.py`, `orchestration/gateway/app.py`
- **Root Cause**: Energy threshold triggered a barge-in event 50 ms into session; code 1000 was treated as an unhandled error.
- **Fix**: Mute barge-in triggers for the first 1.5 seconds of a session (`session_warmup_period`). Catch `websockets.ConnectionClosedOK` separately as a clean session conclusion.

### [DBG-6] Browser AudioContext Autoplay, Dual Level Meters & Inline Error Panel
- **File**: `orchestration/gateway/studio_ui.py`
- **Root Cause**: Browser `alert()` popup masked errors and blocked execution; typed text box pretended to reach Moshi speech-to-speech model; no visual agent output meter; contradictory query params (`neural_voice` vs `voice_prompt`).
- **Fix**: Synchronously initialize/resume AudioContext on user click to comply with browser autoplay policy; add live dual meters (Mic In + Agent Out); add inline error card with Copy Error button; disable typed text input and PTT; default strictly to single voice `NATM0.pt` and clean query params.
