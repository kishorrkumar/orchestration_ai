# Voice Agent Diagnostics & Root Cause Analysis

**Repository:** `kishorrkumar/orchestration_ai`  
**Target Environment:** Krutrim Cloud A100 SXM4 40GB (Ubuntu 22.04+, Python 3.13)  
**Date:** October 7, 2026  

---

## 1. Executive Summary & Root Cause Ranking

Based on log analysis from Krutrim Cloud, frontend client traces, and audio signal pipeline diagnostics, the observed failure modes ("agent talks to itself", double greeting with literal `{{company}}`, ~9.4 s priming freeze followed by instant disconnects, and inaudible audio) stem from six distinct, compounding root causes:

| Rank | Issue | Root Cause | Impact |
|---|---|---|---|
| **1** | **Unrendered Template Variables & Dual Greeting** | `compiler.py` never defined defaults for `{{company}}` or `{{goal}}`, leaving raw `{{company}}` in prompts. Concurrently, `voice_v2.py` dispatched a synthetic greeting JSON while PersonaPlex model also vocalized the greeting from the system prompt. | Model vocalizes literal `{{company}}`, followed by a duplicate spoken greeting. |
| **2** | **Acoustic Feedback / Self-Echo (Talking to Itself)** | Browser `ScriptProcessorNode` routed mic directly to `destination` (loopback). Concurrently, `getUserMedia` lacked explicit constraint enforcement, and gateway lacked mic mute during assistant speech. | Speaker audio leaked directly into mic; model heard its own speech and replied ("Oh no, that's frustrating..."). |
| **3** | **Unnotified 9.4s Priming Wait & Client Disconnection** | PersonaPlex 7B Mimi encoder + LM prompt conditioning requires ~8–9.5s on GPU. Gateway sent 0 progress updates or keepalives. Browser state was uninformative and closed before worker was ready. | 3 out of 4 sessions terminated within 150–300ms of handshake completion due to client-side abandonment. |
| **4** | **Browser AudioContext Suspended & Jitter Underruns** | Modern browsers suspend `AudioContext` unless initialized or resumed synchronously on user gesture. Concurrently, chunk playback directly scheduled `start(now)` without an initial jitter buffer (80–160ms). | Transcripts streamed into UI, but zero audio was audible to the user. |
| **5** | **Audio Frame Chunk Mismatch (480 vs 1920 samples)** | Gateway `StreamingResampleBuffer` was chunking at 480 samples (20ms at 24 kHz) instead of Moshi's native 1920 samples (80ms at 24 kHz = 12.5 Hz). | Mock worker dropped all incoming chunks as undersized, and native worker suffered buffer fragmentation. |
| **6** | **Tokenizer FileNotFoundError** | `compiler.py` threw uncaught `FileNotFoundError` if `tokenizer_spm_32k_3.model` was only present in HF cache or still downloading. | Caused fatal 1011 WebSocket disconnection upon call start. (Now patched with multi-dir discovery and BPE fallback). |

---

## 2. Technical Evidence & Deep Dive

### 2.1 Template Variables & Prompt Style
* **Evidence:** The prompt compiled for Alex contained:
  ```
  You are Alex from {{company}} support.
  Start: Open the call by saying: "Hi, this is Alex from {{company}} support. What's going on today?"
  ```
  Because `{{company}}` was not in `var_dict`, the prompt sent to the Moshi server literally contained `{{company}}`.
* **PersonaPlex Invariant:** PersonaPlex is an end-to-end full-duplex speech-to-speech foundation model, not a text chatbot. Scripts with `Start: Open the call by saying: "..."` and `Close: ...` confuse the model's auto-regressive audio token stream, causing it to restart turns or vocalize punctuation.

### 2.2 Bidirectional Audio Resampling Verification
We conducted a bidirectional 440 Hz pure tone round-trip test (`tests/test_audio_resampler_diagnostics.py`):
1. **Hop 1 (Browser -> Gateway):** 16 kHz PCM16 mono -> Float32 `[-1.0, 1.0]`.
2. **Hop 2 (Gateway -> Worker):** Libsoxr polyphase resampler (16 kHz -> 24 kHz) accumulated into exact **1920-sample (80ms)** Float32 frames. Emitted 24 kHz audio is confirmed clean ($f = 440.0 \text{ Hz}$).
3. **Hop 3 (Worker -> Gateway):** 24 kHz Float32 received from worker.
4. **Hop 4 (Gateway -> Browser):** Resampled (24 kHz -> 16 kHz) via `AudioResampler(24000, 16000, quality='QQ')`, soft-clipped to 0.92 threshold, and converted to Little-Endian Int16 PCM.
* **Finding:** RMS preservation ratio was $1.002$ ($<0.3\%$ deviation), and peak FFT frequency matched $440.0 \text{ Hz}$ exactly. The audio math is bit-exact when frame size is set to 1920.

### 2.3 Acoustic Feedback Loop
* **Evidence:** In `TestCallModal.tsx`:
  ```typescript
  source.connect(processor);
  processor.connect(audioCtx.destination); // BUG: Outputted mic to speakers
  ```
  Connecting `processor` to `destination` without a zero-gain mute node created an acoustic loop.
  When the agent spoke, sound from the laptop speakers entered the microphone, passed through `voice_v2.py`, and was forwarded to the worker as user speech, causing the agent to interrupt and reply to itself.

### 2.4 Worker Priming Latency (9.4s)
* **Evidence:** Worker handshake log showed:
  ```
  Priming wait: 9410.2 ms
  ```
  The gateway previously sent no intermediate messages during this 9.4s window. If the user clicked outside the modal or the browser socket timed out, the session aborted immediately after the GPU worker became ready.

### 2.5 Audio Message Buffer Type Mismatch
* **Evidence:** When `worker_to_client_loop` received `AudioMessage` frames from the worker, `msg.data` was binary `bytes`. Calling `np.abs(raw_audio)` on bytes raised:
  ```
  ufunc 'absolute' did not contain a loop with signature matching types <class 'numpy.dtypes.BytesDType'> -> None
  ```
  This immediately aborted the connection with close code 1000 OK as soon as the agent began vocalizing. Unpacking `msg.data` with `np.frombuffer(raw_audio, dtype=np.float32)` resolved the crash.

---

## 3. Implementations & Architectural Fixes

### 3.1 Strict Prompt Compilation & Natural Persona Prose
1. **Defaults & Strict Resolution:** Defined `DEFAULT_VARIABLE_VALUES = {"company": "BrightNet", ...}` in `orchestration/prompts/compiler.py`. If any placeholder has no value, raises `TemplateResolutionError` (HTTP 422) at publish/save time.
2. **Prose Style (PersonaPlex Conditioning):** Removed rigid `Start:` and `Close:` quote scripts. Replaced with natural conversational scenario descriptions. Phrased time context naturally: `"It is Wednesday morning for the caller."`
3. **Seeded Agents:**
   - **Aarav (Default):** Warm, colloquial young Indian man having a relaxed conversation with natural markers (*achha, haan, actually, no worries, sure sure*).
   - **Support Agent (Alex):** Patient ISP support representative at {{company}} (BrightNet).
   - **Aarav - Support:** Indian-English conversational style applied to ISP troubleshooting.

### 3.2 High-Performance Frontend Audio Engine
1. **AudioWorklet (`audio-capture-worklet.js`):** Completely replaced deprecated `ScriptProcessorNode`. Captures microphone input at browser native rate, resamples to 16 kHz in real time, and sends fixed 320-sample (20ms) PCM16 frames.
2. **Jitter-Buffered Playback:** Output scheduling uses a 120ms jitter buffer (`nextPlayTimeRef`), smoothing network variance and completely eliminating underrun pops.
3. **Dual Live Level Meters:** Separate visual volume meters for Mic Input and Agent Audio.
4. **Echo Suppression:** Added "Use headphones" advisory banner and optional "Mute mic while agent speaks" toggle.
5. **Proxy Recovery:** Added automatic exponential backoff retry in `index.html` on proxy 502 errors.

### 3.3 Custom Indian-English Voice Conditioning
PersonaPlex conditioning accepts custom audio reference prompts.
- **Endpoint:** `POST /v2/agents/voices/upload`
- **Audio Specification:** Clean mono WAV, 10–20 seconds duration, 24 kHz sample rate, single Indian-English speaker reading naturally without music or background noise.
- **Recording Guide:**
  1. Record in a quiet room with minimal reverb.
  2. Speak at a normal conversational pace in colloquial Indian English.
  3. Avoid pauses longer than 1.0 second.
  4. PersonaPlex is English-trained; accent timbre is heavily driven by the voice reference file in combination with colloquial phrasing in the persona text.

### 3.4 Automated Smoke Test
Run `scripts/smoke_call.py` to verify full-duplex functionality:
```bash
python scripts/smoke_call.py --url ws://127.0.0.1:8000/v2/voice --agent-id aarav
```
Measures and prints priming time, Time to First Audio (TTFA), duration, audio RMS, and records the reply to `eval_out/reply.wav`.
