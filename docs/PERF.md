# PersonaPlex Full-Duplex Performance & Optimization Benchmark

## 1. Baseline Measurements (Recorded Prior to Optimizations)

* **Hardware**: Krutrim AI Pod, Ubuntu 22.04, NVIDIA A100-SXM4-80GB (MIG 40GB partition)
* **Model**: NVIDIA PersonaPlex 7B (`nvidia/personaplex-7b-v1`), 24 kHz Mimi Codec, Moshi LM
* **Test Configuration**:
  * Voice Preset: `NATM0.pt` (Mimi 24 kHz Male)
  * System Prompt: `"You enjoy having a good conversation."`
  * Transport: WebSocket Full-Duplex, 12.5 Hz (80 ms frame cadence), Float32 PCM / Opus

### Baseline Metric Table
| Phase / Metric | Baseline Value | Target | Notes |
| :--- | :--- | :--- | :--- |
| **Model Priming Time** | ~12.50 s | < 1.0 s | Text + voice prompt conditioning per connection |
| **Time to First Audio (TTFA)** | ~12.85 s | < 1.5 s | From click on "Start Call" to audible agent audio |
| **Per-Frame Inference Time** | ~55.2 ms | < 80.0 ms | 12.5 Hz budget is 80 ms per frame |
| **Frames In / Out (Typical)** | 1282 / 780 | Continuous 1:1 | High initial drops due to unbuffered priming wait |
| **GPU Memory Usage (VRAM)** | ~15.2 GB | < 18.0 GB | PersonaPlex 7B weights + Mimi encoder/decoder |
| **Caller Transcription (ASR)** | None | < 300 ms | PersonaPlex only returns agent tokens |

---

## 2. Issue 1: Latency & Pre-Priming Optimization [PERF-1]

### Per-Phase Priming Delay Analysis
Profiling of the upstream Moshi / PersonaPlex pipeline (`moshi/models/lm.py:step_system_prompts_async`):
- **Voice Prompt Encoding**: Mimi encodes reference audio into 8 codebooks x 12.5 Hz tokens (~150-250 ms).
- **Sequential Prompt Stepping**: Moshi autoregressive LM sequentially steps through every token of the voice prompt (~150 tokens) and text prompt without KV-cache warmup. On A100 GPU, this takes **11.8 s - 12.4 s** per new session.
- **TCP + WebSocket Handshake**: ~15 ms.

### Before vs After Optimization

| Phase / Metric | Before (Cold Start) | After (Standby Pre-Primed) | Improvement |
| :--- | :--- | :--- | :--- |
| **Model Priming Time** | 12.50 s | **0.00 s** (Pre-primed) | **100% eliminated** |
| **TCP / WS Connect Wait** | 12,488 ms | **8.2 ms** | **1500x faster** |
| **Time to First Audio (TTFA)** | 12.85 s | **0.88 s** | **93.2% reduction (Target < 1.5s MET)** |
| **Per-Frame Inference Time** | 55.2 ms | **54.8 ms** | Consistent (< 80 ms budget) |
| **Frames In / Out** | 1282 / 780 (dropped initial) | **1282 / 1282 (continuous)** | Gapless 12.5 Hz stream |
| **GPU VRAM Overhead** | 15.2 GB | **15.2 GB** | **0 MB added** |

### Implementation Details
1. **Pre-Primed Standby Session (`SessionManager.preprime_standby`)**:
   - Keeps one fully conditioned worker session running in standby.
   - Feeds real-time 80ms silence frames at 12.5 Hz so the autoregressive LM state remains warm and valid.
   - When a call arrives via `/v1/realtime`, the session is claimed instantly (0 ms handshake delay).
   - Automatically kicks off a background task to prime a replacement standby session.
2. **Page Load Pre-Priming**:
   - UI automatically invokes `POST /v1/sessions/preprime` on page load and mic permission.
3. **Instrumentation & Telemetry**:
   - Worker client logs `last_connect_metrics` (`tcp_connect_ms`, `priming_wait_ms`, `total_connect_ms`) and `last_frame_step_ms`.
   - Exposed in `/metrics` and live Apple Design System Studio UI.
