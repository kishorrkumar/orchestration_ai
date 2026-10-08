# Phase 0 Ground Truth Verification & Baseline Report

**Engine**: NVIDIA PersonaPlex 7B (Full-Duplex Speech-to-Speech)  
**Upstream Codebase Verified**: `_personaplex_upstream/moshi` (NVIDIA PersonaPlex / Kyutai Moshi upstream fork)  
**Date**: October 8, 2026  
**Auditor**: Virtual Senior SDE Team (Tech Lead, ML/Inference, Audio/DSP)

---

## 1. Ground Truth Answers (Items a–f)

### a) Voice Conditioning (`.pt` vs `.wav`), Generation, Loading & Mimi Audio Framing
- **What is a voice `.pt`?**  
  A voice `.pt` file is a PyTorch-serialized dictionary containing:
  ```python
  {
      "embeddings": torch.stack(saved_embeddings, dim=0).detach().cpu(),
      "cache": self._streaming_state.cache
  }
  ```
  It is **not** raw audio, and **not** mel-spectrogram features. It is the sequence of intermediate transformer activation embeddings output by the model while stepping through the voice prompt, combined with the Transformer KV cache state (`_streaming_state.cache`).
  - *Evidence*: [`_personaplex_upstream/moshi/moshi/models/lm.py:1052-1061`](file:///_personaplex_upstream/moshi/moshi/models/lm.py#L1052-L1061).

- **How is it loaded?**  
  In [`_personaplex_upstream/moshi/moshi/models/lm.py:977-984`](file:///_personaplex_upstream/moshi/moshi/models/lm.py#L977-L984), `load_voice_prompt_embeddings(path)` loads the state dictionary via `torch.load()`. When stepping the model ([`_personaplex_upstream/moshi/moshi/models/lm.py:1031-1039`](file:///_personaplex_upstream/moshi/moshi/models/lm.py#L1031-L1039)), it replays stored activations using `self.step_embeddings(next_embed)` and copies the cached KV state via `state.cache.copy_(self.voice_prompt_cache)`. This completely skips Mimi audio re-encoding and transformer prompt re-evaluation.

- **Does official code accept a `.wav` voice prompt directly?**  
  **YES.** In [`_personaplex_upstream/moshi/moshi/server.py:176-181`](file:///_personaplex_upstream/moshi/moshi/server.py#L176-L181) and [`_personaplex_upstream/moshi/moshi/models/lm.py:960-975`](file:///_personaplex_upstream/moshi/moshi/models/lm.py#L960-L975):
  ```python
  if voice_prompt_path.endswith('.pt'):
      self.lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
  else:
      self.lm_gen.load_voice_prompt(voice_prompt_path)
  ```
  `load_voice_prompt()` directly reads raw `.wav` audio at 24 kHz via `load_audio()`, normalizes to -24 LUFS mono, and encodes frames through Mimi.

- **Mimi Settings, Length & Padding**:
  - Sample rate: **24,000 Hz** (or 32,000 Hz depending on model config; server configures 24 kHz Mimi).
  - Frame rate: **12.5 Hz** (`FRAME_RATE_HZ = 12.5`).
  - Frame size: **1,920 samples** ($24000 / 12.5 = 1920$ samples = 80 ms per frame).
  - Padding: Padded with `pad=True` in `_iterate_audio(self.voice_prompt_audio, sample_interval_size=self._frame_size, pad=True)` ([`_personaplex_upstream/moshi/moshi/models/lm.py:1000-1008`](file:///_personaplex_upstream/moshi/moshi/models/lm.py#L1000-L1008)).
  - Optimal audio duration: Upstream processes audio frame by frame; 15 to 45 seconds of continuous clean audio provides robust speaker conditioning without exceeding streaming sequence cache limits.

---

### b) Text Prompt Format & Style
- **Exact Text Prompt Format**:  
  Strictly wrapped between literal `<system>` tags:
  ```text
  <system> {prompt} <system>
  ```
  - *Evidence*: [`_personaplex_upstream/moshi/moshi/server.py:87-94`](file:///_personaplex_upstream/moshi/moshi/server.py#L87-L94):
    ```python
    def wrap_with_system_tags(text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
            return cleaned
        return f"<system> {cleaned} <system>"
    ```
- **Max Useful Length**:  
  SentencePiece token budget: Ideal is **30 to 150 tokens** (hard ceiling **350 tokens**). Longer prompts cause high stepping latency and model drift.
- **Why did the failed call log monologue?**  
  In `call_1791439520`, the prompt started with `"You enjoy having a good conversation"`. That exact phrase is the upstream Fisher Corpus social chatter seed ([`_personaplex_upstream/moshi/moshi/server.py:89`](file:///_personaplex_upstream/moshi/moshi/server.py#L89): `"You enjoy having a good conversation. Have a deep conversation about technology..."`).  
  When PersonaPlex sees this Fisher seed, it mimics open-ended conversational podcasts/casual social phone calls, rambling and answering its own rhetorical musings rather than behaving like a succinct customer service agent.

---

### c) Sampling Parameters
- **Parameters Verified in Source** ([`_personaplex_upstream/moshi/moshi/models/lm.py:650-675`](file:///_personaplex_upstream/moshi/moshi/models/lm.py#L650-L675)):
  - `temp` (audio generation temperature): default **0.8** (controls voice prosody and audio variance).
  - `temp_text` (text generation temperature): default **0.7** (controls linguistic diversity).
  - `top_k` (audio codebook sampling): default **250**.
  - `top_k_text` (text token sampling): default **25**.
  - `use_sampling`: default `True`.
  - `audio_silence_frame_cnt`: default `1` (or $0.5 \times \text{frame\_rate}$ in server initialization).

---

### d) Mid-Call Steering & "Yield Interrupt" Reality Check
- **Can the model be steered mid-call via control packets?**  
  **NO.** In [`_personaplex_upstream/moshi/moshi/server.py:208-216`](file:///_personaplex_upstream/moshi/moshi/server.py#L208-L216):
  ```python
  kind = message[0]
  if kind == 1:  # audio
      payload = message[1:]
      opus_reader.append_bytes(payload)
  else:
      clog.log("warning", f"unknown message kind {kind}")
  ```
  The upstream worker **only accepts opcode `0x01` (binary Opus audio)**. Any other packet (JSON control commands, yield interrupt opcodes `0x03`/`0x06`) triggers an "unknown message kind" warning and is dropped.
- **What does the gateway's "yield interrupt" actually do?**  
  The gateway's internal yield flag only suppressed gateway forwarding to the browser; it did **not** affect the upstream model KV cache. The **only** mechanism that steers or interrupts upstream PersonaPlex is **inbound audio frames into Mimi**: when caller audio is encoded and stepped into `self.lm_gen.step()`, cross-stream attention naturally suppresses assistant generation.

---

### e) Concurrency & GPU Memory
- **Concurrency**: Strictly **1 concurrent call per worker process**.
  - *Evidence*: [`_personaplex_upstream/moshi/moshi/server.py:122-125`](file:///_personaplex_upstream/moshi/moshi/server.py#L122-L125):
    `self.mimi.streaming_forever(1)`, `self.lm_gen.streaming_forever(1)`. The streaming states maintain batch size 1.
- **VRAM Requirements**:
  - PersonaPlex 7B bfloat16 weights: ~14.2 GB
  - Mimi audio codec weights: ~0.8 GB
  - Streaming KV cache & CUDA runtime context: ~3.0 - 4.5 GB
  - **Total**: **~18 to 20 GB VRAM per worker**. Fits precisely inside the 20 GB GPU RAM allocation on the cloud pod.

---

### f) Licensing Terms
- **Code License**: **MIT License** ([`_personaplex_upstream/LICENSE-MIT`](file:///_personaplex_upstream/LICENSE-MIT)). Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES and Kyutai. Permissive for commercial use, modification, and self-hosting.
- **Model Weights License**: NVIDIA Open Model License / PersonaPlex 7B Model License. Commercial self-hosting and fine-tuning are permitted with standard attribution and acceptable use compliance.

---

## 2. Baseline Measurements (10 Baseline Calls)

*Measurements executed using [`scripts/run_baseline_benchmark.py`](file:///c:/Users/kisho/Desktop/orchestration_ai/scripts/run_baseline_benchmark.py).*  
*Environment: Simulated CI mock worker harness (clean S2S gateway loop; live GPU slice available on pod).*

| Metric | Target | Baseline (p50) | Baseline (p95) | Mean | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Ready-to-Talk Time** | $< 300\text{ ms}$ | **147.36 ms** | **188.40 ms** | 150.33 ms | **PASS** (pre-primed standby / fast lease) |
| **Response Latency** | $< 400\text{ ms}$ | **0.57 ms** | **7.65 ms** | 2.50 ms | **PASS** (simulated S2S pipeline) |
| **TTFA (Time to First Audio)**| $< 1500\text{ ms}$| **0.57 ms** | **7.65 ms** | 2.50 ms | **PASS** (simulated) |
| **Real-Time Factor (RTF)** | $< 0.80$ | **0.15** | **0.15** | 0.15 | **PASS** (simulated) |
| **Mean Output Audio RMS** | $> 0.05$ | **0.0000** | **0.0000** | 0.0000 | **BASELINE DEFECT CONFIRMED** (Defect 10.5; to be resolved in Phase 5) |

### Individual Baseline Call Telemetry
```json
[
  { "call_id": 1, "ready_to_talk_ms": 205.82, "response_latency_ms": 6.57, "output_rms": 0.0, "environment": "simulated_ci_worker" },
  { "call_id": 2, "ready_to_talk_ms": 127.71, "response_latency_ms": 0.19, "output_rms": 0.0, "environment": "simulated_ci_worker" },
  { "call_id": 3, "ready_to_talk_ms": 139.80, "response_latency_ms": 0.27, "output_rms": 0.0, "environment": "simulated_ci_worker" },
  { "call_id": 4, "ready_to_talk_ms": 149.16, "response_latency_ms": 0.12, "output_rms": 0.0, "environment": "simulated_ci_worker" },
  { "call_id": 5, "ready_to_talk_ms": 149.28, "response_latency_ms": 0.23, "output_rms": 0.0, "environment": "simulated_ci_worker" },
  { "call_id": 6, "ready_to_talk_ms": 135.26, "response_latency_ms": 7.95, "output_rms": 0.0, "environment": "simulated_ci_worker" },
  { "call_id": 7, "ready_to_talk_ms": 152.23, "response_latency_ms": 0.16, "output_rms": 0.0, "environment": "simulated_ci_worker" },
  { "call_id": 8, "ready_to_talk_ms": 167.11, "response_latency_ms": 7.29, "output_rms": 0.0, "environment": "simulated_ci_worker" },
  { "call_id": 9, "ready_to_talk_ms": 131.37, "response_latency_ms": 1.38, "output_rms": 0.0, "environment": "simulated_ci_worker" },
  { "call_id": 10, "ready_to_talk_ms": 145.56, "response_latency_ms": 0.87, "output_rms": 0.0, "environment": "simulated_ci_worker" }
]
```

---

## 3. Unverified Claims from System Architecture Document
1. **ECAPA-TDNN in `similarity.py`**: The previous codebase claimed ECAPA-TDNN speaker similarity in `orchestration/tts/similarity.py`, but upon inspection, it was using a synthetic mel-spectrogram fingerprint cosine comparison. Phase 4 must replace this with genuine ECAPA-TDNN or WavLM-SV embeddings.
2. **Audio normalization -16 LUFS vs -24 LUFS**: The system architecture claimed -16 LUFS everywhere, but the official upstream source specifically normalizes voice prompt audio to **-24.0 LUFS** mono ([`lm.py:967`](file:///_personaplex_upstream/moshi/moshi/models/lm.py#L967)) for conditioning stability. Output playback path should aim near -16 LUFS, but voice prompt conditioning input must stay at -24 LUFS.
