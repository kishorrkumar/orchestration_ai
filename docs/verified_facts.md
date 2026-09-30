# NVIDIA PersonaPlex: Verified Ground Truth Specifications

This document records the empirical ground truths verified by inspecting the upstream NVIDIA PersonaPlex codebase (`NVIDIA/personaplex` and `kyutai-labs/moshi`), specifically `moshi/server.py`, `moshi/models/lm.py`, `client/src/pages/Queue/Queue.tsx`, and the model checkpoint definitions.

---

## 1. Supported Languages
* **Verified Fact:** PersonaPlex is strictly **English-only**.
* **Source Evidence:**
  - Upstream models are trained on the **Fisher English Corpus** (`LDC2004T19`) and English synthetic dialogues.
  - The SentencePiece tokenizer (`tokenizer_spm_32k_3.model`) has a 32,000 English-centric vocabulary.
  - Upstream contains no multilingual training or evaluation pipeline.
* **Operational Rule:** Do NOT claim native multilingual or Indic (Hindi, Tamil, etc.) support in PersonaPlex. If non-English audio is passed to PersonaPlex, the model will output garbled tokens or English phonetic hallucinations. For Indic multilingual conversational voice, the system provides our cascaded pipeline (Whisper/Faster-Whisper ASR + Ollama LLM + Kokoro/EdgeTTS) as a documented, honest alternative, while reserving native PersonaPlex for English-native speech-to-speech.

---

## 2. Voice Conditioning & Cloning (.wav vs .pt)
* **Verified Fact:** Upstream supports both pre-computed `.pt` embeddings and raw `.wav` audio files.
* **18 Built-in Presets:**
  - **Natural Female (4):** `NATF0.pt`, `NATF1.pt`, `NATF2.pt`, `NATF3.pt`
  - **Natural Male (4):** `NATM0.pt`, `NATM1.pt`, `NATM2.pt`, `NATM3.pt`
  - **Variety Female (5):** `VARF0.pt`, `VARF1.pt`, `VARF2.pt`, `VARF3.pt`, `VARF4.pt`
  - **Variety Male (5):** `VARM0.pt`, `VARM1.pt`, `VARM2.pt`, `VARM3.pt`, `VARM4.pt`
* **Custom Voice (.wav) Processing Pipeline (`load_voice_prompt`):**
  1. Audio is loaded via `sphn.read` and resampled to **24,000 Hz mono**.
  2. Loudness is normalized to **-24.0 LUFS** (using `pyloudnorm` / `sphn`).
  3. Audio is split into **1,920 sample frames** (80 ms per frame at 24 kHz).
  4. Each frame is encoded into 8 codebooks via `mimi.encode()`.
  5. Frames are sequentially stepped through the LM (`LMGen.step()`) with:
     - `moshi_tokens = voice_prompt_frame_tokens`
     - `text_token = zero_text_code (3)`
     - `input_tokens = sine_frame`
  6. When `save_voice_prompt_embeddings = True`, upstream saves:
     ```python
     torch.save({
         "embeddings": torch.stack(saved_embeddings, dim=0).detach().cpu(),
         "cache": self._streaming_state.cache
     }, f"{voice_id}.pt")
     ```
* **Sample Requirements & Constraints:**
  - **Format:** Clean single-speaker audio, SNR > 20 dB, no background music or cross-talk.
  - **Duration:** Recommended **5 to 12 seconds** (ideal ~8-10 seconds).
  - **Why not longer?** Stepping through voice frames happens synchronously during connection initialization before the WebSocket handshake. An 80ms frame takes ~15-25ms to process. A 10-second reference takes 125 frames (~2-3 seconds of init latency). A 60-second reference would delay the handshake by 15-20 seconds, causing browser WebSocket timeouts.

---

## 3. System Prompt Format & Delimiters
* **Verified Fact:** The exact wrapping format is:
  ```
  <system> {system_prompt} <system>
  ```
* **Key Observations:**
  - Both opening and closing delimiters are `<system>` (literally `<system>`, NOT `</system>`).
  - Leading and trailing whitespace within the prompt is trimmed.
  - Tokenization: Tokens are generated with `SentencePieceProcessor(tokenizer_spm_32k_3.model)`.
* **Prompt Stepping Sequence:**
  1. `_step_voice_prompt` (voice frames or embeddings)
  2. `_step_audio_silence` (0.5 seconds = 6 frames of silence tokens + sine input)
  3. `_step_text_prompt` (text tokens stepped one-by-one: `moshi_tokens=zero_frame`, `text_token=token_id`, `input_tokens=sine_frame`)
  4. `_step_audio_silence` (0.5 seconds = 6 frames)
  5. Handshake `0x00` byte dispatched to client.
* **Prompt Length Limits:**
  - Every token in the system prompt adds one autoregressive step to the session startup.
  - Recommended max prompt length: **150 tokens** (~100-120 words).
  - Hard limit enforced by orchestration layer: **350 tokens** to prevent startup timeouts (>7 seconds startup).

---

## 4. WebSocket Wire Protocol
The WebSocket connection (`/api/chat` upstream, `/v1/realtime` gateway) uses a 1-byte opcode prefix:
- `0x00`: **Handshake** (2 bytes: version, model) - sent by server after prompt stepping completes.
- `0x01`: **Audio Frame** (Opus stream packets or raw 24 kHz 16-bit linear PCM).
- `0x02`: **Text Token** (UTF-8 string, special token `▁` converted to space).
- `0x03`: **Control Signal** (1 byte: `0=start`, `1=endTurn`, `2=pause`, `3=restart`).
- `0x04`: **Metadata** (UTF-8 JSON string).
- `0x05`: **Error** (UTF-8 string error message).
- `0x06`: **Ping / Heartbeat**.

---

## 5. Concurrency & Resource Budget
* **Upstream Concurrency:** Upstream `moshi.server` wraps every connection in an exclusive lock:
  ```python
  self.lock = asyncio.Lock()
  async with self.lock:
      ...
  ```
  **A single worker process can serve exactly ONE concurrent audio stream.**
* **VRAM Allocation (PersonaPlex 7B):**
  - Model weights: ~14.5 GB (BF16).
  - KV-cache & Mimi codec activations: ~3.5 GB.
  - Total per worker: **~18 - 20 GB VRAM**.
* **GPU Capacity:**
  - 24 GB GPU (NVIDIA A10G / RTX 3090 / RTX 4090): **1 worker process** (1 concurrent stream).
  - 48 GB GPU (NVIDIA A40 / RTX 6000 Ada): **2 worker processes** (2 concurrent streams).
  - 80 GB GPU (NVIDIA A100 / H100): **3-4 worker processes** (3-4 concurrent streams).
* **Worker Pool Architecture:**
  - Multi-stream concurrency requires running $N$ worker instances on distinct local ports (`8998`, `8999`, etc.), bound to `127.0.0.1`.
  - The Gateway acts as the reverse-proxy dispatcher, leasing idle workers and recycling them on disconnect.
