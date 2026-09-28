# NVIDIA PersonaPlex: Technical Architecture & System Notes

## 1. Overview
**PersonaPlex** is a real-time, full-duplex speech-to-speech conversational AI model developed by NVIDIA, based on the **Moshi** architecture (Kyutai) and the **Mimi** neural audio codec. Unlike traditional cascaded voice agent pipelines (Automatic Speech Recognition [ASR] → Large Language Model [LLM] → Text-to-Speech [TTS]), PersonaPlex operates end-to-end on continuous audio streams. It simultaneously listens and speaks, enabling natural conversational dynamics including barge-ins, mid-sentence interruptions, overlapping speech, and backchanneling (e.g., "uh-huh", "got it").

Model weights are hosted on Hugging Face under [`nvidia/personaplex-7b-v1`](https://huggingface.co/nvidia/personaplex-7b-v1), and the official code repository is hosted at [`NVIDIA/personaplex`](https://github.com/NVIDIA/personaplex).

---

## 2. Technical Specifications

### 2.1 Audio Format, Sample Rate, and Frame Size
* **Sampling Rate:** `24,000 Hz` (24 kHz), single channel (mono), 32-bit float or 16-bit PCM.
* **Frame Rate:** `12.5 Hz` (12.5 frames per second).
* **Frame Size:** `1,920 samples` per step (`24,000 samples/sec / 12.5 frames/sec = 1,920 samples`).
* **Frame Duration:** `80 milliseconds` (`1,920 / 24,000 = 0.08 s`).
* **Audio Codec (Mimi):**
  * Mimi compresses 24 kHz raw audio into discrete audio codebooks at 12.5 Hz.
  * Mimi quantizer: Split Residual Vector Quantizer with `n_q=32` codebooks, codebook cardinality `bins=2048`.
  * For Moshi/PersonaPlex autoregression: `dep_q=8` agent audio codebooks, `n_q=16` total audio codebooks, delays: `[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]`.
* **Streaming Wire Transport:**
  * Uses Opus streaming packets (`sphn.OpusStreamWriter` and `sphn.OpusStreamReader`) over binary WebSockets or raw linear 24 kHz PCM frames.

---

## 3. Inference Entrypoints

### 3.1 Official Server: `python -m moshi.server`
* **File:** `moshi/moshi/server.py`
* **Transport:** Asynchronous HTTP + WebSocket server built on `aiohttp`.
* **Default Port:** `8998`
* **Endpoint:** `GET /api/chat` (upgrades to WebSocket)
* **Query Parameters:**
  * `text_prompt`: String specifying persona / role instructions.
  * `voice_prompt`: Filename of the voice prompt (e.g., `NATF2.pt`).
  * `seed`: Optional random seed integer for reproducibility.
  * `audio_temperature`, `text_temperature`, `audio_topk`, `text_topk`: Generation parameters.
* **CLI Flags:**
  * `--host`: Host to bind (default `localhost`).
  * `--port`: Port number (default `8998`).
  * `--device`: Compute device (`cuda`, `cpu`, default: `cuda`).
  * `--cpu-offload`: Offloads LM model layers to host CPU memory when GPU memory is limited (requires `accelerate`).
  * `--tokenizer`: Path to custom SentencePiece tokenizer file (default: downloaded from HF `tokenizer_spm_32k_3.model`).
  * `--moshi-weight`: Path to LM checkpoint (default: `model.safetensors`).
  * `--mimi-weight`: Path to Mimi checkpoint (default: `tokenizer-e351c8d8-checkpoint125.safetensors`).
  * `--voice-prompt-dir`: Directory storing voice embeddings `.pt` files.
  * `--ssl`: Path to directory containing `key.pem` and `cert.pem` for HTTPS/WSS.

### 3.2 Offline Batch Inference: `python -m moshi.offline`
* **File:** `moshi/moshi/offline.py`
* **Purpose:** File-to-file batch evaluation. Reads a user input `.wav`, streams 1920-sample chunks through Mimi and the Moshi LM, and generates an output `.wav` and token trace `.json` of matching duration.
* **CLI Arguments:**
  * `--input-wav`: Path to input WAV (converted to 24kHz mono).
  * `--output-wav`: Destination path for agent spoken output WAV.
  * `--output-text`: Destination path for text transcript JSON.
  * `--voice-prompt`: Path or filename for voice conditioning (e.g. `NATF2.pt`).
  * `--text-prompt`: System instruction string.
  * `--seed`: PRNG seed.
  * `--cpu-offload`: Enables offloading for low-VRAM environments.

### 3.3 Python In-Memory API
* `loaders.get_mimi(weight_path, device)`: Loads the Mimi neural audio encoder/decoder.
* `loaders.get_moshi_lm(weight_path, device, cpu_offload=False)`: Loads the 7B Transformer LM (`LMModel`).
* `sentencepiece.SentencePieceProcessor(tokenizer_path)`: Loads the 32k SentencePiece tokenizer.
* `LMGen(lm, sample_rate=24000, frame_rate=12.5, ...)`: Stateful generator managing autoregressive rollouts, prompt conditioning, and frame generation.

---

## 4. WebSocket Streaming Protocol Specification

The WebSocket connection uses a binary protocol where the first byte (`data[0]`) indicates the message type:

| Type Byte | Message Kind | Payload Format | Description |
|:---|:---|:---|:---|
| `0x00` | **Handshake** | 2 bytes (`version`, `model`) | Sent by server once prompt initialization is complete. Signals client that audio streaming can commence. |
| `0x01` | **Audio** | Opus packet / raw PCM byte stream | Client sends user microphone audio chunks; server streams back agent voice chunks. |
| `0x02` | **Text** | UTF-8 encoded text chunk | Streaming text tokens generated by the model (special token `▁` is converted to space). |
| `0x03` | **Control** | 1 byte (`0=start`, `1=endTurn`, `2=pause`, `3=restart`) | Session flow control signals. |
| `0x04` | **Metadata**| UTF-8 JSON payload | Session configuration, latency measurements, or turn status. |
| `0x05` | **Error** | UTF-8 string | Error descriptions returned to client. |
| `0x06` | **Ping** | Empty payload | Connection heartbeat. |

---

## 5. Voice and Text Prompt Conditioning

PersonaPlex conditions the conversation using two decoupled modalities:

### 5.1 Voice Prompts (Vocal Identity)
* Audio characteristics (timbre, pitch, dialect, cadence) are injected via acoustic conditioning embeddings.
* Supplied either as:
  1. Pre-computed embedding `.pt` files (`lm_gen.load_voice_prompt_embeddings(path)`), or
  2. Raw audio reference `.wav` files (`lm_gen.load_voice_prompt(path)`).
* The model repository ships with **18 preset voice embeddings**:
  * **Natural Female (NATF):** `NATF0.pt`, `NATF1.pt`, `NATF2.pt`, `NATF3.pt`
  * **Natural Male (NATM):** `NATM0.pt`, `NATM1.pt`, `NATM2.pt`, `NATM3.pt`
  * **Variety Female (VARF):** `VARF0.pt`, `VARF1.pt`, `VARF2.pt`, `VARF3.pt`, `VARF4.pt`
  * **Variety Male (VARM):** `VARM0.pt`, `VARM1.pt`, `VARM2.pt`, `VARM3.pt`, `VARM4.pt`
* **Prompt Phase Sequence in LMGen:**
  1. Stepping the voice prompt tokens through the model (forced agent audio channels).
  2. Appending an audio silence spacer: `int(0.5 * mimi.frame_rate)` = ~6 frames (0.5 seconds).

### 5.2 Text Prompts (Behavior & Persona)
* Persona, domain facts, behavioral constraints, and instructions are supplied as plain text.
* The system wraps the text in special delimiters:
  ```
  <system> {user_text_prompt} <system>
  ```
* The wrapped text is tokenized with `tokenizer_spm_32k_3.model` (`SentencePieceProcessor`).
* The token sequence is fed into text channel ($k=0$) of the LM.
* Followed by a second 0.5s audio silence spacer before interactive dialogue begins.

---

## 6. Hardware & VRAM Requirements

| Configuration | Model Weights Precision | Required VRAM | Target Hardware | Notes |
|:---|:---|:---|:---|:---|
| **Full Precision / BF16** | 16-bit BF16/FP16 | **14 - 16 GB** (weights only), **~20 - 24 GB** total | NVIDIA RTX 3090, 4090, A10G, A5000, A100 | Recommended for production. Guarantees step latency < 80ms for real-time streaming. |
| **Quantized (Q8 / Q4)** | 8-bit / 4-bit INT | **5 - 8 GB** | NVIDIA RTX 3070, 3080, 4070 | Viable for local workstations and developer environments. |
| **CPU Offload (`--cpu-offload`)** | Split GPU / CPU RAM | **4 - 8 GB VRAM** + **16+ GB System RAM** | Consumer GPUs (e.g. RTX 3050 4GB) | Supported via HuggingFace `accelerate`. Step time may exceed 80ms budget on weaker CPUs. |
| **Mimi Neural Codec** | FP32 / FP16 | **~500 MB** | Co-located on GPU or CPU | Extremely fast convolutional/transformer encoder-decoder. |

---

## 7. Concurrency & Stream Capacity per GPU

### 7.1 Single Instance Limitation
* The official `moshi.server` implementation wraps the active chat handler with an exclusive lock:
  ```python
  self.lock = asyncio.Lock()
  async with self.lock:
      ...
  ```
* Because `MimiModel` and `LMGen` maintain persistent internal streaming buffers (KV caches, delay lines, causal temporal conv buffers), **a single in-memory model instance can only serve 1 active stream at a time**.

### 7.2 Scaling to Multiple Concurrent Streams
* **Worker Process Isolation:** To serve $N$ concurrent streams on a GPU or cluster, the orchestration layer must manage $N$ worker instances (either as distinct Python worker processes or containerized replicas).
* **GPU Memory Footprint per Worker:**
  * On a 24 GB GPU: 1 full-precision 7B model occupies ~16-20 GB, limiting the GPU to **1 active worker**. With 4-bit or 8-bit quantization, 2–3 worker processes can fit on a 24 GB GPU.
  * On an 80 GB A100/H100: 3–4 full-precision workers can run concurrently in parallel processes (or via multi-stream batched engines like vLLM-Omni).
* **Hard Real-Time Latency Constraint:**
  * For full-duplex speech, each 80ms audio frame must be generated in **strictly less than 80ms** (target: 30–50ms). If multiple sessions run on the same GPU compute cores and step latency exceeds 80ms, audio underruns ("clicks" and buffering pauses) will occur.
* **Orchestration Requirement:**
  * Phase 1 must feature a **Worker Pool Manager / Dispatcher** that tracks worker state (`IDLE`, `BUSY`, `UNHEALTHY`), routes new sessions to idle workers, rejects or queues sessions when capacity is exhausted, and provides health checks and graceful restarts.

---

## 8. Licensing & Usage Terms

* **Source Code:** Released under the permissive **MIT License** (`SPDX-License-Identifier: MIT`). Allows open-source modification, redistribution, and commercial deployment.
* **Model Weights (`nvidia/personaplex-7b-v1`):** Released under the **NVIDIA Open Model License Agreement**.
  * Free for research and commercial use subject to acceptable use policies and standard NVIDIA attribution requirements.
  * Download requires accepting terms on Hugging Face and providing `HF_TOKEN`.

---

## 9. Design Implications for Phase 1 Orchestration Layer

1. **Protocol Compatibility:** Support the native PersonaPlex binary WebSocket protocol (`0x00` handshake, `0x01` audio, `0x02` text, `0x03` control, `0x04` metadata, `0x05` error) so that standard PersonaPlex Web clients, telephone bridges, and external endpoints connect seamlessly.
2. **Audio Streaming Pipeline:** Provide frame ingestion, buffering, and resampling to 24 kHz mono float/PCM, with chunking at 1,920 samples (80ms frames).
3. **Session & Prompt Management:** Support declarative configuration of voice prompts (preset `.pt` selection or custom `.wav`) and system text prompt wrapping (`<system> ... <system>`).
4. **Worker Pool & Concurrency Management:** Manage backend worker processes, track availability, support simulated/mock workers for environments without 24GB GPUs (e.g. CI, tests, and laptops), and gracefully handle worker failures.
5. **Telemetry & Turn Observability:** Track frame-by-frame round-trip latency, audio underruns, text token transcription, barge-in / interruption events, and session state transitions.
