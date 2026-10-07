# NVIDIA PersonaPlex-7B (Kyutai Moshi) Protocol Specification

This document details the exact network, framing, encoding, and prompt conditioning protocol implemented by the installed PersonaPlex / Moshi server (`moshi/server.py` and its imported modules).

---

## 1. Upstream Server Endpoint & Parameters

**Endpoint:** `GET /api/chat` (upgraded to WebSocket via `aiohttp.web.WebSocketResponse`)  
**Source References:**  
- `moshi/server.py`, lines 135–172 (`handle_chat`)
- `moshi/server.py`, lines 459–460 (`app.router.add_get("/api/chat", state.handle_chat)`)

### Query Parameters

| Parameter | Type | Required / Default | Upstream Source Reference | Notes & Observed Server Quirks |
|---|---|---|---|---|
| `text_prompt` | `str` | Required (can be empty string) | `moshi/server.py:170` | Text conditioning prompt. Wrapped with `<system> ... <system>` tags if not present. Encoded with SentencePiece tokenizer into token IDs. |
| `voice_prompt` | `str` | Required (if `--voice-prompt-dir` configured) | `moshi/server.py:151-169` | Filename of reference voice prompt located inside `--voice-prompt-dir`. If `.pt`, loaded as precomputed embeddings (`load_voice_prompt_embeddings`). If `.wav`, loaded as audio (`load_voice_prompt`). |
| `seed` | `int` | Optional (default: `None` / `42424242`) | `moshi/server.py:171` | **Warning / Server Bug:** Line 171 has `seed = int(request["seed"]) if "seed" in request.query else None`. In `aiohttp`, `request` dictionary subscripting raises `KeyError` unless present in `request` attributes. Do not pass `seed` unless fixed upstream. |
| `audio_temperature` | `float` | Default: `0.8` (in `LMGen`) | `moshi/server.py:143` | **Note:** Commented out in upstream `server.py` (`# self.lm_gen.temp = ...`). Passing it in URL is tolerated by `aiohttp` but ignored by the worker. |
| `text_temperature` | `float` | Default: `0.7` (in `LMGen`) | `moshi/server.py:144` | **Note:** Commented out in upstream `server.py`. Ignored by the worker. |
| `audio_topk` | `int` | Default: `250` (in `LMGen`) | `moshi/server.py:146` | **Note:** Commented out in upstream `server.py`. Ignored by the worker. |
| `text_topk` | `int` | Default: `25` (in `LMGen`) | `moshi/server.py:145` | **Note:** Commented out in upstream `server.py`. Ignored by the worker. |

---

## 2. Binary Message Framing

**Frame Structure:**  
All messages on the WebSocket are binary (`aiohttp.WSMsgType.BINARY`).  
The first byte (`payload[0]`) indicates the message type (`kind`), followed by the message payload (`payload[1:]`).

**Source Reference:** `moshi/server.py:194-200`, `moshi/server.py:239-251`, `_personaplex_upstream/client/src/protocol/encoder.ts:9-33`.

### Client -> Worker (Inbound)

| Kind Byte | Name | Payload | Behavior in `moshi/server.py` |
|---|---|---|---|
| `0x01` | Audio Frame | Ogg-Opus stream bytes | Appended to `opus_reader` (`opus_reader.append_bytes(payload)`). |
| *Other* | Any other byte | Ignored | Server logs `clog.log("warning", f"unknown message kind {kind}")` and discards. |

### Worker -> Client (Outbound)

| Kind Byte | Name | Payload | Behavior in `moshi/server.py` |
|---|---|---|---|
| `0x00` | Handshake | None (length 1: `b"\x00"`) | Emitted **only once**, immediately after `step_system_prompts_async` completes (`server.py:288`). Indicates the worker has completed priming and is ready for streaming. |
| `0x01` | Audio Frame | Ogg-Opus stream bytes | Read from `opus_writer` (`opus_writer.read_bytes()`) and prefixed with `b"\x01"` (`server.py:251`). Encodes 24 kHz audio. |
| `0x02` | Text Token | UTF-8 string | Emitted whenever LM produces a non-padding token (`server.py:239`). Token ID converted via `text_tokenizer.id_to_piece`, space marker `▁` replaced with `' '`. |

---

## 3. Audio Encoding, Sample Rate, and Framing

**Source References:**  
- `moshi/server.py:105, 218-223, 263-264`
- `moshi/models/loaders.py:39-40` (`SAMPLE_RATE = 24000`, `FRAME_RATE = 12.5`)
- `sphn` (C++ libopus wrapper)

1. **Sample Rate:** `24,000 Hz` strictly.
2. **Frame Rate:** `12.5 Hz` (1 step per 80 ms).
3. **Frame Size:** `1920 samples` per step ($24000 / 12.5 = 1920$).
4. **Wire Encoding:**
   - **Worker Inbound:** Ogg-Opus stream at 24 kHz created by `sphn.OpusStreamWriter(24000)`.
   - **Worker Outbound:** Ogg-Opus stream at 24 kHz unpacked by `sphn.OpusStreamReader(24000)`.
   - Note on Ogg-Opus stream initialization: The first 1–2 frames contain Ogg headers (`OpusHead`, `OpusTags`, ~47–53 bytes). The reader yields 0 PCM samples until the container headers are parsed, after which it yields exactly 1920 Float32 samples per frame.

---

## 4. Text Prompt Wrapping

**Source Reference:** `moshi/server.py:79-86` (`wrap_with_system_tags`)

```python
def wrap_with_system_tags(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"
```

- System prompt tokens are encoded by `SentencePiece` (`tokenizer_spm_32k_3.model`).
- Must begin and end with `<system>` tags with surrounding whitespace.
- Markdown headers (`#`), bullet points (`-`), asterisks (`**`), emojis, and quotation scripts should be stripped prior to wrapping, as the S2S model treats all tokens as spoken context.

---

## 5. Handshake & Priming Mechanics

**Source Reference:** `moshi/server.py:259-290`

1. Connection opens (`/api/chat`).
2. Global server lock acquired (`async with self.lock:`).
3. Codecs reset: `mimi.reset_streaming()`, `lm_gen.reset_streaming()`, `opus_writer`, `opus_reader`.
4. Priming sequence executed via `await self.lm_gen.step_system_prompts_async(self.mimi, is_alive=is_alive)`:
   - Stepping through voice prompt (~10–15 s audio @ 12.5 Hz = ~125–180 steps).
   - 0.5 s audio silence (6 steps).
   - Stepping through all text prompt tokens (e.g., 200 tokens = 200 steps).
   - 0.5 s audio silence (6 steps).
   - Total steps: ~350–400 forward passes on the 7B model. On an NVIDIA A100 GPU, this takes ~8.5–9.5 seconds ("Priming wait").
5. Only when priming is complete, `b"\x00"` is sent to the client.
6. The connection then enters the concurrent bidirectional loop (`recv_loop`, `opus_loop`, `send_loop`).

---

## 6. Behavior When User Is Silent

**Source Reference:** `moshi/server.py:204-243` (`opus_loop`)

- The inference stepping loop is **clocked by incoming audio**:
  ```python
  pcm = opus_reader.read_pcm()
  ...
  while all_pcm_data.shape[-1] >= self.frame_size:
      chunk = all_pcm_data[: self.frame_size]
      all_pcm_data = all_pcm_data[self.frame_size:]
      codes = self.mimi.encode(chunk)
      tokens = self.lm_gen.step(codes[:, :, c: c + 1])
  ```
- If the user sends nothing (or socket is silent without packets), `all_pcm_data` does not accumulate 1920 samples, and `self.lm_gen.step()` is **never called**. The model generation freezes.
- **Critical Requirement:** The client or orchestrator MUST stream audio continuously (either live microphone or silent PCM frames `np.zeros(1920, dtype=np.float32)`) at 12.5 Hz (every 80 ms) for the conversation to progress and for the agent's greeting or reply to be generated.

---

## 7. Session Locking & Concurrency

**Source Reference:** `moshi/server.py:95, 114, 259`

- The worker has a single `self.lock = asyncio.Lock()`.
- Only **one** WebSocket connection can be actively served by a worker process at any given moment.
- Any subsequent connection attempt will hang in `async with self.lock:` until the active session closes its connection and releases the lock.
- If a client disconnects during priming or active conversation, clean cancellation of `tasks` and release of the lock is mandatory.

---

## 8. Custom Voice Prompts & Embedding Cache

**Source Reference:** `moshi/server.py:151-169`, `moshi/models/lm.py:1000-1065`

1. **Storage Directory:** Configured via `--voice-prompt-dir`.
2. **File Formats:**
   - **Precomputed Embeddings (`.pt`):** Saved via PyTorch containing pre-stepped embeddings and KV cache. Skips raw audio encoding and speeds up priming.
   - **Reference Audio (`.wav`):** Mono audio, 24 kHz (auto-resampled by `sphn` if needed), 10–20 seconds long.
3. **Loading:** Passed via `voice_prompt=<filename>` in query string. Worker resolves `os.path.join(voice_prompt_dir, filename)`. If the file does not exist, worker raises `FileNotFoundError`.
