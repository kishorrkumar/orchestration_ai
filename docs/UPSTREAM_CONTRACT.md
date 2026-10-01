# Upstream PersonaPlex (Moshi) Server Contract

**Source Reference**: `_personaplex_upstream/moshi/moshi/server.py` and `_personaplex_upstream/client/src/protocol/`

---

## 1. WebSocket Endpoint & Query Parameters

* **URL Path**: `ws://<host>:<port>/api/chat`
* **Query Parameters Defined by Upstream**:
  | Parameter | Type | Required | Default | Description |
  | :--- | :--- | :--- | :--- | :--- |
  | `text_prompt` | string | Yes | `""` | Conditioning text prompt. Wrapped by upstream if missing `<system>` tags (`wrap_with_system_tags`). |
  | `voice_prompt` | string | Yes | (Required if `--voice-prompt-dir` set) | File name of voice preset inside `voice_prompt_dir` (e.g. `NATM0.pt` or `NATF2.pt`). |
  | `audio_temperature` | float | No | 0.8 | Temperature for audio token generation. |
  | `text_temperature` | float | No | 0.7 | Temperature for text token generation. |
  | `audio_topk` | int | No | 250 | Top-k sampling bound for audio. |
  | `text_topk` | int | No | 25 | Top-k sampling bound for text. |
  | `seed` | int | No | -1 / None | RNG seed for deterministic generation. |

> [!WARNING]
> Upstream `server.py` **only** accepts the parameters listed above. Any extra parameters (`accent`, `character`, `neural_voice`, `call_flow`, `persona_id`) are gateway-internal and MUST NOT be forwarded in the upstream URL.

---

## 2. Handshake & Prompt Priming Lifecycle

1. **Connection & Concurrency Lock**:
   - `moshi.server` acquires an exclusive `async with self.lock:`.
   - Strictly **1 concurrent session per worker process**.
2. **System Prompt Conditioning (Priming Phase)**:
   - Runs `await self.lm_gen.step_system_prompts_async(self.mimi, is_alive=is_alive)`.
   - On an NVIDIA A100 GPU, this takes **4.0 to 7.0 seconds**.
   - During this time, `is_alive()` polls `ws.receive()` with a 10 ms timeout.
   - **CRITICAL**: The client MUST NOT send audio frames during priming. Any early message can be consumed by `is_alive()` or cause frame buffer misalignment.
3. **Handshake Emission**:
   - Upstream sends single binary byte: `b"\x00"` (`MessageType.HANDSHAKE`).
   - Only after receiving `0x00` is the worker ready to accept and produce audio.

---

## 3. Wire Protocol & Message Formats

All WebSocket frames are **Binary** (`aiohttp.WSMsgType.BINARY`).

| Kind Byte | Wire Header | Payload Encoding | Description | Direction |
| :--- | :--- | :--- | :--- | :--- |
| `0x00` | `0x00` (1 byte) | None (or version/model bytes) | Handshake acknowledgement. Sent by server when priming completes. | Server -> Client |
| `0x01` | `0x01` (1 byte) | **Ogg Opus bitstream** (variable length Ogg pages) | Audio stream. MUST be valid Ogg Opus container pages at 24,000 Hz. | Bi-directional |
| `0x02` | `0x02` (1 byte) | UTF-8 encoded string | Text token emitted by Moshi LM transformer. | Server -> Client |
| `0x03` | `0x03` (1 byte) | 1 byte action (`0x00`..`0x03`) | Control signal (upstream logs warning on unknown kind). | Client -> Server |

> [!IMPORTANT]
> **Audio Payload MUST Be Ogg Opus**:
> - Upstream uses `sphn.OpusStreamReader(24000)` on incoming `0x01` payloads.
> - Upstream uses `sphn.OpusStreamWriter(24000)` to produce outgoing `0x01` payloads.
> - Sending raw PCM to upstream causes `opus_reader.read_pcm()` to fail, instantly terminating `opus_loop` and closing the connection with code `1000`.

---

## 4. Audio Cadence & Concurrency Model

* **Sample Rate**: 24,000 Hz Mono.
* **Frame Size**: 1,920 samples (80 ms @ 12.5 Hz frame rate).
* **Streaming Behavior**: Full-duplex continuous loop. Each incoming 80 ms audio frame drives 1 step of Moshi LM and produces 1 outgoing 80 ms audio frame.
* **Continuous Input Stream**: The worker MUST receive a continuous audio stream at real-time 80 ms intervals (including silence frames during pauses). Gating on VAD stalls generation.
* **Stream Termination**:
  - `recv_loop`, `opus_loop`, and `send_loop` run via `asyncio.wait(return_when=FIRST_COMPLETED)`.
  - When the client disconnects or closes, the server terminates the remaining tasks and closes with code `1000 (OK)`.
