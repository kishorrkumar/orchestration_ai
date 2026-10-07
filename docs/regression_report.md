# Audio Pipeline Regression Report: Transparent Binary Relay vs. Gateway Transcoding

**Date:** 2026-10-07  
**Scope:** Voice Pipeline Architecture in `orchestration_ai`

---

## 1. Commit Analysis & History

| State | Commit SHA | Date | Commit Message | Description |
|---|---|---|---|---|
| **Good (Working)** | `6dafea7c` | 2026-10-01 13:28:26 +0530 | `fix(transcription): activate webkitSpeechRecognition, continuous Ogg Opus stream decoding, and sync system_prompt override [DBG-8]` | **Transparent Binary Relay**: Gateway acted as a pure protocol proxy (`/v1/realtime`). Audio frames and text tokens from worker were forwarded raw via `encode_message(worker_msg)` directly to the client WebSocket without server-side decode/resample. |
| **Breaking Commit** | `259ff65` | 2026-10-03 | `feat(phase3): implement full-duplex S2S runtime on /v2/voice with 16k/8k routing and turn logging` | **Introduced Gateway Transcoding**: Replaced transparent relay with server-side 16k PCM16 decoding, resamplers, software clipping, and `sphn.OpusStreamWriter` / `OpusStreamReader` bridging. |

---

## 2. What Changed Between the Two Commits

### In `6dafea7` (Good Architecture)
```python
# orchestration/session/manager.py
async def run_worker_forwarder(self, send_to_client_fn) -> None:
    async for worker_msg in self.worker.recv_messages():
        if isinstance(worker_msg, AudioMessage):
            # Transparent forward of raw worker audio frames (0x01)
            await send_to_client_fn(encode_message(worker_msg))
        elif isinstance(worker_msg, TextMessage):
            # Transparent forward of text tokens (0x02)
            await send_to_client_fn(encode_message(worker_msg))
```
- Client and worker communicated using native 24 kHz Ogg-Opus binary packets.
- The browser encoded mic audio via `opus-recorder` directly into valid Ogg pages.
- The worker generated Ogg pages directly from Mimi tokens via `opus_writer`.
- The gateway performed session leasing, prompt parameter construction, and byte forwarding.

### In `259ff65` (Regression Architecture)
```python
# orchestration/api/voice_v2.py
# Inbound (Client -> Gateway -> Worker):
pcm16 = np.frombuffer(data, dtype=np.int16)
f32_samples = int16_to_float32(pcm16)
frames_24k = in_buffer.push_chunk(f32_samples)
for frame_24k in frames_24k:
    await worker_client.send_audio(frame_24k)  # Uses sphn.OpusStreamWriter

# Outbound (Worker -> Gateway -> Client):
# worker_client decodes with sphn.OpusStreamReader -> f32 24k
resampled = out_resampler.resample_chunk(f32_worker, last=False)  # 24k -> 16k
clipped = soft_clip(resampled, threshold=0.92)
pcm_bytes = float32_to_int16(clipped).tobytes()
await websocket.send_bytes(pcm_bytes)
```

---

## 3. Why the Transcoding Path Broke Audio (Root Cause)

1. **`sphn.OpusStreamWriter` Initial Buffering:**
   - When PCM frames are fed to `sphn.OpusStreamWriter`, the encoder buffers samples to form Opus packet frames.
   - For the first 3–4 frames (each 80 ms), `read_bytes()` returns `b""` (0 bytes).
   - In `worker_client.send_audio`, `if not payload: return` dropped those initial packets.
   - The upstream worker's `opus_loop` received no initial Ogg BOS header or valid container pages.
2. **Worker Inference Starvation:**
   - In `moshi/server.py`, `opus_reader.read_pcm()` only returns samples when valid continuous Ogg pages have been ingested.
   - Because the initial Ogg BOS header was missing or fragmented by intermediate queues, `opus_reader` produced 0 PCM samples, leaving the worker's Mimi neural encoder starved for input.
3. **Double Resampling & Degradation:**
   - Resampling 16 kHz web mic audio up to 24 kHz for Mimi, then downsampling generated 24 kHz speech to 16 kHz int16 for the browser, introduced phase smearing, clipping artifacts, and CPU overhead.
4. **Client Inactivity Timeout (10-Second Drops):**
   - The client watchdog disconnects if no network message is received for 10 seconds.
   - Because the gateway was silent while waiting for transcoding buffers or priming to complete, the client closed the socket at exactly 10.0 seconds.

---

## 4. Remediation

- **Revert to Transparent Binary Relay:**
  - Forward raw `0x01` (Ogg-Opus audio), `0x02` (text tokens), and `0x03` (control actions) in both directions untouched.
  - Zero decoding, zero resampling, and zero re-encoding on the gateway.
  - Preserve all Ogg header pages from the browser.
- **Client Native Pipeline:**
  - Browser mic -> `opus-recorder` -> `0x01` binary frame.
  - Worker `0x01` binary frame -> `decoderWorker.min.js` (WASM libopus) -> `AudioWorklet` (`MoshiProcessor`) with 80 ms jitter buffer.
