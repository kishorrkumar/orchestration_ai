# PersonaPlex Full-Duplex Speech-to-Speech Protocol Specification

**Source Ground Truth:**
- Upstream Worker: `moshi/server.py` (`_personaplex_upstream/moshi/moshi/server.py`)
- Official Client: `_personaplex_upstream/client/src/`
- Reference Repo: `github.com/NVIDIA/personaplex`

---

## 1. WebSocket Endpoint & Connection Handshake

### 1.1 Endpoint URI & Query Parameters
Clients connect to:
`ws://<worker_host>:<worker_port>/api/chat?<query_params>`

Reference: `_personaplex_upstream/client/src/pages/Conversation/Conversation.tsx` (lines 56–76) & `_personaplex_upstream/moshi/moshi/server.py` (lines 143–175)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text_prompt` | `string` | `""` | Agent system scenario wrapped in `<system> ... <system>`. |
| `voice_prompt` | `string` | `"NATF2.pt"` | Filename of the target voice embedding preset (e.g. `NATF0.pt` to `VARM4.pt`). Joined with `--voice-prompt-dir`. |
| `audio_temperature` | `float` | `0.8` | Temperature for audio codebook generation (0.0 – 2.0). |
| `text_temperature` | `float` | `0.7` | Temperature for text token generation (0.0 – 2.0). |
| `audio_topk` | `int` | `250` | Top-K sampling cutoff for Mimi audio codebooks. |
| `text_topk` | `int` | `25` | Top-K sampling cutoff for text LM token sampling. |
| `seed` | `int` | `None` | Optional RNG seed for deterministic model execution. |

### 1.2 Handshake Sequence (Opcode 0x00)
1. Upon WebSocket connection, the worker accepts the socket (`await ws.prepare(request)`).
2. The worker loads the voice prompt embeddings (`lm_gen.load_voice_prompt_embeddings`) and encodes the system prompt tokens (`lm_gen.text_prompt_tokens`).
3. The worker executes `lm_gen.step_system_prompts_async(self.mimi, is_alive=is_alive)` (priming phase: takes **~9.4 seconds** on NVIDIA A100 GPU).
4. Once system prompt stepping is complete, the worker transmits a single **Handshake Byte** (`0x00`):
   ```python
   # moshi/server.py: line 307
   await ws.send_bytes(b"\x00")
   ```
5. The client transitions from `connecting` / `priming` state to `connected` / `ready` state upon receiving this `0x00` byte (`useSocket.ts: line 57`).
6. Interactive bidirectional audio streaming begins immediately after `0x00`.

---

## 2. Binary Framing & Message Types

All messages exchanged over `/api/chat` after handshake are binary packets. The **first byte (byte 0)** is the opcode (Kind / MessageType).

Reference: `_personaplex_upstream/client/src/protocol/types.ts` & `encoder.ts` (lines 9–89)

| Opcode | Name | Direction | Payload Structure |
|---|---|---|---|
| `0x00` | `Handshake` | Worker -> Client | Version (byte 1), Model (byte 2). |
| `0x01` | `Audio` | Both directions | Ogg-Opus page byte sequence (`b"OggS..."`). |
| `0x02` | `Text` | Both directions | UTF-8 encoded text chunk (`bytes(token, "utf-8")`). |
| `0x03` | `Control` | Client -> Worker | Single byte action: `0x00` (start), `0x01` (endTurn), `0x02` (pause/interrupt), `0x03` (restart). |
| `0x04` | `Metadata` | Worker -> Client | UTF-8 JSON payload. |
| `0x05` | `Error` | Worker -> Client | UTF-8 encoded error description string. |
| `0x06` | `Ping` | Client -> Worker | Keepalive probe (single byte `0x06`). |

---

## 3. Audio Pipeline Specifications

### 3.1 Worker Audio Model Requirements
- **Audio Sample Rate:** Exactly **24,000 Hz** (24 kHz mono).
- **Frame Size:** Exactly **1,920 samples** (80 ms at 24 kHz = 12.5 Hz cadence).
- **Audio Codec:** Native **Ogg-Opus container streaming**.
- **Container Structure:**
  - The worker uses `sphn.OpusStreamReader(24000)` and `sphn.OpusStreamWriter(24000)`.
  - The browser transmits valid continuous Ogg pages beginning with the Ogg BOS (`Beginning of Stream`) header containing the `OpusHead` and `OpusTags` packet headers.
  - The worker's `opus_reader.append_bytes(payload)` consumes Ogg pages and produces 24 kHz PCM chunks for neural model encoding.
  - The worker's `opus_writer.append_pcm(main_pcm)` encodes generated 24 kHz audio chunks and produces Ogg-Opus pages (`b"OggS..."`).
  - The worker transmits these Ogg pages over WebSocket prefixed with `0x01`:
    ```python
    # moshi/server.py: line 264
    await ws.send_bytes(b"\x01" + msg)
    ```

### 3.2 Client-Side Microphone Capture
Reference: `_personaplex_upstream/client/src/pages/Conversation/hooks/useUserAudio.ts` (lines 67–85)
- The official client captures the user microphone via `opus-recorder`:
  ```javascript
  const recorderOptions = {
    mediaTrackConstraints: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
    encoderPath: "opus-recorder/dist/encoderWorker.min.js",
    bufferLength: Math.round(960 * audioContext.sampleRate / 24000),
    encoderFrameSize: 20,       // 20 ms frames
    encoderSampleRate: 24000,   // 24 kHz target rate
    maxFramesPerPage: 2,        // 2 frames = 40 ms per Ogg page
    numberOfChannels: 1,        // Mono
    recordingGain: 1,
    resampleQuality: 3,
    encoderComplexity: 0,
    encoderApplication: 2049,   // OPUS_APPLICATION_VOIP
    streamPages: true,          // Continuous Ogg streaming with headers
  };
  ```
- Every generated chunk is emitted via `ondataavailable` and wrapped into a message prefixed with `0x01`:
  `new Uint8Array([0x01, ...oggPageBytes])`

### 3.3 Client-Side Audio Playback & Jitter Buffer
Reference: `_personaplex_upstream/client/src/decoder/decoderWorker.ts` & `_personaplex_upstream/client/src/audio-processor.ts`
- **Decoder WebWorker (`decoderWorker.min.js`):**
  - Receives incoming `0x01` payloads.
  - Runs WebAssembly libopus decoder to decompress Ogg-Opus pages back to raw 24 kHz float32 PCM samples.
  - Pre-warmed on page load using a synthetic warmup BOS page (`createWarmupBosPage()`) to eliminate decoder startup latency.
- **AudioWorklet (`MoshiProcessor` in `audio-processor.ts`):**
  - Decoded float32 PCM frames are posted to an `AudioWorkletNode` running `MoshiProcessor`.
  - Maintains an initial buffer of 80 ms (`initialBufferSamples = 1920 samples`) and an adaptive jitter buffer:
    - Snap delay: 20 ms.
    - Max buffer: up to 80 ms before dropping oldest frames to keep interactive conversational latency tight.
  - Plays audio through the Web Audio `AudioContext` destination.

---

## 4. Jitter-Buffer, Keepalive & Inactivity Disconnect Behavior

Reference: `_personaplex_upstream/client/src/pages/Conversation/hooks/useSocket.ts` (lines 103–115)
1. The client maintains an inactivity watchdog:
   ```typescript
   if (lastMessageTime.current && Date.now() - lastMessageTime.current > 10000) {
     console.log("closing socket due to inactivity", socketRef.current);
     socketRef.current?.close();
   }
   ```
2. **Why calls dropped at ~10 seconds:**
   - If the gateway was silent during the ~9.4 s priming phase or did not forward messages within 10 seconds, the client's inactivity watchdog aborted the connection at 10.0 seconds.
3. **Keepalive Solution:**
   - During the priming phase, the gateway emits continuous status keepalive heartbeats (`{"type": "status", "status": "priming", "elapsed_ms": ...}`) every 1.0 second.
   - The client also transmits keepalive pings (`{"type": "ping"}`) every 3.0 seconds.
   - In active conversation, the worker continuously streams 12.5 Hz audio pages (`0x01`) and text tokens (`0x02`), resetting the watchdog timer on every received network packet.

---

## 5. Architectural Rule for the Orchestration Gateway

**Do NOT Transcode in the Gateway:**
- Attempting to decode Ogg-Opus into PCM16, resample 24k -> 16k, and re-encode to Opus with `sphn.OpusStreamWriter` creates buffering delays (the first 3–4 frames return 0 bytes) and corrupts Ogg container headers, causing silence in the worker's `opus_loop`.
- The gateway MUST operate as a **transparent binary relay**:
  - Browser Ogg-Opus (`0x01`) -> Gateway (verifies/logs) -> Worker unchanged.
  - Worker Ogg-Opus (`0x01`) -> Gateway (verifies/logs) -> Browser unchanged.
