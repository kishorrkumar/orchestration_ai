# PersonaPlex Voice Agent Platform

> Full-duplex speech-to-speech voice agent powered by **NVIDIA PersonaPlex-7B-v1** running on an NVIDIA A100 GPU.
> Features a single-agent declarative architecture (`agent.yaml`), AudioWorklet streaming, pre-warmed priming, and WebRTC/WebSocket transports.

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.13-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-v2-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 1. System Architecture

```
                                      ┌─────────────────────────────────────────────────────────────┐
                                      │                      FastAPI Gateway                        │
                                      │                        (Port 8000)                          │
                                      │                                                             │
┌───────────────────────┐             │  ┌──────────────────────┐      ┌─────────────────────────┐  │      ┌──────────────────────────┐
│     Browser Mic       │             │  │   Transport Ingress  │      │     Continuous Pacer    │  │      │   PersonaPlex 7B S2S     │
│  (16 kHz PCM / Opus)  │ ──────────> │  │ WebSocket / WebRTC   │ ───> │  (12.5 Hz, 1920 frames) │ ────> │   Worker (Port 8998)     │
└───────────────────────┘             │  └──────────────────────┘      └─────────────────────────┘  │      │  (Kyutai Mimi, 24 kHz)   │
                                      │             │                               │               │      └──────────────────────────┘
                                      │             ▼                               ▼               │                    │
┌───────────────────────┐             │  ┌──────────────────────┐      ┌─────────────────────────┐  │                    │
│    Browser Speaker    │ <────────── │  │ Jitter Buffer / VAD  │ <─── │   Binary Wire Decoder   │ <──────────────────┘
│  (16 kHz / WebRTC)    │             │  │   Barge-In Flush     │      │  (0x00 / 0x01 / 0x02)   │  │    (Ogg-Opus / UTF-8 tokens)
└───────────────────────┘             │  └──────────────────────┘      └─────────────────────────┘  │
                                      └─────────────────────────────────────────────────────────────┘
```

PersonaPlex 7B handles listening, turn-taking, backchanneling, and speech generation in a single neural loop. The gateway orchestrator owns session state, audio resampling, silence clocking, template interpolation, VAD metrics, and transport negotiation.

---

## 2. Quickstart on Krutrim Cloud Pod (A100 40GB)

### 2.1 Pod Environment
- **OS:** Linux Ubuntu (NVIDIA A100 SXM4 40GB)
- **Python:** Python 3.13 venv at `/home/jovyan/personaplex_env`
- **Hugging Face Cache:** `HF_HOME=/workspace/huggingface`
- **Inference Worker:** `127.0.0.1:8998`
- **FastAPI Gateway:** `0.0.0.0:8000` (externally reachable via HTTPS reverse proxy)

### 2.2 Start Services

```bash
# 1. Pull latest code
git pull

# 2. Launch PersonaPlex worker and FastAPI gateway
bash deploy/cloud/start_services.sh
```

### 2.3 Where to Click & How to Test
1. Open the cloud proxy HTTPS URL (e.g. `https://<pod-subdomain>.krutrim.com/`) in Google Chrome.
2. Put on **headphones** (essential to prevent acoustic bleed between mic and speakers).
3. The console automatically initiates a **pre-warm** lease in the background.
4. Click **"Test Call"** or **"Start Call"**.
5. The session transitions from `connecting` $\to$ `ready` instantly ($\approx 20$ ms when pre-warmed).
6. Speak naturally into your microphone: *"Hi, can you hear me?"*
7. The agent listens and responds in full-duplex speech.

---

## 3. Single-Agent Configuration (`agent.yaml`)

The platform is strictly **single-agent**. The active voice agent is configured declaratively in `agent.yaml`:

```yaml
name: "Alex"
timezone: "Asia/Kolkata"
voice_prompt: "NATF2.pt"

# Spoken opening behavior
greeting_mode: "agent_first"
greeting_text: "Hello! This is Alex from {{company}}. How can I help you today?"
ending_text: "Thank you for calling. Have a great day, goodbye!"

# Persona system instructions
system_prompt: >
  You are Alex, a helpful and friendly voice assistant at {{company}}.
  You are speaking with {{caller_name}} over the phone.
  You speak in short, natural sentences, the way real people converse.
  You listen carefully, answer concisely, and verify understanding before moving to the next point.

# Template variable defaults (strictly validated; HTTP 422 if unresolved)
variables:
  company: "BrightNet"
  caller_name: "the caller"
  customer_name: "the customer"

# Neural generation settings
generation:
  audio_temperature: 0.8
  text_temperature: 0.7
  audio_topk: 250
  text_topk: 25
  seed: -1

# Session duration safeguards
session:
  max_duration_sec: 600
  end_silence_sec: 20
```

To create a new configuration, copy `agent.example.yaml`:
```bash
cp agent.example.yaml agent.yaml
```

---

## 4. How to Write a Good PersonaPlex System Prompt

PersonaPlex 7B is an end-to-end speech-to-speech model. It behaves differently than text LLMs: **it mirrors spoken cadence, prosody, and brevity**.

### Principles
1. **Short plain prose:** Write role, background, and scenario in simple conversational sentences.
2. **Keep it under 250 tokens:** Token limit is strictly budgeted. The linter warns above 300 tokens and hard-errors at model capacity.
3. **No rigid scripts:** Avoid writing verbatim quotes (`Say: "..."`). The model will speak awkwardly or talk over itself.
4. **No numbered lists or bullet rules:** The model is not a text formatter. Bullet points degrade spoken prosody.
5. **Describe how the agent talks:** Use phrasing like *"You speak in short, casual sentences"*, *"You pause and ask one question at a time"*.

### Before & After Examples

#### ❌ BAD (Text-LLM Style — Do NOT use)
```
You are an AI customer support bot for BrightNet.
Follow these rules strictly:
1. Always say: "Thank you for calling BrightNet, my name is Alex, how may I direct your call today?"
2. Never make assumptions.
3. If the user asks about billing, refer to section 4.2 of the billing guide.
4. Output your answer in 3 numbered bullet points.
```
*Why this fails:* Long token count increases priming delay; numbered rules make the model sound robotic; script quotes conflict with natural turn-taking.

####  GOOD (Conversational Speech Style — Recommended)
```
You are Alex, a friendly customer support specialist at BrightNet internet.
You are on a phone call with a customer who needs help.
You speak casually and warmly, like a helpful friend.
Keep each response under two sentences and ask one simple question at a time.
Verify that each step worked before moving on.
```
*Why this works:* Low token count (~50 tokens), primes in minimal time, guides natural conversational turn-taking, and produces fluent speech.

---

## 5. Adding a Custom Voice Prompt

PersonaPlex uses voice prompts (`.pt` embeddings or clean audio WAV files) to condition the speaker's vocal timbre, pitch, and accent.

### Audio Requirements
- **Format:** Clean mono WAV (16-bit PCM or 32-bit Float).
- **Sample Rate:** `24,000 Hz` native (16,000 Hz acceptable; gateway resamples automatically).
- **Duration:** **10 to 20 seconds** optimal. Avoid files longer than 30 seconds (excessive tokens/VRAM).
- **Acoustic Quality:** Zero background music, zero room reverb/echo, single speaker speaking continuously and clearly.

### How to Apply
1. Place your WAV file in `data/voices/<name>.wav`.
2. Convert to PersonaPlex `.pt` tensor using Moshi tools or specify in `agent.yaml`:
   ```yaml
   voice_prompt: "my_custom_voice.pt"
   ```

---

## 6. Transports: WebSocket vs. WebRTC

| Feature | WebSocket (`/v2/voice`) | WebRTC (`/v2/webrtc/offer`) |
| :--- | :--- | :--- |
| **Status** | **Default / Recommended** | Supported via `aiortc` |
| **Port / Protocol** | TCP port 8000 (HTTP/WSS) | HTTP signaling + UDP media RTP |
| **Cloud Proxy Compatibility** | 100% works through all HTTPS proxies | Requires UDP or TURN over TCP/TLS |
| **Automatic Fallback** | N/A | **Yes** (falls back to WS on ICE failure) |

### WebRTC behind Cloud Pod Proxies (TURN Setup)
Krutrim Cloud pods route incoming traffic through an HTTPS reverse proxy on port 443 $\to$ 8000. Inbound UDP packets for WebRTC media ports are blocked by default. 

To enable WebRTC over restrictive firewalls, deploy an open-source **coturn** server with TURN over TLS:
```bash
# Install coturn
sudo apt-get install coturn

# /etc/turnserver.conf snippet:
listening-port=3478
tls-listening-port=443
realm=voice.example.com
user=agent:secretpassword
cert=/etc/letsencrypt/live/voice.example.com/fullchain.pem
pkey=/etc/letsencrypt/live/voice.example.com/privkey.pem
```
When configured, the browser traverses firewall restrictions over TCP port 443.

---

---

## 7. Voice Cloning & Streaming API Specifications

The platform provides a pure Speech-to-Speech (S2S) voice enrollment and real-time streaming pipeline:

### 7.1 Endpoints

| Endpoint | Method / Protocol | Description |
| :--- | :--- | :--- |
| `/voice/enroll` | `POST` (multipart/form-data) | Ingests 3–30s of recorded speech, runs silence trimming, -22 LUFS normalization, speaker verification QA, and caches conditioning embeddings (`.pt`) + audio (`.wav`). |
| `/voice/{voice_id}` | `GET` | Retrieves enrolled voice metadata, QA verification score, and duration. |
| `/voice/{voice_id}` | `DELETE` | Removes enrolled voice artifacts from disk and runtime caches. |
| `/voice/{voice_id}/preview` | `GET` | Streams WAV audio preview of the enrolled reference. |
| `/voice` | `GET` | Lists all enrolled custom voices and presets. |
| `/agent/stream` | `WebSocket` (`ws://` / `wss://`) | Direct bidirectional S2S audio stream. Accepts `voice_id`, `text_prompt`, `sample_rate` (16000 or 24000). Overlapped frame-by-frame generation, instant barge-in interrupt (`{"type": "interrupt"}`), and per-turn TTFA latency instrumentation. |

### 7.2 Environment Variables

```env
# Gateway & Worker Ports
PORT=8000
WORKER_URL=ws://127.0.0.1:8998/api/chat
ENABLE_PREWARM=true

# Voice Conditioning & S2S Latency
DEFAULT_VOICE=kkishorekumar.wav
WORKER_ALLOW_RAW_PCM=0 # Set to 1 only in mock test environments lacking libopus/sphn

# Audio Transport
AUDIO_SAMPLE_RATE=24000
AUDIO_FRAME_MS=80
```

---

## 8. Troubleshooting Guide

### 1. Browser Voice Enrollment Fails
- **Cause:** Missing PyAV / soundfile WebM container decoder.
- **Fix:** The frontend Web Audio recorder directly generates uncompressed 24 kHz mono 16-bit PCM WAV blobs. The backend additionally features an automatic FFmpeg CLI pipe transcoding fallback.

### 2. Audio Dropped Mid-Sentence
- **Cause:** Premature turn yielding upon encountering text token punctuation (`?` or `.`) while audio generation lagged behind text.
- **Fix:** Fixed in `orchestration/api/voice_v2.py` and `orchestration/api/voice_stream.py`: opcode `0x01` audio frames are never discarded before audio delivery finishes.

### 3. High TTFA (25-30 seconds)
- **Cause:** Re-encoding un-cached raw `.wav` voice prompts on every connection.
- **Fix:** Voice conditioning embeds into cached `.pt` PyTorch tensors at enrollment time, loading instantaneously (< 50ms) into the transformer KV state.

---

## 9. Verification & Test Suite

Run the full automated test suite:
```bash
# Run unit and integration tests (37 passing)
pytest tests/test_voice_clone.py tests/test_voice_custom_upload.py tests/test_voice_stream_e2e.py tests/test_gateway.py tests/test_phase1_s2s.py tests/test_audio.py tests/test_turn_detector.py -v

# Run direct worker wire test
python scripts/worker_direct_test.py --url ws://127.0.0.1:8998/api/chat

# Run end-to-end smoke call
python scripts/smoke_call.py --url ws://127.0.0.1:8000/v2/voice
```

