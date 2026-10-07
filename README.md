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

## 7. Troubleshooting Guide

### 1. Silent Audio (Transcript appears, but no sound)
- **Cause:** Upstream worker starved of audio packets.
- **Verification:** Check gateway logs for `queue_to_worker_pacer`. The pacer must send 1,920-sample silence frames at 12.5 Hz when the user is silent.
- **Browser Check:** Verify browser AudioContext state is `running`. In Chrome, AudioContext requires a user gesture (clicking "Start Call").

### 2. Echo / Agent Talks to Itself
- **Cause:** Microphone picking up speaker output (acoustic loop).
- **Fix 1 (Mandatory):** **Use headphones** while testing.
- **Fix 2:** The gateway UI routes the microphone input through an `AudioWorklet` with a zero-gain destination node (`muteGain.gain.value = 0.0`) so mic audio is never looped back into your speakers locally.

### 3. Slow Priming (~9.4 seconds)
- **Cause:** PersonaPlex processes the voice prompt and system prompt sequentially through its autoregressive transformer before the first audio frame can be produced ($\approx 362$ steps $\times$ 26 ms = 9.4s on A100).
- **Fix:** Enable **Pre-warming** in `.env` (`ENABLE_PREWARM=true`). The gateway pre-primes a standby worker when the browser console loads, reducing click-to-speech time to $< 300$ ms.

---

## 8. Verification & Test Suite

Run the full automated test suite:
```bash
# Run unit and integration tests
pytest tests/ -v

# Run direct worker wire test
python scripts/worker_direct_test.py --url ws://127.0.0.1:8998/api/chat

# Run end-to-end smoke call
python scripts/smoke_call.py --url ws://127.0.0.1:8000/v2/voice
```
