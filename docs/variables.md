# PersonaPlex Orchestration Variables & Settings Directory

This document provides a single reference of all system, audio, networking, and agent configuration variables across the platform.

---

## 1. Core Model & DSP Variables (Protocol Invariants)

These constants are bit-exact with NVIDIA PersonaPlex / Kyutai Moshi upstream specifications.

| Variable | Where Defined | Type & Unit | Default Value | PersonaPlex / Moshi Counterpart | Description |
|---|---|---|---|---|---|
| `SAMPLE_RATE` | `orchestration/settings.py` | `int` (Hz) | `24000` | `moshi.models.loaders.SAMPLE_RATE` | Native sampling rate of the Mimi neural audio codec. |
| `FRAME_SAMPLES` | `orchestration/settings.py` | `int` (samples) | `1920` | `moshi.server.ServerState.frame_size` | Number of audio samples per forward inference step ($24000 \times 0.08$). |
| `FRAME_RATE_HZ` | `orchestration/settings.py` | `float` (Hz) | `12.5` | `moshi.models.loaders.FRAME_RATE` | Frame rate of the Mimi codec and LMGen model steps. |
| `FRAME_MS` | `orchestration/settings.py` | `float` (ms) | `80.0` | `1000 / FRAME_RATE` | Duration of one audio chunk in milliseconds. |
| `CLIENT_SAMPLE_RATE` | `orchestration/settings.py` | `int` (Hz) | `16000` | N/A (Gateway resampler target) | Audio sample rate exchanged with browser web clients. |
| `CLIENT_CAPTURE_RATE` | `orchestration/settings.py` | `int` (Hz) | `16000` | N/A (AudioWorklet output) | Audio capture rate from client microphone. |
| `CLIENT_CODEC` | `orchestration/settings.py` | `str` | `"pcm16"` | N/A | Wire audio format with browser (`pcm16` or `g711_ulaw`). |
| `WIRE_ENCODING` | `orchestration/settings.py` | `str` | `"ogg_opus"` | `sphn.OpusStreamWriter(24000)` | Worker binary wire encoding (Ogg-Opus container @ 24 kHz). |

---

## 2. Worker & Gateway Networking Settings

| Variable | Where Defined | Type & Unit | Default Value | PersonaPlex / Moshi Counterpart | Description |
|---|---|---|---|---|---|
| `WORKER_HOST` | `orchestration/settings.py`, `.env` | `str` (IP/Host) | `"127.0.0.1"` | `--host` in `moshi.server` | Upstream PersonaPlex inference worker host. |
| `WORKER_PORT` | `orchestration/settings.py`, `.env` | `int` (Port) | `8998` | `--port` in `moshi.server` | Upstream PersonaPlex worker listening port. |
| `GATEWAY_HOST` | `orchestration/settings.py`, `.env` | `str` (IP/Host) | `"0.0.0.0"` | N/A | FastAPI gateway bind host. |
| `GATEWAY_PORT` | `orchestration/settings.py`, `.env` | `int` (Port) | `8000` | N/A | FastAPI gateway bind port (exposed to cloud HTTPS proxy). |
| `AUTH_TOKEN` | `orchestration/settings.py`, `.env` | `str \| None` | `None` | N/A | Optional Bearer authentication token for REST and WebSocket calls. |
| `CONNECT_TIMEOUT_SEC`| `orchestration/settings.py` | `float` (seconds) | `15.0` | N/A | Maximum wait for TCP connection to worker socket. |
| `HANDSHAKE_TIMEOUT_SEC`| `orchestration/settings.py` | `float` (seconds) | `60.0` | System prompt priming loop | Maximum wait for worker priming and initial `0x00` handshake byte. |
| `JITTER_BUFFER_MS` | `orchestration/settings.py`, `agent.yaml` | `int` (ms) | `120` | N/A | Audio playback jitter buffer window in browser client to prevent underruns. |
| `ENABLE_PREWARM` | `orchestration/settings.py` | `bool` | `True` | Standby session lease | Automatically connect and prime worker on page load to eliminate TTFA latency. |
| `PREWARM_IDLE_TIMEOUT_SEC` | `orchestration/settings.py` | `float` (seconds) | `120.0` | N/A | Inactivity timeout after which an unattached pre-warmed worker lease is released. |

---

## 3. Single Agent Configuration (`agent.yaml`)

| Variable | Where Defined | Type | Default Value | PersonaPlex / Moshi Counterpart | Description |
|---|---|---|---|---|---|
| `name` | `agent.yaml` | `str` | `"Alex"` | Template variable `{{agent_name}}` | Name of the voice persona. |
| `timezone` | `agent.yaml` | `str` | `"Asia/Kolkata"` | ZoneInfo context | IANA timezone used for greeting time-of-day resolution. |
| `voice_prompt` | `agent.yaml` | `str` | `"NATF2.pt"` | `request.query["voice_prompt"]` | Voice reference file (.pt embedding or .wav in voices directory). |
| `greeting_mode` | `agent.yaml` | `str` | `"agent_first"` | Model prompt instruction | Turn sequence: `"agent_first"` (speaks first) or `"user_first"` (listens first). |
| `greeting_text` | `agent.yaml` | `str` | `"Hello! This is Alex..."` | Model prompt instruction | Opening sentence for the agent. |
| `ending_text` | `agent.yaml` | `str` | `"Thank you for calling..."` | End-of-call detector regex | Concluding line used by termination detector. |
| `system_prompt` | `agent.yaml` | `str` | *(prose)* | `request.query["text_prompt"]` | Persona role, background, scenario, and speaking style. |
| `variables` | `agent.yaml` | `dict[str, str]` | `company`, `caller_name`, ... | Template substitution values | Strict default values for all `{{variable}}` template placeholders. |
| `audio_temperature`| `agent.yaml` | `float` | `0.8` | `LMGen.temp` | Audio sampling temperature. |
| `text_temperature` | `agent.yaml` | `float` | `0.7` | `LMGen.temp_text` | Text token sampling temperature. |
| `audio_topk` | `agent.yaml` | `int` | `250` | `LMGen.top_k` | Top-K sampling for audio codebooks. |
| `text_topk` | `agent.yaml` | `int` | `25` | `LMGen.top_k_text` | Top-K sampling for text generation. |
| `seed` | `agent.yaml` | `int \| None` | `-1` | `torch.manual_seed` | Generation seed (-1 for random). |
| `max_duration_sec` | `agent.yaml` | `int` (seconds) | `600` | Session timer | Hard call duration cutoff. |
| `end_silence_sec` | `agent.yaml` | `int` (seconds) | `20` | Silence watchdog | Disconnect call if silence exceeds this duration. |
