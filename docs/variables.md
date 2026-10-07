# Configuration Variables Reference: PersonaPlex Voice Orchestrator

**Module:** `orchestration/settings.py` (backed by `pydantic-settings` `BaseSettings`)  
**Configuration Sources:** Environment variables, `.env` file, and `agent.yaml`

---

## 1. System & Protocol Invariants

| Variable Name | Defined In | Type / Unit | Default Value | PersonaPlex Counterpart / Note |
|---|---|---|---|---|
| `SAMPLE_RATE` | `Settings` | `int` (Hz) | `24000` | Native Mimi neural codec sample rate (24 kHz mono). Non-configurable. |
| `FRAME_SAMPLES` | `Settings` | `int` (samples) | `1920` | Mimi frame size (80 ms at 24 kHz). Non-configurable. |
| `FRAME_RATE_HZ` | `Settings` | `float` (Hz) | `12.5` | S2S stepping frequency (1 / 0.080s). Non-configurable. |
| `FRAME_MS` | `Settings` | `float` (ms) | `80.0` | S2S frame duration in milliseconds. Non-configurable. |
| `CLIENT_SAMPLE_RATE` | `Settings` | `int` (Hz) | `16000` | Default sample rate for web clients (can be 16000 or 8000). |
| `WIRE_ENCODING` | `Settings` | `string` | `"ogg_opus"` | Ogg containerized Opus frames over WebSocket (`0x01`). |

---

## 2. Network & Server Endpoints

| Variable Name | Defined In | Type / Unit | Default Value | PersonaPlex Counterpart / Note |
|---|---|---|---|---|
| `WORKER_HOST` | `Settings` | `string` | `"127.0.0.1"` | Upstream `moshi.server` host. |
| `WORKER_PORT` | `Settings` | `int` (port) | `8998` | Upstream `moshi.server` port. |
| `GATEWAY_HOST` | `Settings` | `string` | `"0.0.0.0"` | FastAPI gateway bind address. |
| `GATEWAY_PORT` | `Settings` | `int` (port) | `8000` | FastAPI gateway bind port (exposed to Cloud HTTPS proxy). |
| `AUTH_TOKEN` | `Settings` | `string | None` | `None` | Optional Bearer authentication token. |
| `CONNECT_TIMEOUT_SEC`| `Settings` | `float` (s) | `15.0` | Maximum wait for TCP connection to worker. |
| `HANDSHAKE_TIMEOUT_SEC`| `Settings` | `float` (s) | `60.0` | Maximum wait for model priming and handshake (`0x00`). |

---

## 3. Agent & Persona Parameters

| Variable Name | Defined In | Type / Unit | Default Value | PersonaPlex Counterpart / Note |
|---|---|---|---|---|
| `name` | `AgentConfig` | `string` | `"Alex"` | Agent display name. |
| `timezone` | `AgentConfig` | `string` | `"Asia/Kolkata"` | Timezone for dynamic local time prompt injection. |
| `voice_prompt` | `AgentConfig` | `string` | `"NATF2.pt"` | Voice embedding filename in `--voice-prompt-dir`. Maps to `/api/chat?voice_prompt=...`. |
| `greeting_mode` | `AgentConfig` | `string` | `"agent_first"` | `"agent_first"` (agent speaks upon connect) or `"user_first"` (agent listens). |
| `greeting_text` | `AgentConfig` | `string` | `""` | Optional opening greeting scenario statement. |
| `ending_text` | `AgentConfig` | `string` | `""` | Optional closing statement for natural call termination. |
| `system_prompt` | `AgentConfig` | `string` | `""` | Persona definition wrapped in `<system> ... <system>`. |
| `audio_temperature`| `AgentConfig` | `float` (0.0-2.0)| `0.8` | Maps to `moshi.server` query param `audio_temperature`. |
| `text_temperature` | `AgentConfig` | `float` (0.0-2.0)| `0.7` | Maps to `moshi.server` query param `text_temperature`. |
| `audio_topk` | `AgentConfig` | `int` | `250` | Maps to `moshi.server` query param `audio_topk`. |
| `text_topk` | `AgentConfig` | `int` | `25` | Maps to `moshi.server` query param `text_topk`. |
| `seed` | `AgentConfig` | `int` | `-1` | Maps to `moshi.server` query param `seed` (`-1` = randomized). |
| `max_duration_sec` | `AgentConfig` | `int` (s) | `1800` (30 min) | Maximum session duration before automatic closure. |
| `end_silence_sec` | `AgentConfig` | `int` (s) | `1800` (30 min) | Silence timeout before automatic closure. |
