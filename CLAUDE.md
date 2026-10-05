# Claude System Guide: PersonaPlex Orchestration AI

> For the exhaustive, full-depth technical manual, see [PROJECT_DETAILS.md](file:///c:/Users/kisho/Desktop/orchestration_ai/PROJECT_DETAILS.md).

## Quick Reference & Cheatsheet for Claude

### What is this Project?
A full-duplex speech-to-speech conversational AI orchestration gateway for **NVIDIA PersonaPlex 7B** (Kyutai Moshi architecture + Mimi 24kHz neural audio codec) with local cascaded fallbacks (Whisper + Ollama + Kokoro/EdgeTTS) and an Apple-inspired Developer Studio web console.

### Key Ground Truths
1. **English Only**: PersonaPlex has zero multilingual or Indic training. Non-English speech must route to the cascaded fallback (`indian_pro` / `indian_priya`).
2. **Audio Specs**: 24,000 Hz, mono, 12.5 Hz, exactly 1,920 audio samples per frame (80 ms).
3. **Concurrency**: Upstream `moshi.server` wraps every connection in `async with self.lock:`. Exactly 1 active audio stream per worker process. Concurrency is handled by running multiple workers on distinct local ports (`8998`, `8999`).
4. **VRAM Footprint**: ~18–20 GB VRAM per 7B worker process.
5. **System Prompt Delimiters**: `<system> {prompt} <system>` (both tags are `<system>`, not `</system>`). Max 350 tokens during handshake.
6. **Binary WebSocket Protocol**: `/v1/realtime` uses 1-byte opcodes:
   - `0x00`: Handshake (version + model)
   - `0x01`: Audio PCM (1,920 samples, 24kHz, 16-bit linear)
   - `0x02`: Text Token (UTF-8, special prefix `▁` becomes space)
   - `0x03`: Control Signal (`0=start`, `1=endTurn`, `2=pause`, `3=restart`, `4=bargeIn`)
   - `0x04`: Metadata (JSON)
   - `0x05`: Error message
   - `0x06`: Ping/Heartbeat

### How to Run Locally (Zero GPU / Windows)
Use `.venv-gpu` which contains all required dependencies:
```powershell
.\.venv-gpu\Scripts\python.exe -m uvicorn orchestration.gateway.app:create_app --factory --host 127.0.0.1 --port 8000
```
- Open console in browser: `http://localhost:8000/console` (or `http://localhost:8000/`)
- Exported standalone frontend: [frontend.html](file:///c:/Users/kisho/Desktop/orchestration_ai/frontend.html)
- Automatically boots the local mock worker on `ws://127.0.0.1:8998` if no GPU worker is reachable.

### Project Layout
- `orchestration/gateway/`: FastAPI app, security, and `studio_ui.py` (Developer Console HTML).
- `orchestration/worker/`: `pool.py` (worker distribution), `mock_worker.py` (zero-GPU dev), `local_cascade.py` (multilingual fallback).
- `orchestration/audio/`: `cleaner.py` (80Hz HPF + RNNoise + Silero VAD voice isolation), `turn_detector.py` (<200ms barge-in).
- `orchestration/persona/`: Persona registry (Alex, Elena, Marcus, Sam, Aarav, Priya).
- `orchestration/tts/`: Voice cloning pipeline with cosine similarity and consent check.
- `_personaplex_upstream/`: Upstream NVIDIA/Kyutai repository and React client.
