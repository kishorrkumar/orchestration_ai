# Claude System Guide: PersonaPlex Orchestration AI

> For the exhaustive, full-depth technical manual, see [PROJECT_DETAILS.md](file:///c:/Users/kisho/Desktop/orchestration_ai/PROJECT_DETAILS.md).

## Quick Reference & Cheatsheet for Claude

### What is this Project?
A production-grade **Lean S2S (Speech-to-Speech) Voice Agent Platform** for **NVIDIA PersonaPlex 7B** (Kyutai Moshi architecture + Mimi 24kHz neural audio codec) with local cascaded fallbacks, SQLite/Neon PostgreSQL persistence with immutable versioning, a bit-exact 16 kHz audio core (with G.711 μ-law telephony readiness), and a modern React 18 Voice Agent Studio.

### Key Ground Truths
1. **English Only**: PersonaPlex has zero multilingual or Indic training. Non-English speech must route to the cascaded fallback (`indian_pro` / `indian_priya`).
2. **Audio Specs**: 
   - PersonaPlex Model: 24,000 Hz, mono, 12.5 Hz, exactly 1,920 audio samples per frame (80 ms) or 480 samples per 20 ms.
   - Client Web Audio: 16,000 Hz, mono, 20 ms frames (320 samples / 640 bytes PCM16).
   - Telephony Audio: 8,000 Hz, mono, 20 ms frames (160 samples / 160 bytes G.711 μ-law).
   - Anti-Aliasing Resampling: High-quality libsoxr (`soxr.ResampleStream`) with >82 dB SNR and <0.2 ms delay.
3. **Concurrency**: Upstream `moshi.server` wraps every connection in `async with self.lock:`. Exactly 1 active audio stream per worker process. Concurrency is handled by running multiple workers on distinct local ports (`8998`, `8999`).
4. **VRAM Footprint**: ~18–20 GB VRAM per 7B worker process.
5. **System Prompt Delimiters**: `<system> {prompt} <system>` (both tags are `<system>`, not `</system>`). Max 350 tokens during handshake (ideal ≤150). Token counting uses the verified SentencePiece model `models/tokenizer_spm_32k_3.model`.
6. **Lean Agent Model (6 Fields)**:
   - `name`: Display name and `{{agent_name}}` template variable.
   - `voice_id`: One of the 18 official PersonaPlex presets (`orchestration/persona/presets.py`).
   - `greeting`: Initial spoken greeting + `agent_speaks_first` boolean.
   - `system_prompt`: Spoken persona instructions with dynamic variables (`{{time}}`, `{{date}}`, etc.).
   - `ending`: Final goodbye line + `end_call_timeout_sec` + `silence_timeout_sec`.
   - `timezone`: IANA timezone string (e.g., `UTC`, `America/New_York`, `Asia/Kolkata`).
7. **Database & Versioning**:
   - `orchestration/db/models.py`: `AgentRecord`, `AgentVersionRecord`, `CallSessionRecord`, `CallTurnRecord`.
   - Immutable versions (`v1`, `v2`, etc.). Editing an agent automatically creates a new immutable version.
   - Dual-engine: SQLite (`data/platform.db`) for local dev, Neon PostgreSQL (`postgresql+asyncpg://...`) for cloud production.
8. **WebSocket Endpoints**:
   - `/v2/voice`: Lean S2S endpoint supporting 16 kHz web PCM and 8 kHz telephony G.711 μ-law, real-time turn tracking, call session recording, and end-of-call detection.
   - `/v1/realtime`: Legacy 24 kHz raw framing endpoint.

### How to Run Locally (Zero GPU / Windows)
Use `.venv-gpu` which contains all required dependencies:
```powershell
.\.venv-gpu\Scripts\python.exe -m uvicorn orchestration.gateway.app:create_app --factory --host 127.0.0.1 --port 8000
```
- **Lean Voice Agent Studio**: `http://localhost:8000/` (or `http://localhost:8000/studio`)
- **Legacy Developer Console**: `http://localhost:8000/console/legacy`
- Automatically boots the local mock worker on `ws://127.0.0.1:8998` if no GPU worker is reachable.

### Test Suite Execution
```powershell
.\.venv-gpu\Scripts\python.exe -m pytest
```
119 passed, 1 skipped (GPU-only test `test_persona_gpu.py`), 0 failures.

### Project Layout
- `orchestration/db/`: Database models, async session manager (SQLite + Neon Postgres), `AgentService`, and `CallSessionService`.
- `orchestration/prompts/`: SentencePiece token counter, voice prompt linter, and prompt compiler.
- `orchestration/protocol/`: Canonical audio constants (`CLIENT_SAMPLE_RATE=16000`, `TELEPHONY_SAMPLE_RATE=8000`), binary framing opcodes.
- `orchestration/audio/`: 
  - `codecs.py`: Pure NumPy bit-exact ITU-T G.711 μ-law encoder/decoder.
  - `resample.py`: Anti-aliased `AudioResampler` and `StreamingResampleBuffer` using libsoxr.
  - `dsp.py`: Dynamic range soft clipping, RMS computation, format converters.
  - `cleaner.py`: 80Hz HPF + RNNoise + Silero VAD voice isolation.
  - `turn_detector.py`: Barge-in detector (<200ms cutoff).
- `orchestration/pipeline/`: `EndOfCallDetector` (fuzzy match, silence timeout, max duration).
- `orchestration/gateway/`: 
  - `app.py`: FastAPI server, REST routes, legacy and v2 WebSocket handlers.
  - `lean_studio_ui.py`: React 18 Lean Voice Agent Studio (glassmorphism UI, 18 voice previews, live token counter, live microphone visualizer).
  - `studio_ui.py`: Legacy developer console.
- `orchestration/persona/`: 18 PersonaPlex presets (`presets.py`), registry, dialogue tracking.
- `orchestration/worker/`: `pool.py`, `mock_worker.py`, `local_cascade.py`.
