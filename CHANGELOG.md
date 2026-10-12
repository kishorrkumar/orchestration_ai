# Changelog

All notable changes to the PersonaPlex Orchestration AI platform are documented in this file following the Keep a Changelog standard and Semantic Versioning.

---

## [Unreleased] - Phase 3: Production Voice Cloning & Safety Enforcement

### Added
- **`tools/make_voice_prompt.py` CLI & Compilation Pipeline**:
  - Industrial command-line tool compiling raw audio (WAV, MP3, FLAC, M4A, WebM) into 24 kHz mono PersonaPlex voice prompt artifacts.
  - Generates both 16-bit 24,000 Hz normalized WAV (`.wav`) and PyTorch `.pt` voice prompt tensor containers (`.pt`) for `<100ms` instant worker priming.
  - Generates comprehensive `metadata.json` with acoustic metrics, SHA-256 fingerprint, gender detection, and authorization logging.
- **Pre-Flight Acoustic Quality Gates**:
  - SNR gate: minimum SNR $\ge 15.0\text{ dB}$ (calculated via 90th percentile speech vs 10th percentile ambient noise floor).
  - Digital clipping gate: rejects samples with $> 1.0\%$ clipped samples ($|x| \ge 0.999$).
  - Duration bounds: enforces $5.0\text{s}$ to $30.0\text{s}$ valid conditioning length.
  - Quiet/silent audio rejection (minimum RMS $\ge 0.008$).
  - Speech activity presence threshold ($\ge 35\%$).
- **Acoustic Preprocessing & Loudness Normalization**:
  - 80 Hz 1st-order IIR high-pass filter attenuating low-frequency rumble by $> 6\text{ dB}$.
  - Resampling to 24 kHz mono float32.
  - Silence trimming with 5.0s optimal conditioning window selection.
  - EBU R128 loudness normalization to $-24.0\text{ LUFS}$ standard with peak ceiling limiter at $0.92$.
- **Safety & Consent Enforcement**:
  - Mandatory user consent enforcement (`--consent` flag / API parameter).
  - Expanded celebrity / public figure blocklist (`PUBLIC_FIGURE_BLOCKLIST`) blocking unauthorized impersonation.
  - Cryptographic SHA-256 reference hashing for enrollment traceability.
- **Enhanced `VoiceCloner` & Test Suite**:
  - Updated `orchestration/tts/voice_clone.py` to compile and mirror `.pt` voice prompt containers directly into worker search directories.
  - Added `tests/test_phase3_voice_cloning.py` with 19 comprehensive unit & CLI benchmark tests. Total test suite reached **206 passed tests (100% pass rate)**.

### Rollback Note
- To roll back Phase 3:
  ```bash
  git revert HEAD
  ```

---

## [1.2.0] - Phase 2: Advanced Audio Front-End (Noise Cancellation & AEC)

### Added
- **Streaming Acoustic Echo Cancellation (AEC)**:
  - Created `orchestration/audio/aec.py` with `AcousticEchoCanceller`.
  - Normalized Least Mean Squares (NLMS) adaptive filtering with circular reference ring buffer lookback.
  - Normalized cross-correlation acoustic delay estimation and acoustic coupling gain scaling.
  - Double-Talk Detector (DTD): preserves caller voice completely during simultaneous speech (barge-in) and prevents filter divergence.
  - Single-talk ERLE (Echo Return Loss Enhancement) > 10 dB with zero attenuation of caller double-talk.
- **Enhanced CallerAudioCleaner**:
  - Integrated AEC into audio pipeline (`orchestration/audio/cleaner.py`).
  - Added `push_agent_reference(ref_audio)` registered on outbound agent playback frames in `orchestration/api/voice_v2.py`.
  - Optimized non-gating 24 kHz spectral fallback preventing resampling aliasing artifacts when native C RNNoise is unavailable.
  - Non-gating VAD telemetry: reports speech probability while passing low-energy pause frames without zero-gating, preserving PersonaPlex full-duplex conversational dynamics.
  - Runtime per-session configuration flags: `denoise`, `denoise_mode` (`rnnoise`, `isolation`, `bypass`), and `enable_aec`.
- **Synthetic Noise & Latency Benchmark Suite**:
  - Created `tests/test_phase2_noise_benchmark.py` covering 4 noise profiles (white, babble, street, keyboard clicks) across 4 SNR levels (20, 10, 5, 0 dB).
  - Validated SNR improvement across all 16 conditions.
  - Real-time latency benchmark: added processing overhead < 0.20 ms per 80 ms frame (well within the 15 ms budget).
  - Clean speech preservation: amplitude and spectral distortion < 1.0 dB.

### Rollback Note
- All Phase 2 features can be bypassed per session via query parameters (`?denoise=false&enable_aec=false`) or globally configured in `Settings`.
- To roll back Phase 2:
  ```bash
  git revert HEAD
  ```

---

## [1.1.0] - Phase 1: Core Robustness & Lifecycle

### Added
- **Dual Idle/Silence Timeouts**:
  - Implemented `soft_silence_timeout_sec` (default: 10.0s) and `silence_timeout_sec` (default: 25.0s) in `EndOfCallDetector`.
  - Added real-time soft silence event (`soft_silence_prompt`) over WebSocket allowing conversational nudging when caller pauses.
  - Automatically resets silence timers on user speech activity (`on_user_speech` / `on_user_speech_end`).
  - Added clean session teardown upon hard silence timeout (`idle_silence_timeout`).
- **Bounded Session Pool & Backpressure**:
  - Added `max_queue_size` (default: 5) to `WorkerPool` to prevent unbounded request queueing.
  - Fast capacity rejection (`PoolCapacityExceededError` / WebSocket code `1013 TRY_AGAIN_LATER`) when pool is saturated.
  - Added live queue metrics: `waiting_queue_depth` and `max_queue_size` in `get_stats()`.
- **Worker Connection Retry**:
  - Implemented `connect_with_retry(max_attempts=3, backoff_sec=0.5)` with exponential backoff in `PersonaPlexWorkerClient`.
- **Operational Metrics & Observability**:
  - Created `orchestration/telemetry/metrics.py` thread-safe `MetricsCollector` calculating p50/p95/p99 frame processing latency, TTFA percentiles, active session gauge, frame drop counters, and live GPU telemetry.
  - Exposed `/metrics` endpoint on FastAPI gateway compatible with Prometheus and Datadog monitoring.
- **Structured JSON Logging with Correlation IDs**:
  - Added `bind_session_context(session_id, agent_id)` and `clear_session_context()` in `orchestration/shared/logging.py`.
- **Test Suite**:
  - Added `tests/test_phase1_lifecycle.py` with 8 comprehensive unit & integration tests (168 tests total, 100% passing).

### Changed
- Updated `orchestration/shared/settings.py` and `orchestration/settings.py` with typed Pydantic fields for timeouts, retries, bounded queues, and CPU offload flags.
- Enhanced `/readyz` probe to report database health, worker pool statistics, and GPU metrics.

### Rollback Note
- To roll back Phase 1 changes:
  ```bash
  git revert HEAD
  ```
  All Phase 1 features sit behind typed config flags in `Settings` (`IDLE_SOFT_SILENCE_SEC`, `IDLE_HARD_SILENCE_SEC`, `POOL_ACQUIRE_TIMEOUT_SEC`, `WORKER_RETRY_ATTEMPTS`) and can be disabled via environment variables without code modification.

---

## [1.0.0] - Phase 0: Baseline Audit & Repository Hygiene

### Added
- `AUDIT.md`: Complete architecture analysis and P0–P3 defect categorization.
- `THIRD_PARTY.md`: Comprehensive open-source software license audit (MIT, BSD-3, Apache-2.0).
- `.dockerignore`: Exclusion rules for lean non-root Docker builds.

### Fixed
- **P0.1 Voice Prompt Format**: Fixed `.wav` vs `.pt` voice prompt routing in `orchestration/worker/client.py` with preset fallback for raw audio files.
- **P0.2 Standby Deadlock**: Resolved single-worker standby lease deadlock on voice mismatch in `orchestration/session/manager.py`.
- **P0.3 Audio Cleaner**: Implemented `CallerAudioCleaner` (80 Hz HPF + RNNoise + Silero VAD) to eliminate noise gain amplification and spurious turn-taking hallucinations.
- **Repository Bloat**: Cleaned 5.95 GB of unignored caches and virtual environments, reducing repository footprint from 6.32 GB (69,494 files) to 373 MB (1,635 files).
