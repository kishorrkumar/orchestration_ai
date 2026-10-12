# Changelog

All notable changes to the PersonaPlex Orchestration AI platform are documented in this file following the Keep a Changelog standard and Semantic Versioning.

---

## [Unreleased] - Phase 1: Core Robustness & Lifecycle

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
