# Phase 1: Deletion Manifest for S2S Rebuild

**Tag**: `pre-cleanup` (Commit `10db60927c4011d35654ae72f8062ce1ef19eb6d`)  
**Status**: Approved by User (`"approve manifest be"` on 2026-10-08) - Executed & Verified Green  
**Policy**: Pure S2S foundation only. Zero Engine B / cascaded code in call path. Zero scripted greetings or closings.

---

## 1. Candidate Deletion Manifest with Evidence

| Category | File / Path | Evidence for Deletion |
| :--- | :--- | :--- |
| **Engine B / Cascaded Pipeline** | `orchestration/engines/cascaded/engine.py` | Cascaded fallback pipeline. Zero imports from active S2S runtime; replaced by pure PersonaPlex S2S. |
| | `orchestration/engines/cascaded/fakes.py` | Fake STT/LLM/TTS mocks used solely for cascaded pipeline tests. |
| | `orchestration/engines/cascaded/pipeline_factory.py` | Factory assembling STT $\to$ LLM $\to$ TTS pipeline. Deleted per Decision 1 (S2S only). |
| | `orchestration/engines/cascaded/tools.py` | Cascaded tool-calling shim. Not part of PersonaPlex full-duplex speech loop. |
| **Provider Catalog (Engine B)** | `orchestration/providers/catalog/*.yaml` (11 files) | Provider manifests for Cartesia, Deepgram, ElevenLabs, OpenAI, Anthropic, Sarvam. Never used in pure S2S. |
| **Cascaded Workers** | `orchestration/worker/cascaded_worker.py` | Local worker bridging cascaded pipeline over WebSocket. |
| | `orchestration/worker/local_cascade.py` | Legacy cascade worker runner. |
| **Text Chunker / Normalizer** | `orchestration/chunker/__init__.py` | Text clause chunker for cascading LLM output into TTS. S2S operates directly on audio frames. |
| | `orchestration/chunker/bridge.py` | Bridge chunking text. Zero live S2S references. |
| | `orchestration/chunker/normalizer.py` | Text normalizer for TTS. Redundant in speech-to-speech. |
| **Dormant Folder** | `orchestration/dormant/` (except `voice_clone.py`) | Stale copies of personas, rag, tts, and worker modules. *Note: `voice_clone.py` (20 KB real implementation) was preserved and moved to `orchestration/tts/voice_clone.py`.* |
| **Obsolete Tests** | `tests/test_cascaded_pipeline.py` | Only tests `CascadedVoiceEngine`. |
| | `tests/test_chunker.py` | Only tests `ClauseChunker`. |
| | `tests/test_normalizer.py` | Only tests `ClauseNormalizer`. |
| | `tests/test_engine_b_voice.py` | Only tests Engine B routing. |
| | `tests/test_local_cascade_worker.py` | Only tests `CascadedLocalWorkerServer`. |
| | `tests/test_migration_engine_b.py` | Only tests Engine B DB migrations. |
| | `tests/test_migration_provider_credentials.py`| Only tests cloud provider API key storage. |
| | `tests/test_provider_api.py` | Only tests `/v2/providers` endpoints. |
| | `tests/test_provider_catalog.py` | Only tests YAML provider catalogs. |
| | `tests/test_rag.py` | Only tests legacy TF-IDF/Chroma RAG (pure BM25 moved to `orchestration/rag/engine.py`). |
| | `tests/test_agent_engine_b_editor.py` | Tests Engine B editor UI config. |

---

## 2. Explicitly Kept Files (Preserved Baseline)
- **PersonaPlex Core**: `orchestration/worker/client.py`, `orchestration/worker/pool.py`, `orchestration/worker/mock_worker.py`.
- **Session & Telemetry**: `orchestration/session/manager.py`, `orchestration/api/voice_v2.py`.
- **Audio DSP & Codecs**: `orchestration/audio/dsp.py`, `orchestration/audio/framing.py`, `orchestration/audio/codecs.py`, `orchestration/audio/recorder.py`, `orchestration/audio/detokenizer.py`.
- **Voice Cloning Core**: `orchestration/tts/voice_clone.py` (migrated from dormant).
- **Domain & Application**: `orchestration/domain/`, `orchestration/application/`, `orchestration/infrastructure/`.
- **Database & Repositories**: `orchestration/db/` (all schema, models, sessions).
- **Protected User Data**: `.env`, `.secrets/`, `data/`, `data/voice_refs/owner/`.
