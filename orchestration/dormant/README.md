# Dormant Fallback Subsystems

> **Status:** ISOLATED & DORMANT  
> **Purpose:** Fallback pipelines retained in repository for research, evaluation, and multilingual fallback. Not exposed in the primary Lean S2S Voice Agent product UI or default routes.

---

## Contained Subsystems

1. **`rag/` (`engine.py`):**
   - Vector chunking and semantic document retrieval engine.
   - Preserved for offline knowledge-grounded experimentation.

2. **`tts/voice_clone.py`:**
   - Zero-shot voice cloning pipeline with cosine similarity validation and speaker consent verification.
   - PersonaPlex native voice conditioning uses the 18 official `.pt` preset embeddings.

3. **`worker/local_cascade.py` & `worker/cascaded_worker.py`:**
   - Local cascaded fallback worker (Faster-Whisper ASR + Ollama/Qwen 2.5 LLM + Kokoro/EdgeTTS).
   - Designed for low-resource environments and non-English / Indian English conversation.

4. **`personas/`:**
   - Markdown prompt definitions for the cascaded fallback personas (Aarav, Priya).
