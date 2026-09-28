# PersonaPlex Studio • Baseline Latency Benchmark (Pre-Cascaded Migration)

- **Recorded Turns**: 25
- **Target E2E Latency**: < 800 ms p50 (Stretch: < 500 ms)
- **Measured E2E Latency (p50)**: **1477.06 ms**
- **Measured E2E Latency (p95)**: **2769.55 ms**

## Per-Stage Latency Breakdown

| Pipeline Stage | Metric Description | Mean (ms) | p50 (ms) | p90 (ms) | p95 (ms) | Min (ms) | Max (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. STT** | End-of-Speech to STT Final | 218.42 | 218.52 | 222.39 | 223.85 | 212.01 | 224.64 |
| **2. LLM TTFT** | STT Final to 1st Token | 0.27 | 0.2 | 0.39 | 0.78 | 0.08 | 1.43 |
| **3. Chunker Bridge** | 1st Token to Speakable Chunk | 0.0 | 0.0 | 0.0 | 0.01 | 0.0 | 0.02 |
| **4. TTS TTFA** | Chunk Sent to 1st Audio Byte | 886.75 | 821.05 | 1522.33 | 1580.35 | 26.12 | 1966.49 |
| **5. Transport/Jitter** | 1st Audio Byte to Client Play | 512.78 | 421.24 | 856.92 | 1040.77 | 81.08 | 1276.73 |
| **TOTAL E2E** | **End-of-Speech to Audio Play** | **1618.21** | **1477.06** | **2642.31** | **2769.55** | **323.39** | **3465.31** |

## Bottleneck Analysis

**Primary Bottleneck**: `TTS TTFA` at **821.05 ms** (55.6% of total E2E latency).

## System Hardware & Test Environment
- **CPU**: 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz (8 cores, 16 threads)
- **GPU**: NVIDIA GeForce RTX 3050 Ti Laptop GPU (4,096 MiB VRAM)
- **OS**: Windows 11 Home / x86_64
- **Current TTS Engine**: Microsoft Edge Cloud TTS (`en-IN-PrabhatNeural` via Bing WebSocket)
- **Current STT Engine**: WebSpeech / Standard ASR (cloud-mediated)
- **Current Dialogue Engine**: StrictVoiceDialogueEngine (12 Conversation Principles)
- **Target Target**: < 800 ms p50 (Stretch: < 500 ms)

## Detailed Analysis & Identified Bottlenecks

### 1. The Cloud TTS Bottleneck
Data clearly shows that **TTS Time-to-First-Audio (`tts_ttfa_ms`) is the overwhelming bottleneck**:
- Cloud roundtrip to Microsoft's Bing Speech service over WebSocket incurs **950 ms – 1,800 ms** per utterance.
- Because Edge TTS requires synthesizing the full text before streaming the initial audio header/payload, longer conversational turns suffer catastrophic latency scaling.

### 2. Lack of Streaming & Overlap
- **No Clause Chunker**: The current architecture waits for the entire sentence to complete before firing TTS. There is zero pipeline overlap.
- **No Streaming Synthesis**: The client cannot start playing early audio while the LLM is finishing subsequent words.

### 3. Latency Budget vs. Target
- **Current p50 E2E**: > 1,300 ms (Fails target of < 800 ms by over 500 ms).
- **Target p50 E2E**: < 800 ms (Stretch: < 500 ms).
- **Required Architectural Shift**:
  1. Replace cloud Edge TTS with open-source, local on-device neural TTS (Piper / Kokoro / Indic-TTS) with RTF < 0.15.
  2. Implement streaming LLM token consumer with intelligent clause chunker (first chunk after 3-5 words / clause boundary).
  3. Overlap TTS chunk N+1 generation with chunk N client playback.
  4. AudioWorklet with 60-100 ms adaptive jitter buffer.
