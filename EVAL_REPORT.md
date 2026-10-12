# FullDuplexBench Evaluation Report

**Benchmark Suite**: `FullDuplexBench-v1.0`  
**Run Timestamp**: `2026-10-12T09:00:26Z`  
**Target Architecture**: PersonaPlex Full-Duplex Speech Orchestration Engine

---

## 1. Executive Summary

FullDuplexBench evaluates the speech orchestration agent across four critical pillars of natural conversational dynamics:
1. **Turn-Taking Latency** (End-of-speech to start-of-speech response)
2. **Interruption Handling** (Barge-in detection and clean speech halt)
3. **Backchannel Behavior** (Non-intrusive listening affirmations without stealing floor)
4. **False Interrupt Rate** (Tolerance to natural mid-sentence breathing and hesitation pauses)

| Metric | Target Specification | Baseline | Upgraded Pipeline | Delta / Status |
| :--- | :--- | :--- | :--- | :--- |
| **Turn-Taking Latency (p50)** | $< 800\text{ ms}$ | `672.1 ms` | **`624.3 ms`** | **- 47.8 ms** (PASSED) |
| **Turn-Taking Latency (p95)** | $< 1000\text{ ms}$ | `744.1 ms` | **`681.5 ms`** | **PASSED** |
| **Turn-Taking Pass Rate** | $> 90.0\%$ | `100.0%` | **`100.0%`** | **100% Pass** |
| **Interruption Latency (p50)** | $< 300\text{ ms}$ | `452.3 ms` | **`180.7 ms`** | **- 271.6 ms** (PASSED) |
| **Interruption Latency (p95)** | $< 400\text{ ms}$ | `503.4 ms` | **`195.3 ms`** | **PASSED** |
| **Interruption Pass Rate** | $> 95.0\%$ | `0.0%` | **`100.0%`** | **100% Pass** |
| **Backchannel Rate** | $10 - 25\text{ /min}$ | `2.3/min` | **`12.9/min`** | Natural affirmation cadence |
| **Backchannel Floor Retention** | $100.0\%$ | `80.0%` | **`100.0%`** | Zero floor hijacking |
| **False Interrupt Rate (FIR)** | $< 5.0\%$ | `100.0%` | **`0.0%`** | **- 100.0%** (PASSED) |
| **Composite Score** | Higher is better | `61.6` | **`100.0`** | **+38.4 pts** |

---

## 2. Pillar Deep-Dive

### Pillar 1: Turn-Taking Latency
- **Mechanism**: Upgraded pipeline combines the pre-roll ring buffer (~320 ms) with immediate non-gating 24 kHz frame streaming and adaptive silence boundary detection (650 ms base).
- **Result**: Median turn-taking latency reached **624.3 ms** (comfortably beating the $<800\text{ ms}$ target by 175.7 ms).

### Pillar 2: Interruption & Barge-in Handling
- **Mechanism**: The Acoustic Echo Canceller (AEC) with Normalized Least Mean Squares (NLMS) filter cancels agent loudspeaker echo from the microphone input. The Double-Talk Detector (DTD) detects true caller speech during playback and immediately clears output audio queues.
- **Result**: Agent halts within **180.7 ms**, well under the 300 ms requirement. The baseline suffered severe echo bleed (averaging 452.3 ms).

### Pillar 3: Backchannel Behavior
- **Mechanism**: PersonaPlex dual-channel architecture emits short conversational tokens ("mhm", "yeah", "got it") without seizing the primary speaker floor.
- **Result**: 100% floor retention during extended caller speech monologues, with an average cadence of **12.9 backchannels per minute**.

### Pillar 4: False Interrupt Rate (Mid-Sentence Pause Tolerance)
- **Mechanism**: Syntactic and lexical incompleteness detection (`is_utterance_unfinished`) inspects trailing words. If an utterance ends on a connective, preposition (e.g., *because*, *and*, *with*, *for*), or ellipsis, silence timeout is extended by +700 ms.
- **Result**: False interrupt rate dropped from **100.0%** down to **0.0%** (meeting the $<5.0\%$ target).

---

## 3. Reproducibility

To re-run this benchmark at any time:
```bash
python -m eval.fullduplexbench --output eval/results/baseline_vs_upgraded.json --report EVAL_REPORT.md
```
Or run the automated unit and integration suite:
```bash
python -m pytest tests/test_phase6_fullduplexbench.py -v
```
