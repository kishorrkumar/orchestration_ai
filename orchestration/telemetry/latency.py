"""
High-Resolution Per-Turn Latency Instrumentation & Telemetry.

Measures all 6 critical pipeline timestamps:
- t0_eos: End-of-user-speech (VAD or utterance boundary)
- t1_stt: STT transcription finalized
- t2_llm_first: First LLM token generated (TTFT)
- t3_tts_chunk_sent: First speakable text chunk sent to TTS
- t4_tts_audio_first: First synthesized audio byte/frame generated (TTFA)
- t5_client_play: First audio buffer played in client

Target: End-of-speech to first audio played < 800 ms p50 (stretch: 500 ms).
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any


@dataclass
class TurnTimestamps:
    turn_id: str
    text_input: str = ""
    text_output: str = ""
    t0_eos: float = 0.0
    t1_stt: float = 0.0
    t2_llm_first: float = 0.0
    t3_tts_chunk_sent: float = 0.0
    t4_tts_audio_first: float = 0.0
    t5_client_play: float = 0.0

    @property
    def stt_ms(self) -> float:
        if self.t1_stt > 0 and self.t0_eos > 0:
            return max(0.0, (self.t1_stt - self.t0_eos) * 1000.0)
        return 0.0

    @property
    def llm_ttft_ms(self) -> float:
        if self.t2_llm_first > 0 and self.t1_stt > 0:
            return max(0.0, (self.t2_llm_first - self.t1_stt) * 1000.0)
        return 0.0

    @property
    def chunker_ms(self) -> float:
        if self.t3_tts_chunk_sent > 0 and self.t2_llm_first > 0:
            return max(0.0, (self.t3_tts_chunk_sent - self.t2_llm_first) * 1000.0)
        return 0.0

    @property
    def tts_ttfa_ms(self) -> float:
        if self.t4_tts_audio_first > 0 and self.t3_tts_chunk_sent > 0:
            return max(0.0, (self.t4_tts_audio_first - self.t3_tts_chunk_sent) * 1000.0)
        return 0.0

    @property
    def transport_jitter_ms(self) -> float:
        if self.t5_client_play > 0 and self.t4_tts_audio_first > 0:
            return max(0.0, (self.t5_client_play - self.t4_tts_audio_first) * 1000.0)
        return 0.0

    @property
    def e2e_ms(self) -> float:
        if self.t5_client_play > 0 and self.t0_eos > 0:
            return max(0.0, (self.t5_client_play - self.t0_eos) * 1000.0)
        elif self.t4_tts_audio_first > 0 and self.t0_eos > 0:
            return max(0.0, (self.t4_tts_audio_first - self.t0_eos) * 1000.0)
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "text_input": self.text_input,
            "text_output": self.text_output,
            "stt_ms": round(self.stt_ms, 2),
            "llm_ttft_ms": round(self.llm_ttft_ms, 2),
            "chunker_ms": round(self.chunker_ms, 2),
            "tts_ttfa_ms": round(self.tts_ttfa_ms, 2),
            "transport_jitter_ms": round(self.transport_jitter_ms, 2),
            "e2e_ms": round(self.e2e_ms, 2),
        }


class LatencyTracker:
    """Session-wide latency tracking, percentiles, and telemetry exporter."""

    def __init__(self, session_id: str = "default") -> None:
        self.session_id = session_id
        self.turns: List[TurnTimestamps] = []
        self._current_turn: Optional[TurnTimestamps] = None

    def start_turn(self, turn_id: str, text_input: str = "", t0: Optional[float] = None) -> TurnTimestamps:
        now = t0 or time.perf_counter()
        turn = TurnTimestamps(turn_id=turn_id, text_input=text_input, t0_eos=now)
        self._current_turn = turn
        return turn

    def mark_stt_final(self, t1: Optional[float] = None) -> None:
        if self._current_turn:
            self._current_turn.t1_stt = t1 or time.perf_counter()

    def mark_llm_first_token(self, t2: Optional[float] = None) -> None:
        if self._current_turn:
            self._current_turn.t2_llm_first = t2 or time.perf_counter()

    def mark_tts_chunk_sent(self, t3: Optional[float] = None) -> None:
        if self._current_turn:
            self._current_turn.t3_tts_chunk_sent = t3 or time.perf_counter()

    def mark_tts_audio_first_byte(self, t4: Optional[float] = None) -> None:
        if self._current_turn:
            self._current_turn.t4_tts_audio_first = t4 or time.perf_counter()

    def mark_client_audio_played(self, t5: Optional[float] = None, text_output: str = "") -> TurnTimestamps:
        if self._current_turn:
            self._current_turn.t5_client_play = t5 or time.perf_counter()
            if text_output:
                self._current_turn.text_output = text_output
            turn = self._current_turn
            self.turns.append(turn)
            self._current_turn = None
            return turn
        dummy = TurnTimestamps(turn_id="unknown")
        return dummy

    @staticmethod
    def _percentile(values: List[float], p: float) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        idx = (len(s) - 1) * (p / 100.0)
        lower = math.floor(idx)
        upper = math.ceil(idx)
        if lower == upper:
            return s[int(idx)]
        weight = idx - lower
        return s[lower] * (1.0 - weight) + s[upper] * weight

    def get_summary(self) -> Dict[str, Any]:
        if not self.turns:
            return {
                "turn_count": 0,
                "stt_ms": {"p50": 0.0, "p90": 0.0, "p95": 0.0, "mean": 0.0},
                "llm_ttft_ms": {"p50": 0.0, "p90": 0.0, "p95": 0.0, "mean": 0.0},
                "chunker_ms": {"p50": 0.0, "p90": 0.0, "p95": 0.0, "mean": 0.0},
                "tts_ttfa_ms": {"p50": 0.0, "p90": 0.0, "p95": 0.0, "mean": 0.0},
                "transport_jitter_ms": {"p50": 0.0, "p90": 0.0, "p95": 0.0, "mean": 0.0},
                "e2e_ms": {"p50": 0.0, "p90": 0.0, "p95": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0},
            }

        def stats_for(metric: str) -> Dict[str, float]:
            vals = [getattr(t, metric) for t in self.turns]
            return {
                "p50": round(self._percentile(vals, 50), 2),
                "p90": round(self._percentile(vals, 90), 2),
                "p95": round(self._percentile(vals, 95), 2),
                "mean": round(sum(vals) / len(vals), 2),
                "min": round(min(vals), 2),
                "max": round(max(vals), 2),
            }

        return {
            "turn_count": len(self.turns),
            "stt_ms": stats_for("stt_ms"),
            "llm_ttft_ms": stats_for("llm_ttft_ms"),
            "chunker_ms": stats_for("chunker_ms"),
            "tts_ttfa_ms": stats_for("tts_ttfa_ms"),
            "transport_jitter_ms": stats_for("transport_jitter_ms"),
            "e2e_ms": stats_for("e2e_ms"),
        }

    def generate_markdown_report(self, title: str = "Latency Baseline Report") -> str:
        s = self.get_summary()
        tc = s["turn_count"]
        e2e = s["e2e_ms"]
        stt = s["stt_ms"]
        llm = s["llm_ttft_ms"]
        chunker = s["chunker_ms"]
        tts = s["tts_ttfa_ms"]
        transport = s["transport_jitter_ms"]

        md = [
            f"# {title}",
            "",
            f"- **Recorded Turns**: {tc}",
            f"- **Target E2E Latency**: < 800 ms p50 (Stretch: < 500 ms)",
            f"- **Measured E2E Latency (p50)**: **{e2e['p50']} ms**",
            f"- **Measured E2E Latency (p95)**: **{e2e['p95']} ms**",
            "",
            "## Per-Stage Latency Breakdown",
            "",
            "| Pipeline Stage | Metric Description | Mean (ms) | p50 (ms) | p90 (ms) | p95 (ms) | Min (ms) | Max (ms) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
            f"| **1. STT** | End-of-Speech to STT Final | {stt['mean']} | {stt['p50']} | {stt['p90']} | {stt['p95']} | {stt['min']} | {stt['max']} |",
            f"| **2. LLM TTFT** | STT Final to 1st Token | {llm['mean']} | {llm['p50']} | {llm['p90']} | {llm['p95']} | {llm['min']} | {llm['max']} |",
            f"| **3. Chunker Bridge** | 1st Token to Speakable Chunk | {chunker['mean']} | {chunker['p50']} | {chunker['p90']} | {chunker['p95']} | {chunker['min']} | {chunker['max']} |",
            f"| **4. TTS TTFA** | Chunk Sent to 1st Audio Byte | {tts['mean']} | {tts['p50']} | {tts['p90']} | {tts['p95']} | {tts['min']} | {tts['max']} |",
            f"| **5. Transport/Jitter** | 1st Audio Byte to Client Play | {transport['mean']} | {transport['p50']} | {transport['p90']} | {transport['p95']} | {transport['min']} | {transport['max']} |",
            f"| **TOTAL E2E** | **End-of-Speech to Audio Play** | **{e2e['mean']}** | **{e2e['p50']}** | **{e2e['p90']}** | **{e2e['p95']}** | **{e2e['min']}** | **{e2e['max']}** |",
            "",
            "## Bottleneck Analysis",
            "",
        ]

        # Identify biggest bottleneck
        stages = [
            ("STT", stt["p50"]),
            ("LLM TTFT", llm["p50"]),
            ("Chunker Bridge", chunker["p50"]),
            ("TTS TTFA", tts["p50"]),
            ("Transport & Jitter Buffer", transport["p50"]),
        ]
        stages.sort(key=lambda x: x[1], reverse=True)
        top_stage, top_val = stages[0]
        pct = round((top_val / max(1.0, e2e["p50"])) * 100, 1)

        md.append(f"**Primary Bottleneck**: `{top_stage}` at **{top_val} ms** ({pct}% of total E2E latency).")
        md.append("")
        return "\n".join(md)
