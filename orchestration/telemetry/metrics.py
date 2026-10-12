"""
Operational Metrics & Observability Engine for PersonaPlex Voice Platform.

Tracks and calculates:
- Latency percentiles (p50, p90, p95, p99) for per-frame processing and TTFA
- Frame drops and throughput
- Active session concurrency
- Worker pool saturation and backpressure queue depth
- Call disconnect reason distribution
- Hardware telemetry (GPU VRAM, temperature, utilization)
"""

from __future__ import annotations

import math
import threading
import time
from collections import Counter
from typing import Any

from .gpu import get_live_gpu_telemetry


class MetricsCollector:
    """Thread-safe, sliding-window operational metrics collector."""

    def __init__(self, max_samples: int = 2000) -> None:
        self.max_samples = max_samples
        self._lock = threading.Lock()

        # Latency samples
        self._frame_step_times_ms: list[float] = []
        self._ttfa_ms: list[float] = []

        # Counters
        self._total_sessions_started: int = 0
        self._total_sessions_completed: int = 0
        self._total_frames_in: int = 0
        self._total_frames_out: int = 0
        self._total_frame_drops: int = 0
        self._total_barge_ins: int = 0
        self._total_retries: int = 0
        self._disconnect_reasons: Counter[str] = Counter()

        # Gauge trackers
        self._active_sessions: int = 0

    def record_session_start(self) -> None:
        with self._lock:
            self._total_sessions_started += 1
            self._active_sessions += 1

    def record_session_end(self, reason: str = "completed") -> None:
        with self._lock:
            self._total_sessions_completed += 1
            self._active_sessions = max(0, self._active_sessions - 1)
            self._disconnect_reasons[reason] += 1

    def record_frame_step_latency(self, latency_ms: float) -> None:
        if latency_ms <= 0:
            return
        with self._lock:
            self._frame_step_times_ms.append(latency_ms)
            if len(self._frame_step_times_ms) > self.max_samples:
                self._frame_step_times_ms.pop(0)

    def record_ttfa(self, ttfa_ms: float) -> None:
        if ttfa_ms <= 0:
            return
        with self._lock:
            self._ttfa_ms.append(ttfa_ms)
            if len(self._ttfa_ms) > self.max_samples:
                self._ttfa_ms.pop(0)

    def record_frames(self, in_delta: int = 0, out_delta: int = 0, drop_delta: int = 0) -> None:
        with self._lock:
            self._total_frames_in += in_delta
            self._total_frames_out += out_delta
            self._total_frame_drops += drop_delta

    def record_barge_in(self) -> None:
        with self._lock:
            self._total_barge_ins += 1

    def record_retry(self) -> None:
        with self._lock:
            self._total_retries += 1

    @staticmethod
    def _percentile(data: list[float], p: float) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_data[int(k)]
        d0 = sorted_data[int(f)] * (c - k)
        d1 = sorted_data[int(c)] * (k - f)
        return d0 + d1

    def get_summary(self, pool_stats: dict[str, Any] | None = None) -> dict[str, Any]:
        """Generate structured metrics payload for monitoring endpoints."""
        with self._lock:
            frame_latencies = list(self._frame_step_times_ms)
            ttfas = list(self._ttfa_ms)
            active_cnt = self._active_sessions
            total_started = self._total_sessions_started
            total_completed = self._total_sessions_completed
            frames_in = self._total_frames_in
            frames_out = self._total_frames_out
            frame_drops = self._total_frame_drops
            barge_ins = self._total_barge_ins
            retries = self._total_retries
            reasons = dict(self._disconnect_reasons)

        gpu_info = get_live_gpu_telemetry()

        p50_latency = round(self._percentile(frame_latencies, 50), 2)
        p95_latency = round(self._percentile(frame_latencies, 95), 2)
        p99_latency = round(self._percentile(frame_latencies, 99), 2)

        p50_ttfa = round(self._percentile(ttfas, 50), 2)
        p95_ttfa = round(self._percentile(ttfas, 95), 2)

        pool = pool_stats or {}

        return {
            "timestamp": time.time(),
            "active_sessions": active_cnt,
            "sessions_total_started": total_started,
            "sessions_total_completed": total_completed,
            "workers_total": pool.get("total_workers", 0),
            "workers_idle": pool.get("idle_workers", 0),
            "workers_busy": pool.get("busy_workers", 0),
            "workers_unhealthy": pool.get("unhealthy_workers", 0),
            "total_user_frames_in": frames_in,
            "total_agent_frames_out": frames_out,
            "total_barge_in_events": barge_ins,
            "frames": {
                "inbound_total": frames_in,
                "outbound_total": frames_out,
                "dropped_total": frame_drops,
                "barge_in_events": barge_ins,
            },
            "latency_ms": {
                "p50_frame_step": p50_latency,
                "p95_frame_step": p95_latency,
                "p99_frame_step": p99_latency,
                "p50_ttfa": p50_ttfa,
                "p95_ttfa": p95_ttfa,
            },
            "retries_total": retries,
            "disconnect_reasons": reasons,
            "worker_pool": {
                "total_workers": pool.get("total_workers", 0),
                "idle_workers": pool.get("idle_workers", 0),
                "busy_workers": pool.get("busy_workers", 0),
                "unhealthy_workers": pool.get("unhealthy_workers", 0),
                "waiting_queue_depth": pool.get("waiting_queue_depth", 0),
                "max_queue_size": pool.get("max_queue_size", 5),
                "utilization_pct": pool.get("utilization_pct", 0.0),
            },
            "hardware": {
                "cuda_available": gpu_info.get("cuda_available", False),
                "device_name": gpu_info.get("device_name", "None"),
                "vram_used_mb": gpu_info.get("vram_used_mb", 0.0),
                "vram_total_mb": gpu_info.get("vram_total_mb", 0.0),
                "vram_free_mb": gpu_info.get("vram_free_mb", 0.0),
                "gpu_util_pct": gpu_info.get("gpu_util_pct", 0),
                "temperature_c": gpu_info.get("temperature_c"),
            },
        }

    def reset(self) -> None:
        """Reset all metrics (primarily for unit testing)."""
        with self._lock:
            self._frame_step_times_ms.clear()
            self._ttfa_ms.clear()
            self._total_sessions_started = 0
            self._total_sessions_completed = 0
            self._total_frames_in = 0
            self._total_frames_out = 0
            self._total_frame_drops = 0
            self._total_barge_ins = 0
            self._total_retries = 0
            self._disconnect_reasons.clear()
            self._active_sessions = 0


# Global singleton metrics collector
metrics_collector = MetricsCollector()
