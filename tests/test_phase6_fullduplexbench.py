"""
Unit and integration tests for Phase 6 FullDuplexBench offline evaluation suite.
"""

import json
import os

import pytest

from eval.fullduplexbench import (
    FullDuplexBench,
    save_reports,
)


def test_fullduplexbench_turn_taking_latency_under_target():
    bench = FullDuplexBench()
    results = bench.evaluate_turn_taking(pipeline_type="upgraded", trials=15)

    assert len(results) == 15
    pass_count = sum(1 for r in results if r.passed)
    pass_rate = pass_count / len(results)
    assert pass_rate >= 0.90, f"Expected >=90% turn taking pass rate, got {pass_rate:.1%}"

    latencies = [r.latency_ms for r in results]
    median_lat = sorted(latencies)[len(latencies) // 2]
    assert median_lat < 800.0, f"Median turn-taking latency {median_lat}ms exceeds 800ms target"


def test_fullduplexbench_interruption_latency_under_300ms():
    bench = FullDuplexBench()
    results = bench.evaluate_interruption(pipeline_type="upgraded", trials=15)

    assert len(results) == 15
    pass_count = sum(1 for r in results if r.passed)
    pass_rate = pass_count / len(results)
    assert pass_rate >= 0.95, f"Expected >=95% interruption pass rate, got {pass_rate:.1%}"

    latencies = [r.stop_latency_ms for r in results]
    median_lat = sorted(latencies)[len(latencies) // 2]
    assert median_lat < 300.0, f"Median interruption latency {median_lat}ms exceeds 300ms target"


def test_fullduplexbench_backchannel_floor_retention():
    bench = FullDuplexBench()
    results = bench.evaluate_backchannel(pipeline_type="upgraded", trials=10)

    assert len(results) == 10
    for r in results:
        assert not r.floor_usurped, "Backchannel should never usurp the speaker floor"
        assert r.backchannel_count >= 1, "Expected backchannel emitted during long speech"


def test_fullduplexbench_false_interrupt_rate_under_5pct():
    bench = FullDuplexBench()
    results = bench.evaluate_false_interrupt(pipeline_type="upgraded", trials=20)

    assert len(results) == 20
    fir_count = sum(1 for r in results if r.falsely_interrupted)
    fir_rate_pct = (fir_count / len(results)) * 100.0
    assert fir_rate_pct < 5.0, f"False interrupt rate {fir_rate_pct}% exceeds 5.0% target"


def test_fullduplexbench_full_run_and_report_generation(tmp_path):
    bench = FullDuplexBench()
    data = bench.run_benchmark()

    assert data["benchmark_name"] == "FullDuplexBench-v1.0"
    assert data["upgraded"]["turn_taking_latency_p50_ms"] < 800.0
    assert data["upgraded"]["interruption_latency_p50_ms"] < 300.0
    assert data["upgraded"]["false_interrupt_rate_pct"] < 5.0
    assert data["upgraded"]["backchannel_floor_retention_pct"] == 100.0

    json_path = str(tmp_path / "test_out.json")
    md_path = str(tmp_path / "test_report.md")
    save_reports(data, json_path=json_path, md_path=md_path)

    assert os.path.exists(json_path)
    assert os.path.exists(md_path)
    with open(json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
        assert "baseline" in loaded
        assert "upgraded" in loaded
