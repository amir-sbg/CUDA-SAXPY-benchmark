from __future__ import annotations

from pathlib import Path

import pytest

from scripts.summarize_sweep import (
    best_by_problem_size,
    bootstrap_median_interval,
    bootstrap_paired_bandwidth_delta,
    compare_summaries,
    summarize_configurations,
    write_markdown,
)


def _row(block: int, bandwidth: float, kernel_ms: float) -> dict[str, str]:
    return {
        "workload": "saxpy",
        "effective_alpha": "2.0",
        "tolerance": "1e-5",
        "elements": "1024",
        "block_size": str(block),
        "gpu_kernel_ms": str(kernel_ms),
        "effective_bandwidth_gbps": str(bandwidth),
        "gflops": str(bandwidth / 6),
        "arithmetic_intensity_flop_per_byte": str(1 / 6),
        "kernel_speedup": "2.0",
        "maximum_absolute_error": "1e-7",
        "mean_absolute_error": "1e-8",
    }


def test_configuration_summary_uses_robust_statistics() -> None:
    rows = [_row(128, 100, 2.0), _row(128, 102, 1.8), _row(128, 400, 0.4)]

    summary = summarize_configurations(rows)[0]

    assert summary["runs"] == "3"
    assert float(summary["effective_bandwidth_gbps"]) == 102.0
    assert float(summary["gpu_kernel_ms"]) == 1.8
    assert float(summary["kernel_ns_per_element"]) == pytest.approx(1757.8125, rel=1e-3)
    assert float(summary["kernel_iqr_ms"]) > 0.0
    assert summary["bandwidth_stability"] == "noisy"
    assert float(summary["bandwidth_mad_gbps"]) == 2.0
    assert summary["bandwidth_outlier_runs"] == "1"
    assert float(summary["bandwidth_ci95_low_gbps"]) <= 102.0
    assert float(summary["bandwidth_ci95_high_gbps"]) >= 102.0


def test_bootstrap_median_interval_is_reproducible() -> None:
    first = bootstrap_median_interval([95.0, 100.0, 102.0, 104.0, 300.0], seed=8)
    second = bootstrap_median_interval([95.0, 100.0, 102.0, 104.0, 300.0], seed=8)

    assert first == second
    assert first[0] <= 102.0 <= first[1]


def test_paired_bootstrap_detects_consistent_bandwidth_gain() -> None:
    report = bootstrap_paired_bandwidth_delta(
        [100.0, 102.0, 98.0, 101.0],
        [110.0, 111.0, 108.0, 112.0],
        samples=500,
        seed=4,
    )

    assert report["ci_low_gbps"] > 0
    assert report["probability_improves"] == pytest.approx(1.0)
    assert report["ci_low_gbps"] <= report["median_delta_gbps"] <= report["ci_high_gbps"]


def test_best_block_uses_median_bandwidth_instead_of_single_outlier() -> None:
    rows = [
        _row(128, 100, 2.0),
        _row(128, 101, 1.9),
        _row(256, 90, 2.2),
        _row(256, 91, 2.1),
        _row(256, 500, 0.3),
    ]

    best = best_by_problem_size(rows)

    assert best[0]["block_size"] == "128"


def test_summary_comparison_flags_bandwidth_regression() -> None:
    baseline = summarize_configurations([_row(256, 120, 1.0)])
    candidate = summarize_configurations([_row(256, 102, 1.2)])

    comparison = compare_summaries(baseline, candidate, regression_threshold=0.10)

    assert len(comparison) == 1
    assert comparison[0]["status"] == "regression"
    assert float(comparison[0]["bandwidth_delta_percent"]) == pytest.approx(-15.0)
    assert float(comparison[0]["kernel_latency_delta_percent"]) == pytest.approx(20.0)


def test_summary_keeps_optimizer_step_sizes_separate() -> None:
    small_step = _row(128, 100, 1.0)
    large_step = _row(128, 105, 0.9)
    small_step["workload"] = "sgd-step"
    large_step["workload"] = "sgd-step"
    small_step["effective_alpha"] = "-0.001"
    large_step["effective_alpha"] = "-0.1"

    summary = summarize_configurations([small_step, large_step])

    assert len(summary) == 2
    assert {row["effective_alpha"] for row in summary} == {"-0.001", "-0.1"}


def test_summary_reports_numerical_headroom() -> None:
    row = _row(128, 100, 1.0)
    row["maximum_absolute_error"] = "2e-6"

    summary = summarize_configurations([row])[0]

    assert float(summary["error_to_tolerance_ratio"]) == pytest.approx(0.2)
    assert summary["numerical_status"] == "pass"

    row["maximum_absolute_error"] = "2e-5"
    assert summarize_configurations([row])[0]["numerical_status"] == "fail"


def test_markdown_can_report_fraction_of_peak_bandwidth(tmp_path: Path) -> None:
    summary = best_by_problem_size([_row(256, 120, 1.0)])
    output = write_markdown(summary, tmp_path / "summary.md", peak_bandwidth_gbps=240.0)

    text = output.read_text(encoding="utf-8")
    assert "Peak bandwidth %" in text
    assert "Kernel ns/elem" in text
    assert "95% CI GB/s" in text
    assert "MAD GB/s" in text
    assert "Outliers" in text
    assert "Correctness" in text
    assert "50.0" in text

    with pytest.raises(ValueError, match="peak bandwidth"):
        write_markdown(summary, tmp_path / "bad.md", peak_bandwidth_gbps=0)


def test_markdown_can_report_roofline_utilization(tmp_path: Path) -> None:
    summary = best_by_problem_size([_row(256, 120, 1.0)])
    output = write_markdown(
        summary,
        tmp_path / "roofline.md",
        peak_bandwidth_gbps=240.0,
        peak_gflops=10_000.0,
    )

    text = output.read_text(encoding="utf-8")
    assert "Roofline %" in text
    assert "memory" in text
    assert "50.0" in text

    with pytest.raises(ValueError, match="peak bandwidth"):
        write_markdown(summary, tmp_path / "bad.md", peak_gflops=10_000.0)
