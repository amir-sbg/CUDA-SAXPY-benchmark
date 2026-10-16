from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from statistics import median, stdev


REQUIRED_COLUMNS = {
    "workload",
    "elements",
    "block_size",
    "gpu_kernel_ms",
    "effective_bandwidth_gbps",
    "gflops",
    "kernel_speedup",
    "maximum_absolute_error",
    "mean_absolute_error",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"sweep CSV is missing columns: {sorted(missing)}")
        return list(reader)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _kernel_ns_per_element(row: dict[str, str]) -> float:
    value = row.get("kernel_ns_per_element")
    if value:
        return float(value)
    return float(row["gpu_kernel_ms"]) * 1_000_000.0 / float(row["elements"])


def bootstrap_median_interval(
    values: list[float],
    *,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    if not values:
        raise ValueError("values must not be empty")
    if samples < 1:
        raise ValueError("samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if len(values) == 1:
        return values[0], values[0]

    generator = random.Random(seed)
    estimates = [
        median(generator.choices(values, k=len(values)))
        for _ in range(samples)
    ]
    tail = (1.0 - confidence) / 2.0
    return _percentile(estimates, tail), _percentile(estimates, 1.0 - tail)


def _stability_label(bandwidth_cv: float) -> str:
    if bandwidth_cv <= 0.03:
        return "stable"
    if bandwidth_cv <= 0.10:
        return "moderate"
    return "noisy"


def summarize_configurations(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row["workload"], int(row["elements"]), int(row["block_size"]))
        groups[key].append(row)

    summaries = []
    for (workload, elements, block_size), group in groups.items():
        timings = [float(row["gpu_kernel_ms"]) for row in group]
        bandwidths = [float(row["effective_bandwidth_gbps"]) for row in group]
        kernel_latencies = [_kernel_ns_per_element(row) for row in group]
        median_bandwidth = median(bandwidths)
        bandwidth_ci_low, bandwidth_ci_high = bootstrap_median_interval(
            bandwidths,
            seed=elements + block_size,
        )
        bandwidth_cv = stdev(bandwidths) / median_bandwidth if len(group) > 1 and median_bandwidth else 0.0
        summaries.append(
            {
                "workload": workload,
                "elements": str(elements),
                "block_size": str(block_size),
                "runs": str(len(group)),
                "gpu_kernel_ms": f"{median(timings):.6f}",
                "kernel_iqr_ms": f"{_percentile(timings, 0.75) - _percentile(timings, 0.25):.6f}",
                "effective_bandwidth_gbps": f"{median_bandwidth:.6f}",
                "bandwidth_ci95_low_gbps": f"{bandwidth_ci_low:.6f}",
                "bandwidth_ci95_high_gbps": f"{bandwidth_ci_high:.6f}",
                "kernel_ns_per_element": f"{median(kernel_latencies):.3f}",
                "bandwidth_cv": f"{bandwidth_cv:.4f}",
                "bandwidth_stability": _stability_label(bandwidth_cv),
                "gflops": f"{median([float(row['gflops']) for row in group]):.6f}",
                "kernel_speedup": f"{median([float(row['kernel_speedup']) for row in group]):.6f}",
                "maximum_absolute_error": f"{max(float(row['maximum_absolute_error']) for row in group):.6g}",
                "mean_absolute_error": f"{median([float(row['mean_absolute_error']) for row in group]):.6g}",
            }
        )
    return summaries


def best_by_problem_size(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in summarize_configurations(rows):
        groups[(row["workload"], int(row["elements"]))].append(row)

    best_rows = []
    for key, group in groups.items():
        workload, elements = key
        best = max(group, key=lambda row: float(row["effective_bandwidth_gbps"]))
        best_rows.append({**best, "workload": workload, "elements": str(elements)})
    return sorted(best_rows, key=lambda row: (row["workload"], int(row["elements"])))


def write_markdown(
    rows: list[dict[str, str]],
    output: Path,
    peak_bandwidth_gbps: float | None = None,
) -> Path:
    if peak_bandwidth_gbps is not None and peak_bandwidth_gbps <= 0:
        raise ValueError("peak bandwidth must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    efficiency_header = " | Peak bandwidth %" if peak_bandwidth_gbps is not None else ""
    efficiency_rule = "|---:" if peak_bandwidth_gbps is not None else ""
    lines = [
        "# CUDA SAXPY Sweep Summary",
        "",
        "| Workload | Elements | Best block | Runs | Median kernel ms | IQR ms | Median GB/s | 95% CI GB/s | Kernel ns/elem | CV | Stability | GFLOP/s | Max error" + efficiency_header + " |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:" + efficiency_rule + "|",
    ]
    for row in rows:
        efficiency = ""
        if peak_bandwidth_gbps is not None:
            percentage = 100.0 * float(row["effective_bandwidth_gbps"]) / peak_bandwidth_gbps
            efficiency = f" {percentage:.1f} |"
        lines.append(
            "| {workload} | {elements} | {block_size} | {runs} | {gpu_kernel_ms} | "
            "{kernel_iqr_ms} | {effective_bandwidth_gbps} | "
            "[{bandwidth_ci95_low_gbps}, {bandwidth_ci95_high_gbps}] | {kernel_ns_per_element} | "
            "{bandwidth_cv} | {bandwidth_stability} | {gflops} | {maximum_absolute_error} |".format(**row) +
            efficiency
        )
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize cuda_saxpy sweep results.")
    parser.add_argument("--input", type=Path, default=Path("reports/block_sweep.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/block_sweep.md"))
    parser.add_argument(
        "--peak-bandwidth-gbps",
        type=float,
        help="optional theoretical device bandwidth for an efficiency estimate",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = read_rows(args.input)
    write_markdown(best_by_problem_size(rows), args.output, args.peak_bandwidth_gbps)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
