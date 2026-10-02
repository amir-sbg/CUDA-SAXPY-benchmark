from __future__ import annotations

import argparse
import csv
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


def summarize_configurations(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row["workload"], int(row["elements"]), int(row["block_size"]))
        groups[key].append(row)

    summaries = []
    for (workload, elements, block_size), group in groups.items():
        timings = [float(row["gpu_kernel_ms"]) for row in group]
        bandwidths = [float(row["effective_bandwidth_gbps"]) for row in group]
        median_bandwidth = median(bandwidths)
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
                "bandwidth_cv": f"{bandwidth_cv:.4f}",
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
        "| Workload | Elements | Best block | Runs | Median kernel ms | IQR ms | Median GB/s | CV | GFLOP/s | Max error" + efficiency_header + " |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:" + efficiency_rule + "|",
    ]
    for row in rows:
        efficiency = ""
        if peak_bandwidth_gbps is not None:
            percentage = 100.0 * float(row["effective_bandwidth_gbps"]) / peak_bandwidth_gbps
            efficiency = f" {percentage:.1f} |"
        lines.append(
            "| {workload} | {elements} | {block_size} | {runs} | {gpu_kernel_ms} | "
            "{kernel_iqr_ms} | {effective_bandwidth_gbps} | {bandwidth_cv} | {gflops} | "
            "{maximum_absolute_error} |".format(**row) + efficiency
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
