from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


REQUIRED_COLUMNS = {
    "workload",
    "elements",
    "block_size",
    "gpu_kernel_ms",
    "effective_bandwidth_gbps",
    "gflops",
    "kernel_speedup",
    "maximum_absolute_error",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"sweep CSV is missing columns: {sorted(missing)}")
        return list(reader)


def best_by_problem_size(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["workload"], int(row["elements"]))].append(row)

    best_rows = []
    for key, group in groups.items():
        workload, elements = key
        best = max(group, key=lambda row: float(row["effective_bandwidth_gbps"]))
        best_rows.append(
            {
                "workload": workload,
                "elements": str(elements),
                "block_size": best["block_size"],
                "gpu_kernel_ms": best["gpu_kernel_ms"],
                "effective_bandwidth_gbps": best["effective_bandwidth_gbps"],
                "gflops": best["gflops"],
                "kernel_speedup": best["kernel_speedup"],
                "maximum_absolute_error": best["maximum_absolute_error"],
            }
        )
    return sorted(best_rows, key=lambda row: (row["workload"], int(row["elements"])))


def write_markdown(rows: list[dict[str, str]], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CUDA SAXPY Sweep Summary",
        "",
        "| Workload | Elements | Best block | Kernel ms | Bandwidth GB/s | GFLOP/s | Speedup | Max error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {workload} | {elements} | {block_size} | {gpu_kernel_ms} | "
            "{effective_bandwidth_gbps} | {gflops} | {kernel_speedup} | "
            "{maximum_absolute_error} |".format(**row)
        )
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize cuda_saxpy sweep results.")
    parser.add_argument("--input", type=Path, default=Path("reports/block_sweep.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/block_sweep.md"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = read_rows(args.input)
    write_markdown(best_by_problem_size(rows), args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
