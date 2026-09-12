from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a block-size sweep for cuda_saxpy.")
    parser.add_argument("--binary", type=Path, default=Path("build/cuda_saxpy"))
    parser.add_argument("--output", type=Path, default=Path("reports/block_sweep.csv"))
    parser.add_argument("--blocks", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--elements", nargs="+", type=int, default=[1 << 24])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--workload", choices=["saxpy", "sgd-step"], default="saxpy")
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def run_sweep(args: argparse.Namespace) -> None:
    if not args.binary.exists():
        raise FileNotFoundError(f"benchmark binary was not found: {args.binary}")
    if any(block < 1 for block in args.blocks):
        raise ValueError("all block sizes must be positive")
    if any(elements < 1 for elements in args.elements) or args.iterations < 1:
        raise ValueError("elements and iterations must be positive")
    if args.tolerance <= 0:
        raise ValueError("tolerance must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for elements in args.elements:
        for block_size in args.blocks:
            command = [
                str(args.binary),
                "--elements",
                str(elements),
                "--iterations",
                str(args.iterations),
                "--warmup",
                str(args.warmup),
                "--block-size",
                str(block_size),
                "--workload",
                args.workload,
                "--alpha",
                str(args.alpha),
                "--learning-rate",
                str(args.learning_rate),
                "--tolerance",
                str(args.tolerance),
                "--seed",
                str(args.seed),
                "--csv-output",
                str(args.output),
            ]
            print(f"running elements={elements} block_size={block_size}")
            subprocess.run(command, check=True)


if __name__ == "__main__":
    run_sweep(build_parser().parse_args())
