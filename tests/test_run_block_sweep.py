from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.run_block_sweep import run_sweep


def _args(tmp_path: Path, **overrides) -> Namespace:
    binary = tmp_path / "cuda_saxpy"
    binary.touch()
    values = {
        "binary": binary,
        "output": tmp_path / "sweep.csv",
        "blocks": [128, 256],
        "elements": [1024, 2048],
        "iterations": 5,
        "warmup": 1,
        "repeats": 2,
        "workload": "saxpy",
        "alpha": 2.0,
        "learning_rate": 0.01,
        "tolerance": 1e-5,
        "seed": 7,
    }
    values.update(overrides)
    return Namespace(**values)


def test_sweep_runs_every_configuration_for_each_repeat(tmp_path: Path) -> None:
    args = _args(tmp_path)

    with patch("scripts.run_block_sweep.subprocess.run") as run:
        run_sweep(args)

    assert run.call_count == 8
    assert all(call.kwargs["check"] is True for call in run.call_args_list)


def test_sweep_rejects_invalid_repeat_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="repeats"):
        run_sweep(_args(tmp_path, repeats=0))
