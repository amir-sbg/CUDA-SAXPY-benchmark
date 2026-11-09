from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.run_block_sweep import build_command, run_sweep


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
        "shuffle": False,
        "workload": "saxpy",
        "alpha": 2.0,
        "learning_rate": 0.01,
        "learning_rates": None,
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


def test_shuffled_sweep_is_seeded_and_keeps_every_configuration(tmp_path: Path) -> None:
    args = _args(tmp_path, shuffle=True, seed=19)

    with patch("scripts.run_block_sweep.subprocess.run") as first:
        run_sweep(args)
    with patch("scripts.run_block_sweep.subprocess.run") as second:
        run_sweep(args)

    first_commands = [call.args[0] for call in first.call_args_list]
    second_commands = [call.args[0] for call in second.call_args_list]
    assert first_commands == second_commands
    assert len(first_commands) == 8
    observed = {
        (command[command.index("--elements") + 1], command[command.index("--block-size") + 1])
        for command in first_commands
    }
    assert observed == {("1024", "128"), ("1024", "256"), ("2048", "128"), ("2048", "256")}


def test_optimizer_sweep_runs_each_learning_rate(tmp_path: Path) -> None:
    args = _args(
        tmp_path,
        workload="sgd-step",
        learning_rates=[0.1, 0.01, 0.001],
        repeats=1,
    )

    with patch("scripts.run_block_sweep.subprocess.run") as run:
        run_sweep(args)

    assert run.call_count == 12
    observed = {
        command[command.index("--learning-rate") + 1]
        for command in (call.args[0] for call in run.call_args_list)
    }
    assert observed == {"0.1", "0.01", "0.001"}


def test_learning_rate_grid_is_rejected_for_plain_saxpy(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sgd-step"):
        run_sweep(_args(tmp_path, learning_rates=[0.1, 0.01]))


def test_command_builder_keeps_optimizer_parameters_explicit(tmp_path: Path) -> None:
    args = _args(tmp_path, workload="sgd-step", alpha=3.0)

    command = build_command(args, elements=4096, block_size=256, learning_rate=0.005)

    assert command[command.index("--workload") + 1] == "sgd-step"
    assert command[command.index("--alpha") + 1] == "3.0"
    assert command[command.index("--learning-rate") + 1] == "0.005"


def test_dry_run_does_not_require_or_launch_binary(tmp_path: Path) -> None:
    args = _args(
        tmp_path,
        binary=tmp_path / "missing-binary",
        repeats=1,
        dry_run=True,
    )

    with patch("scripts.run_block_sweep.subprocess.run") as run:
        run_sweep(args)

    run.assert_not_called()
