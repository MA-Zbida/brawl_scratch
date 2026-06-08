from __future__ import annotations

import sys

import pytest


def test_demo_collection_cli_accepts_current_phase_names(monkeypatch) -> None:
    pytest.importorskip("gymnasium")
    pytest.importorskip("ultralytics")

    from train.collect_bc_locomotion_demos import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_bc_locomotion_demos",
            "--phase",
            "combat_execution",
            "--episodes",
            "1",
            "--max-episode-steps",
            "10",
        ],
    )

    args = parse_args()
    assert args.phase == "combat_execution"
    assert args.episodes == 1


def test_bc_pretrain_cli_accepts_current_phase_names(monkeypatch) -> None:
    pytest.importorskip("gymnasium")
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("ultralytics")

    from train.pretrain_bc_locomotion import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pretrain_bc_locomotion",
            "--phase",
            "movement_fluency",
            "--epochs",
            "1",
            "--demos",
            "train/models/movement_fluency_demos.npz",
        ],
    )

    args = parse_args()
    assert args.phase == "movement_fluency"
    assert args.epochs == 1

