from __future__ import annotations

import sys

from train.train_config import make_config, parse_args


def test_train_config_parses_anti_collapse_flags(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_curriculum",
            "--phase",
            "all_skills_llc",
            "--bc-demos-path",
            "a.npz;b.npz",
            "--anchor-pool-size",
            "5",
            "--eval-include-previous",
            "--retention-scores-path",
            "train/models/llc_retention_best.json",
        ],
    )

    cfg = parse_args()

    assert cfg.phase == "all_skills_llc"
    assert cfg.bc_demos_path == "a.npz;b.npz"
    assert cfg.anchor_pool_size == 5
    assert cfg.eval_include_previous is True
    assert cfg.retention_scores_path == "train/models/llc_retention_best.json"


def test_all_skills_defaults_are_consolidation_oriented() -> None:
    cfg = make_config("all_skills_llc")
    assert cfg.replay_ratio >= 0.40
    assert cfg.anchor_kl_coef >= 0.04
    assert cfg.bc_loss_coef >= 0.08
    assert cfg.eval_include_previous is True

