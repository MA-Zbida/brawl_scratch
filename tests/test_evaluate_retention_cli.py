from __future__ import annotations

import sys

from train.evaluate_retention import parse_args


def test_retention_evaluator_cli_parses_without_training_stack(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_retention",
            "--model",
            "train/models/llc_all_skills_llc.zip",
            "--phase",
            "all_skills_llc",
            "--phases",
            "all",
            "--best-scores",
            "train/models/llc_retention_best.json",
            "--csv",
            "train/models/retention_eval.csv",
        ],
    )

    args = parse_args()

    assert args.phase == "all_skills_llc"
    assert args.phases == "all"
    assert args.best_scores.endswith("llc_retention_best.json")
    assert args.csv.endswith("retention_eval.csv")

