from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path


def test_hsp_runtime_modules_are_removed() -> None:
    assert importlib.util.find_spec("brawl.pipelines.llc.hsp_env") is None
    assert importlib.util.find_spec("brawl.pipelines.llc.train_phase3_hsp") is None
    assert importlib.util.find_spec("brawl.training.hsp_guard") is None


def test_core_training_and_evaluation_import_without_hsp_modules() -> None:
    for module in (
        "brawl.pipelines.llc.train_curriculum",
        "brawl.pipelines.final_agent.run",
        "brawl.pipelines.final_agent.run_match",
        "brawl.pipelines.llc.evaluate_retention",
    ):
        assert importlib.import_module(module) is not None


def test_runtime_code_has_no_hsp_import_or_goal_scheduler() -> None:
    roots = (Path("brawl"), Path("scripts"))
    forbidden = (
        "hierarchical.hsp_env",
        "train.hsp_guard",
        "train_phase3_hsp",
        "HierarchicalAgent",
        "goal_period",
    )
    offenders: list[str] = []
    for root in roots:
        paths = [root] if root.is_file() else list(root.rglob("*.py"))
        for path in paths:
            text = path.read_text(encoding="utf-8")
            for needle in forbidden:
                if needle in text:
                    offenders.append(f"{path}:{needle}")
    assert offenders == []
