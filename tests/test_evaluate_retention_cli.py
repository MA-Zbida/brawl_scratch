from __future__ import annotations

import sys
from types import SimpleNamespace

import brawl.env.live as env_module
import brawl.pipelines.llc.curriculum_config as curriculum_config
import brawl.pipelines.llc.stage_common as llc_stage_common
import brawl.pipelines.llc.train_curriculum as train_curriculum
from brawl.pipelines.llc.evaluate_retention import make_env, parse_args


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


def test_retention_environment_keeps_goal_cursor_visible(monkeypatch) -> None:
    class FakeBaseEnv:
        def __init__(self, *, config) -> None:
            self.config = config

    class FakeStageGoalEnv:
        def __init__(self, env, spec) -> None:
            self.env = env
            self.spec = spec

    class FakeGoalMouseTracker:
        def __init__(self, env) -> None:
            self.env = env

    monkeypatch.setattr(env_module, "EnvConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(env_module, "BrawlDeepEnv", FakeBaseEnv)
    monkeypatch.setattr(
        curriculum_config,
        "build_phase_spec",
        lambda **kwargs: SimpleNamespace(terminate_on_death=True),
    )
    monkeypatch.setattr(llc_stage_common, "StageGoalEnv", FakeStageGoalEnv)
    monkeypatch.setattr(train_curriculum, "GoalMouseTrackerWrapper", FakeGoalMouseTracker)
    args = SimpleNamespace(max_episode_steps=200, yolo_every=1, death_penalty=1.0)

    wrapped = make_env(args, "movement_fluency")

    assert isinstance(wrapped, FakeGoalMouseTracker)
    assert isinstance(wrapped.env, FakeStageGoalEnv)
