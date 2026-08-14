from __future__ import annotations

import csv

import numpy as np
import pytest

from feature_extractor.memory.state_spec import StateSpec
from reward.providers import MatchRewardProvider, NullRewardProvider
from train.final_agent import FinalTrainingMetricsCallback
from train.final_config import parse_args


def test_final_agent_config_is_goal_free_cpu_match_training() -> None:
    cfg = parse_args(["--init-from", "skill.zip", "--timesteps", "0"])

    assert cfg.init_from == "skill.zip"
    assert cfg.device == "cpu"
    assert cfg.goal_conditioning is False
    assert cfg.objective == "match"
    assert cfg.timesteps == 0


@pytest.mark.parametrize(
    "obsolete",
    ["--macro-steps", "--goal-period", "--hsp-checkpoint", "--high-level-policy"],
)
def test_final_agent_rejects_obsolete_hsp_arguments(obsolete: str) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--init-from", "skill.zip", obsolete, "12"])


def test_final_agent_requires_exactly_one_initialization_path() -> None:
    with pytest.raises(SystemExit):
        parse_args([])
    with pytest.raises(SystemExit):
        parse_args(["--init-from", "skill.zip", "--resume", "final.zip"])


def test_legacy_reward_name_is_an_alias_for_the_explicit_match_reward() -> None:
    assert NullRewardProvider is MatchRewardProvider


def test_final_metrics_read_state_from_goal_free_transition(tmp_path) -> None:
    output = tmp_path / "steps.csv"
    callback = FinalTrainingMetricsCallback(output)
    callback._on_training_start()
    obs = np.zeros((1, StateSpec.observation_dim((2, 4, 8))), dtype=np.float32)
    obs[0, StateSpec.index("player_is_offstage")] = 1.0
    obs[0, StateSpec.index("player_has_weapon")] = 1.0
    callback.locals = {
        "new_obs": obs,
        "infos": [{}],
        "rewards": np.asarray([2.5], dtype=np.float32),
        "actions": np.asarray([[7]], dtype=np.int64),
    }
    callback.num_timesteps = 3

    assert callback._on_step() is True
    callback._on_training_end()

    with output.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["environment_step"] == "3"
    assert row["action_id"] == "7"
    assert row["player_offstage"] == "1.0"
    assert row["player_has_weapon"] == "1.0"
