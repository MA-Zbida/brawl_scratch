from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("gymnasium")

import gymnasium as gym

from brawl.control.action_space import ACTION_DIM, Action
from brawl.pipelines.llc.compare_checkpoints import (
    MovementChallenge,
    evaluate_variant,
    movement_challenges,
)


class _FakeModel:
    def __init__(self, action: Action) -> None:
        self.action = int(action)

    def predict(self, obs, *, deterministic):
        return np.asarray([self.action], dtype=np.int64), None


class _FakeMovementEnv:
    def __init__(self) -> None:
        self.action_space = gym.spaces.Discrete(ACTION_DIM)
        self._step = 0
        self._target = None

    def reset(self, *, seed=None):
        self._step = 0
        return np.zeros((4,), dtype=np.float32), {}

    def set_goal_target(self, obs, info, target):
        self._target = np.asarray(target, dtype=np.float32).copy()
        updated = dict(info)
        updated["goal_target_world"] = self._target.copy()
        updated["goal_mask"] = np.ones_like(self._target)
        return obs, updated

    def goal_error(self, obs, info):
        return 0.2

    def step(self, action):
        self._step += 1
        success = self._step == 2
        info = {
            "stage_action": int(action),
            "goal_error": 0.01 if success else 0.1,
            "goal_success": float(success),
            "terminal_success": float(success),
        }
        return np.zeros((4,), dtype=np.float32), 1.0, False, success, info


def test_fixed_movement_challenge_set_covers_ground_and_air() -> None:
    challenges = movement_challenges()

    assert len(challenges) == 5
    assert {challenge.slice for challenge in challenges} == {"ground", "air"}
    assert len({challenge.challenge_id for challenge in challenges}) == len(challenges)


def test_variant_rows_preserve_targets_and_exact_action_sequences() -> None:
    challenges = (
        MovementChallenge("left_ground", "ground", 0.36, 0.57),
        MovementChallenge("right_air", "air", 0.62, 0.42),
    )
    rows, summary = evaluate_variant(
        _FakeModel(Action.JUMP_TOWARD),
        _FakeMovementEnv(),
        challenges,
        model_label="bc",
        model_path="bc.zip",
        deterministic=True,
        seed=42,
    )

    assert [row["challenge_id"] for row in rows] == ["left_ground", "right_air"]
    assert all(json.loads(str(row["action_ids"])) == [int(Action.JUMP_TOWARD)] * 2 for row in rows)
    counts = json.loads(str(rows[0]["action_counts"]))
    assert len(counts) == ACTION_DIM
    assert counts[int(Action.JUMP_TOWARD)] == 2
    assert all(float(row["target_player_x"]) in {0.36, 0.62} for row in rows)
    assert summary["eligible_episodes"] == 2
    assert summary["success_rate"] == pytest.approx(1.0)
