from __future__ import annotations

from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest

from action_space import ACTION_DIM, Action
from feature_extractor.memory.state_spec import StateSpec
from train.curriculum_config import PHASES, build_phase_spec
from train.curriculum_goals import GOAL_INDEX, default_goal_target
from train.llc_stage_common import StageGoalEnv


class _MovingStateEnv(gym.Env):
    metadata = {}

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(StateSpec.dim(),),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(ACTION_DIM)
        self.player_x = 0.2

    def _obs(self) -> np.ndarray:
        obs = np.zeros((StateSpec.dim(),), dtype=np.float32)
        obs[StateSpec.index("player_x")] = self.player_x
        obs[StateSpec.index("player_y")] = 0.5
        obs[StateSpec.index("player_has_weapon")] = 0.0
        obs[StateSpec.index("weapon_dx")] = 0.2
        obs[StateSpec.index("weapon_dy")] = 0.1
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.player_x = 0.2
        return self._obs(), {"player_exists": 1.0, "canon_mirrored": 0.0}

    def step(self, action):
        self.player_x = 0.3
        return self._obs(), 0.0, False, False, {
            "player_exists": 1.0,
            "canon_mirrored": 0.0,
            "op_stock_lost_step": 0.0,
            "self_stock_lost_step": 0.0,
        }


def test_goal_error_is_normalized_by_active_mask_mass() -> None:
    spec = build_phase_spec("movement_fluency", terminate_on_death=False)
    env = StageGoalEnv(_MovingStateEnv(), spec)
    feats = np.zeros(len(spec.feature_names or []), dtype=np.float32)
    target = np.zeros_like(feats)
    mask = np.zeros_like(feats)
    mask[GOAL_INDEX["player_x"]] = 1.0
    mask[GOAL_INDEX["player_y"]] = 1.0
    feats[GOAL_INDEX["player_x"]] = 0.2
    feats[GOAL_INDEX["player_y"]] = 0.2

    assert env._goal_error_from_feats(feats, target, mask=mask) == pytest.approx(0.2)


def test_first_transition_receives_goal_progress_reward() -> None:
    target = default_goal_target()
    target[GOAL_INDEX["player_x"]] = 0.6
    target[GOAL_INDEX["player_y"]] = 0.5
    spec = replace(
        build_phase_spec("movement_fluency", terminate_on_death=False),
        target_sampler=lambda _obs: target.copy(),
        success_bonus=0.0,
        step_penalty=0.0,
        vertical_velocity_penalty_scale=0.0,
        reward_clip=10.0,
    )
    env = StageGoalEnv(_MovingStateEnv(), spec)
    env.reset(seed=7)

    _, reward, _, _, info = env.step(int(Action.MOVE_TOWARD))

    assert info["goal_progress"] > 0.0
    assert reward == pytest.approx(spec.progress_scale * info["goal_progress"])


def test_every_skill_phase_uses_normalized_goal_progress() -> None:
    for phase in PHASES:
        assert build_phase_spec(phase).reward_from_goal_progress is True


def test_combat_goal_is_damage_only() -> None:
    spec = build_phase_spec("combat_execution")
    assert spec.mask[GOAL_INDEX["in_strike_range"]] == 0.0
    assert spec.mask[GOAL_INDEX["opponent_damage_pct"]] == 1.0
    assert spec.in_strike_range_bonus == 0.0


def test_weapon_guidance_mask_disappears_once_armed() -> None:
    spec = build_phase_spec("weapon_acquisition", terminate_on_death=False)
    env = StageGoalEnv(_MovingStateEnv(), spec)
    feats = np.zeros(len(spec.feature_names or []), dtype=np.float32)

    unarmed = env._effective_mask_for_step(feats)
    feats[GOAL_INDEX["player_has_weapon"]] = 1.0
    armed = env._effective_mask_for_step(feats)

    assert unarmed[GOAL_INDEX["weapon_dx"]] > 0.0
    assert unarmed[GOAL_INDEX["weapon_dy"]] > 0.0
    assert armed[GOAL_INDEX["player_has_weapon"]] == 1.0
    assert armed[GOAL_INDEX["weapon_dx"]] == 0.0
    assert armed[GOAL_INDEX["weapon_dy"]] == 0.0
