from __future__ import annotations

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from brawl.control.action_space import ACTION_DIM
from brawl.pipelines.llc.film_extractor import StageGoalFiLMExtractor
from brawl.pipelines.final_agent.extractor import FinalStateExtractor
from brawl.state.state_spec import StateSpec
from brawl.training.checkpoints import (
    FINAL_AGENT,
    GOAL_CONDITIONED_SKILL,
    make_checkpoint_metadata,
    read_checkpoint_metadata,
    save_typed_checkpoint,
)
from brawl.pipelines.final_agent.policy import convert_skill_checkpoint, load_final_model


GOAL_FEATURES = [
    "signed_dx_to_ledge",
    "dy_to_ledge",
    "player_x",
    "player_y",
    "player_has_weapon",
    "weapon_dx",
    "weapon_dy",
    "rel_distance",
    "rel_dy",
    "in_strike_range",
    "opponent_damage_pct",
]


class _ContractEnv(gym.Env):
    def __init__(self, obs_dim: int) -> None:
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(ACTION_DIM)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, False, False, {}


def _write_typed_skill(path) -> None:
    base_dim = StateSpec.observation_dim((2, 4, 8))
    model = PPO(
        "MlpPolicy",
        _ContractEnv(base_dim + 2 * len(GOAL_FEATURES)),
        n_steps=8,
        batch_size=4,
        policy_kwargs={
            "features_extractor_class": StageGoalFiLMExtractor,
            "features_extractor_kwargs": {
                "goal_feature_names": GOAL_FEATURES,
                "features_dim": 256,
            },
            "net_arch": {"pi": [128], "vf": [128]},
        },
        device="cpu",
    )
    metadata = make_checkpoint_metadata(
        policy_type=GOAL_CONDITIONED_SKILL,
        observation_dim=base_dim + 2 * len(GOAL_FEATURES),
        action_dim=ACTION_DIM,
        goal_dim=len(GOAL_FEATURES),
    )
    save_typed_checkpoint(model, path, metadata)


def test_typed_skill_checkpoint_converts_to_goal_free_final_checkpoint(tmp_path) -> None:
    skill = tmp_path / "skill.zip"
    final = tmp_path / "final.zip"
    _write_typed_skill(skill)

    report = convert_skill_checkpoint(skill, final, device="cpu")
    loaded = load_final_model(final, device="cpu")
    metadata = read_checkpoint_metadata(final)

    assert report.transferred_count > 0
    assert metadata["policy_type"] == FINAL_AGENT
    assert metadata["initialized_from"] == str(skill)
    assert loaded.observation_space.shape == (StateSpec.observation_dim((2, 4, 8)),)
    assert isinstance(loaded.policy.features_extractor, FinalStateExtractor)


def test_untyped_legacy_skill_requires_explicit_override(tmp_path) -> None:
    typed = tmp_path / "typed.zip"
    legacy = tmp_path / "legacy.zip"
    final = tmp_path / "final.zip"
    _write_typed_skill(typed)
    legacy.write_bytes(typed.read_bytes())

    try:
        convert_skill_checkpoint(legacy, final, device="cpu")
    except ValueError as exc:
        assert "Legacy untyped" in str(exc)
    else:
        raise AssertionError("untyped skill checkpoint was accepted silently")

    report = convert_skill_checkpoint(
        legacy,
        final,
        device="cpu",
        allow_untyped_skill_checkpoint=True,
    )
    assert report.transferred_count > 0
