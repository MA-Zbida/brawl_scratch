from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from brawl.control.action_space import ACTION_DIM
from brawl.pipelines.llc.film_extractor import StageGoalFiLMExtractor
from brawl.state.state_spec import StateSpec
from brawl.pipelines.final_agent.extractor import FinalStateExtractor
from brawl.pipelines.final_agent.policy_transfer import transfer_skill_policy


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
        return (
            np.zeros(self.observation_space.shape, dtype=np.float32),
            0.0,
            False,
            False,
            {},
        )


def _skill_model() -> PPO:
    base_dim = StateSpec.observation_dim((2, 4, 8))
    return PPO(
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


def _final_model() -> PPO:
    base_dim = StateSpec.observation_dim((2, 4, 8))
    return PPO(
        "MlpPolicy",
        _ContractEnv(base_dim),
        n_steps=8,
        batch_size=4,
        policy_kwargs={
            "features_extractor_class": FinalStateExtractor,
            "features_extractor_kwargs": {"features_dim": 256},
            "net_arch": {"pi": [128], "vf": [128]},
        },
        device="cpu",
    )


def test_skill_policy_remains_goal_conditioned() -> None:
    model = _skill_model()
    base_dim = StateSpec.observation_dim((2, 4, 8))
    obs = np.zeros((1, base_dim + 2 * len(GOAL_FEATURES)), dtype=np.float32)

    action, _ = model.predict(obs, deterministic=True)

    assert np.asarray(action).shape == (1,)
    assert model.policy.features_extractor.base_dim == base_dim
    assert model.observation_space.shape == (168,)


def test_final_policy_accepts_state_without_goal() -> None:
    model = _final_model()
    base_dim = StateSpec.observation_dim((2, 4, 8))
    obs = np.zeros((1, base_dim), dtype=np.float32)

    action, _ = model.predict(obs, deterministic=True)
    features = model.policy.features_extractor(torch.as_tensor(obs))

    assert np.asarray(action).shape == (1,)
    assert features.shape == (1, 256)
    assert model.observation_space.shape == (base_dim,)
    assert not hasattr(model.policy.features_extractor, "goal_encoder")
    assert not hasattr(model.policy.features_extractor, "film_generator")


def test_transfer_copies_compatible_state_and_heads_and_reports_goal_skips() -> None:
    skill = _skill_model()
    final = _final_model()
    base_dim = StateSpec.observation_dim((2, 4, 8))

    with torch.no_grad():
        for index, parameter in enumerate(skill.policy.parameters(), start=1):
            parameter.fill_(index / 100.0)

    fresh_value = final.policy.value_net.weight.detach().clone()
    fresh_critic = final.policy.mlp_extractor.value_net[0].weight.detach().clone()
    report = transfer_skill_policy(skill.policy, final.policy)
    source = skill.policy.state_dict()
    target = final.policy.state_dict()

    assert torch.equal(
        target["features_extractor.state_encoder.0.weight"],
        source["features_extractor.state_encoder.0.weight"][:, :base_dim],
    )
    assert torch.equal(
        target["features_extractor.state_encoder.2.weight"],
        source["features_extractor.state_encoder.2.weight"],
    )
    assert torch.equal(
        target["features_extractor.adapter.1.weight"],
        source["features_extractor.post_film.1.weight"],
    )
    assert torch.equal(target["action_net.weight"], source["action_net.weight"])
    assert torch.equal(target["value_net.weight"], fresh_value)
    assert torch.equal(final.policy.mlp_extractor.value_net[0].weight, fresh_critic)
    assert report.transferred_count > 0
    assert report.skipped_count > 0
    assert any("goal_encoder" in item.source for item in report.skipped)
    assert any("film_generator" in item.source for item in report.skipped)
    assert all(item.reason for item in report.skipped)


def test_transfer_rejects_non_goal_conditioned_source() -> None:
    final_source = _final_model()
    final_target = _final_model()

    try:
        transfer_skill_policy(final_source.policy, final_target.policy)
    except TypeError as exc:
        assert "goal-conditioned skill policy" in str(exc)
    else:
        raise AssertionError("a final policy must not be accepted as a skill source")
