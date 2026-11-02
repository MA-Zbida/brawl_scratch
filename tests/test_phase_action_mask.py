from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import PPO

from brawl.control.action_space import ACTION_DIM, Action
from brawl.pipelines.llc.curriculum_config import build_phase_spec
from brawl.pipelines.llc.phase_action_policy import (
    PhaseMaskedActorCriticPolicy,
    masked_policy_custom_objects,
    phase_allowed_action_ids,
    set_model_phase_action_mask,
)


class _TinyEnv(gym.Env):
    observation_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
    action_space = gym.spaces.Discrete(ACTION_DIM)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(4, dtype=np.float32), 0.0, False, False, {}


def _policy_kwargs(allowed: tuple[int, ...]) -> dict:
    return {
        "net_arch": {"pi": [8], "vf": [8]},
        "allowed_action_ids": allowed,
    }


def test_phase_profiles_expose_one_explicit_legal_action_set() -> None:
    expected_sizes = {
        "movement_fluency": 18,
        "weapon_acquisition": 10,
        "spacing_neutral": 18,
        "combat_execution": 27,
        "recovery_mastery": 20,
    }

    for phase, expected_size in expected_sizes.items():
        allowed = phase_allowed_action_ids(build_phase_spec(phase))
        assert len(allowed) == expected_size
        assert int(Action.NOOP) in allowed

    for phase in ("movement_fluency", "spacing_neutral"):
        allowed = phase_allowed_action_ids(build_phase_spec(phase))
        assert int(Action.PICKUP) not in allowed
        assert all(action < int(Action.LIGHT_NEUTRAL) for action in allowed)


def test_explicit_allowlist_is_the_single_source_of_truth() -> None:
    spec = replace(
        build_phase_spec("combat_execution"),
        allowed_actions=(int(Action.NOOP), int(Action.LIGHT_TOWARD)),
        disable_attack=True,
    )

    assert phase_allowed_action_ids(spec) == (
        int(Action.NOOP),
        int(Action.LIGHT_TOWARD),
    )


def test_masked_policy_never_samples_or_credits_disallowed_actions() -> None:
    allowed = (int(Action.NOOP), int(Action.MOVE_AWAY), int(Action.DODGE_TOWARD))
    model = PPO(
        PhaseMaskedActorCriticPolicy,
        _TinyEnv(),
        n_steps=8,
        batch_size=4,
        policy_kwargs=_policy_kwargs(allowed),
        seed=7,
        device="cpu",
    )
    obs = th.zeros((32, 4), dtype=th.float32)

    with th.no_grad():
        distribution = model.policy.get_distribution(obs).distribution
        probs = distribution.probs
        samples = distribution.sample((128,))

    illegal = sorted(set(range(ACTION_DIM)) - set(allowed))
    assert th.count_nonzero(probs[:, illegal]).item() == 0
    assert set(samples.reshape(-1).tolist()).issubset(set(allowed))
    assert th.isneginf(distribution.log_prob(th.full((32,), int(Action.HEAVY_NEUTRAL)))).all()


def test_retention_can_retarget_one_checkpoint_to_each_evaluated_phase() -> None:
    model = PPO(
        PhaseMaskedActorCriticPolicy,
        _TinyEnv(),
        n_steps=8,
        batch_size=4,
        policy_kwargs=_policy_kwargs(tuple(range(ACTION_DIM))),
        seed=7,
        device="cpu",
    )
    obs = th.zeros((2, 4), dtype=th.float32)

    set_model_phase_action_mask(model, "weapon_acquisition")
    weapon_probs = model.policy.get_distribution(obs).distribution.probs
    assert model.policy.allowed_action_ids == phase_allowed_action_ids(
        build_phase_spec("weapon_acquisition")
    )
    weapon_allowed = set(model.policy.allowed_action_ids)
    weapon_illegal = sorted(set(range(ACTION_DIM)) - weapon_allowed)
    assert th.count_nonzero(weapon_probs[:, weapon_illegal]).item() == 0

    set_model_phase_action_mask(model, "combat_execution")
    combat_probs = model.policy.get_distribution(obs).distribution.probs
    assert model.policy.allowed_action_ids == tuple(range(ACTION_DIM))
    assert th.all(combat_probs > 0.0)


def test_legacy_checkpoint_converts_without_changing_network_weights(tmp_path: Path) -> None:
    legacy_kwargs = {"net_arch": {"pi": [8], "vf": [8]}}
    legacy = PPO(
        "MlpPolicy",
        _TinyEnv(),
        n_steps=8,
        batch_size=4,
        policy_kwargs=legacy_kwargs,
        seed=11,
        device="cpu",
    )
    checkpoint = tmp_path / "legacy.zip"
    legacy.save(checkpoint)
    expected = {key: value.detach().clone() for key, value in legacy.policy.state_dict().items()}

    allowed = (int(Action.NOOP), int(Action.MOVE_TOWARD))
    current_kwargs = {**legacy_kwargs, "allowed_action_ids": allowed}
    converted = PPO.load(
        checkpoint,
        env=_TinyEnv(),
        device="cpu",
        custom_objects=masked_policy_custom_objects(current_kwargs),
    )

    assert isinstance(converted.policy, PhaseMaskedActorCriticPolicy)
    assert converted.policy.allowed_action_ids == allowed
    for key, value in converted.policy.state_dict().items():
        th.testing.assert_close(value, expected[key])

    obs = th.zeros((4, 4), dtype=th.float32)
    probs = converted.policy.get_distribution(obs).distribution.probs
    assert th.count_nonzero(probs[:, 2:]).item() == 0

    masked_checkpoint = tmp_path / "masked.zip"
    converted.save(masked_checkpoint)
    reloaded = PPO.load(masked_checkpoint, env=_TinyEnv(), device="cpu")
    assert isinstance(reloaded.policy, PhaseMaskedActorCriticPolicy)
    assert reloaded.policy.allowed_action_ids == allowed
