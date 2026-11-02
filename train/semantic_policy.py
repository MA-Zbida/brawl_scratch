"""PPO with masks carried IN each observation, including stored rollout rows.

This avoids a second masking implementation or a dependency on sb3-contrib. BC,
forward, evaluate_actions and deterministic predict share the same distribution.
"""
from __future__ import annotations

from pathlib import Path

import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from train.checkpoint_roles import read_checkpoint_metadata, save_typed_checkpoint
from train.semantic_contract import ACTION_DIM, OBSERVATION_DIM, metadata, validate_metadata


class SemanticFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, OBSERVATION_DIM - ACTION_DIM)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return obs[..., :-ACTION_DIM]


class SemanticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        kwargs["features_extractor_class"] = SemanticFeatures
        kwargs["share_features_extractor"] = True
        super().__init__(*args, **kwargs)
        if self.observation_space.shape != (OBSERVATION_DIM,) or self.action_space.n != ACTION_DIM:
            raise ValueError("Semantic policy observation/action dimension mismatch")

    def _distribution(self, obs: th.Tensor, latent: th.Tensor):
        mask = obs[..., -ACTION_DIM:]
        if not th.all((mask == 0) | (mask == 1)) or not th.all(mask.sum(-1) > 0):
            raise ValueError("Invalid or empty semantic action mask")
        logits = self.action_net(latent).masked_fill(mask == 0, -1e9)
        return CategoricalDistribution(ACTION_DIM).proba_distribution(action_logits=logits)

    def get_distribution(self, obs):
        features = self.extract_features(obs)
        latent = self.mlp_extractor.forward_actor(features)
        return self._distribution(obs, latent)

    def forward(self, obs, deterministic=False):
        latent_pi, latent_vf = self.mlp_extractor(self.extract_features(obs))
        distribution = self._distribution(obs, latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions, self.value_net(latent_vf), distribution.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        latent_pi, latent_vf = self.mlp_extractor(self.extract_features(obs))
        distribution = self._distribution(obs, latent_pi)
        return self.value_net(latent_vf), distribution.log_prob(actions.long().flatten()), distribution.entropy()


def build_model(env, **overrides) -> PPO:
    options = dict(n_steps=512, batch_size=256, n_epochs=4, learning_rate=1e-4,
                   gamma=.999, gae_lambda=.95, clip_range=.1, ent_coef=.01,
                   target_kl=.01, device="cpu", seed=42, verbose=0,
                   policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}})
    options.update(overrides)
    return PPO(SemanticPolicy, env, **options)


def save_model(model: PPO, path: Path, **extra) -> Path:
    return save_typed_checkpoint(model, path, metadata() | extra)


def load_model(path: Path, env=None, **overrides) -> PPO:
    validate_metadata(read_checkpoint_metadata(path))
    model = PPO.load(path, env=env, device="cpu", **overrides)
    validate_metadata(model.checkpoint_metadata)
    if not isinstance(model.policy, SemanticPolicy):
        raise ValueError("Checkpoint does not contain a mask-aware semantic policy")
    return model
