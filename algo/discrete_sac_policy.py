"""Discrete SAC policy for MultiDiscrete action spaces (flattened to Discrete).

Implements:
- Categorical actor (64-way softmax)
- Dual Q-networks (each outputs 64 Q-values)
- Target Q-networks (polyak-updated copies)
- Dict observation support: concatenates obs keys → flat → FiLM extractor

Designed to be plugged into ``DiscreteSAC`` via ``policy_class=DiscreteSACPolicy``.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.type_aliases import PyTorchObs


class DictToFlatExtractor(BaseFeaturesExtractor):
    """Bridges Dict obs to a flat-obs FiLM extractor.

    Concatenates ``observation(51) + desired_goal(7) + mask(7)`` from the
    Dict observation space, then delegates to an inner ``BaseFeaturesExtractor``
    that expects flat 65-dim input (e.g. ``StageGoalFiLMExtractor``).

    Parameters
    ----------
    observation_space : gym.spaces.Dict
        Must have keys ``observation``, ``achieved_goal``, ``desired_goal``.
    inner_extractor_class : type[BaseFeaturesExtractor]
        The extractor to use on the flattened obs (e.g. StageGoalFiLMExtractor).
    inner_extractor_kwargs : dict
        Kwargs forwarded to the inner extractor (e.g. goal_feature_names, features_dim).
    mask : array-like
        7-dim constant mask appended to the flat observation.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        inner_extractor_class: type[BaseFeaturesExtractor],
        inner_extractor_kwargs: dict,
        mask: np.ndarray,
    ):
        obs_dim = int(observation_space["observation"].shape[0])
        goal_dim = int(observation_space["desired_goal"].shape[0])
        flat_dim = obs_dim + goal_dim + len(mask)

        flat_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(flat_dim,), dtype=np.float32,
        )
        inner = inner_extractor_class(flat_space, **inner_extractor_kwargs)

        super().__init__(observation_space, features_dim=inner.features_dim)
        self.inner = inner
        self.register_buffer("_mask", th.tensor(mask, dtype=th.float32))
        self._obs_dim = obs_dim
        self._goal_dim = goal_dim

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        obs = observations["observation"]       # (B, 51)
        desired = observations["desired_goal"]  # (B, 7)
        batch_size = obs.shape[0]
        mask = self._mask.unsqueeze(0).expand(batch_size, -1)  # (B, 7)
        flat = th.cat([obs, desired, mask], dim=1)  # (B, 65)
        return self.inner(flat)


class _QNetwork(nn.Module):
    """Single Q-network: features → hidden → Q(s, a) for all discrete actions."""

    def __init__(self, features_dim: int, n_actions: int, net_arch: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        prev = features_dim
        for h in net_arch:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, features: th.Tensor) -> th.Tensor:
        return self.net(features)


class _Actor(nn.Module):
    """Categorical actor: features → hidden → logits for Categorical distribution."""

    def __init__(self, features_dim: int, n_actions: int, net_arch: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        prev = features_dim
        for h in net_arch:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, features: th.Tensor) -> th.Tensor:
        """Return raw logits (batch, n_actions)."""
        return self.net(features)


class DiscreteSACPolicy(BasePolicy):
    """Policy for SAC-Discrete: categorical actor + dual Q-critics.

    Supports both flat ``Box`` and ``Dict`` observation spaces.  When
    ``features_extractor_class`` is ``DictToFlatExtractor``, Dict
    observations are automatically concatenated and forwarded to the
    underlying FiLM extractor.

    Parameters
    ----------
    observation_space, action_space, lr_schedule : standard SB3 signature
    net_arch : list[int]
        Hidden layer sizes for actor and critic heads (after feature extraction).
    features_extractor_class : type[BaseFeaturesExtractor]
        Feature extractor class.
    features_extractor_kwargs : dict
        Extra kwargs for the feature extractor.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule,
        net_arch: Optional[list[int]] = None,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        **kwargs,  # absorb SB3 extras (use_sde, sde_sample_freq, etc.)
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs or {},
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [256, 256]
        self.net_arch = net_arch
        self.n_actions = int(action_space.n)

        # Build feature extractors (separate instances for actor and each critic)
        self.actor_features_extractor = self.make_features_extractor()
        self.critic_features_extractor = self.make_features_extractor()
        self.critic_target_features_extractor = self.make_features_extractor()

        features_dim = self.actor_features_extractor.features_dim

        # Actor
        self.actor = _Actor(features_dim, self.n_actions, self.net_arch)

        # Twin critics
        self.critic_1 = _QNetwork(features_dim, self.n_actions, self.net_arch)
        self.critic_2 = _QNetwork(features_dim, self.n_actions, self.net_arch)

        # Target critics (not optimised directly — polyak-updated)
        self.critic_1_target = deepcopy(self.critic_1)
        self.critic_2_target = deepcopy(self.critic_2)
        self.critic_1_target.requires_grad_(False)
        self.critic_2_target.requires_grad_(False)
        # Target feature extractor
        self.critic_target_features_extractor.load_state_dict(
            self.critic_features_extractor.state_dict()
        )
        self.critic_target_features_extractor.requires_grad_(False)

        # Optimizers — created lazily by DiscreteSAC._setup_model() for
        # lr_schedule support; store the schedule for that purpose.
        self._lr_schedule = lr_schedule
        lr = lr_schedule(1)

        self.actor_optimizer = self.optimizer_class(
            list(self.actor.parameters()) + list(self.actor_features_extractor.parameters()),
            lr=lr,
            **(self.optimizer_kwargs or {}),
        )
        self.critic_optimizer = self.optimizer_class(
            list(self.critic_1.parameters())
            + list(self.critic_2.parameters())
            + list(self.critic_features_extractor.parameters()),
            lr=lr,
            **(self.optimizer_kwargs or {}),
        )

    # ------------------------------------------------------------------
    # Feature extraction helpers
    # ------------------------------------------------------------------
    def _extract(
        self,
        obs: PyTorchObs,
        extractor: BaseFeaturesExtractor,
    ) -> th.Tensor:
        preprocessed = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        return extractor(preprocessed)

    # ------------------------------------------------------------------
    # Forward / predict (used by collect_rollouts → _sample_action)
    # ------------------------------------------------------------------
    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, obs: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        features = self._extract(obs, self.actor_features_extractor)
        logits = self.actor(features)
        if deterministic:
            return th.argmax(logits, dim=1)
        dist = th.distributions.Categorical(logits=logits)
        return dist.sample()

    # ------------------------------------------------------------------
    # Methods called by DiscreteSAC.train()
    # ------------------------------------------------------------------
    def get_action_dist(self, obs: PyTorchObs):
        """Return (probs, log_probs) for all actions, shape (B, n_actions)."""
        features = self._extract(obs, self.actor_features_extractor)
        logits = self.actor(features)
        probs = th.softmax(logits, dim=1)
        # Clamp for numerical stability of log
        log_probs = th.log(probs.clamp(min=1e-8))
        return probs, log_probs

    def q_values(self, obs: PyTorchObs):
        """Return (Q1, Q2) each shape (B, n_actions)."""
        features = self._extract(obs, self.critic_features_extractor)
        return self.critic_1(features), self.critic_2(features)

    def q_values_target(self, obs: PyTorchObs):
        """Return (Q1_target, Q2_target) each shape (B, n_actions)."""
        features = self._extract(obs, self.critic_target_features_extractor)
        return self.critic_1_target(features), self.critic_2_target(features)
