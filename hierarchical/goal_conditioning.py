"""Goal-conditioned modulation feature extractor for SB3 PPO policies."""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from hierarchical.goals import GOAL_DIM


class GoalConditionedModulationExtractor(BaseFeaturesExtractor):
    """Encode state and goal separately, then multiplicatively modulate features."""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        obs_dim = int(observation_space.shape[0])
        if obs_dim <= GOAL_DIM:
            raise ValueError(f"Observation dim {obs_dim} must be > GOAL_DIM={GOAL_DIM}")

        self.state_dim = obs_dim - GOAL_DIM
        self.goal_dim = GOAL_DIM

        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )
        self.goal_encoder = nn.Sequential(
            nn.Linear(self.goal_dim, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.Sigmoid(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        state = observations[:, : self.state_dim]
        goal = observations[:, self.state_dim :]
        state_features = self.state_encoder(state)
        modulation = self.goal_encoder(goal)
        return state_features * modulation
