"""Goal-conditioned FiLM feature extractor for SB3 PPO policies."""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from feature_extractor.memory.state_spec import StateSpec
from hierarchical.goals import GOAL_DIM


class GoalConditionedModulationExtractor(BaseFeaturesExtractor):
    """Encode augmented state and goal, then apply FiLM modulation.

    Final actor/critic feature h uses:
        h = gamma(g) * phi([s, goal_dx, goal_dy, goal_distance, goal_v_alignment]) + beta(g)
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        obs_dim = int(observation_space.shape[0])
        if obs_dim <= GOAL_DIM:
            raise ValueError(f"Observation dim {obs_dim} must be > GOAL_DIM={GOAL_DIM}")

        self.state_dim = obs_dim - GOAL_DIM
        self.goal_dim = GOAL_DIM
        self.goal_xy_dim = 2
        self.aug_extra_dim = 4

        self.idx_player_x = StateSpec.index("player_x")
        self.idx_player_y = StateSpec.index("player_y")
        self.idx_player_vx = StateSpec.index("player_vx")
        self.idx_player_vy = StateSpec.index("player_vy")

        aug_state_dim = self.state_dim + self.aug_extra_dim

        self.state_encoder = nn.Sequential(
            nn.Linear(aug_state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.goal_encoder = nn.Sequential(
            nn.Linear(self.goal_xy_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.film_generator = nn.Linear(128, 512)
        self.post_film = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        state = observations[:, : self.state_dim]
        goal_full = observations[:, self.state_dim :]
        goal_xy = goal_full[:, : self.goal_xy_dim]

        player_x = state[:, self.idx_player_x]
        player_y = state[:, self.idx_player_y]
        player_vx = state[:, self.idx_player_vx]
        player_vy = state[:, self.idx_player_vy]

        goal_dx = goal_xy[:, 0] - player_x
        goal_dy = goal_xy[:, 1] - player_y
        goal_distance = torch.sqrt(goal_dx.square() + goal_dy.square() + 1e-8)
        goal_v_alignment = player_vx * goal_dx + player_vy * goal_dy

        state_aug = torch.cat(
            [
                state,
                goal_dx.unsqueeze(1),
                goal_dy.unsqueeze(1),
                goal_distance.unsqueeze(1),
                goal_v_alignment.unsqueeze(1),
            ],
            dim=1,
        )

        phi_s = self.state_encoder(state_aug)
        goal_latent = self.goal_encoder(goal_xy)
        gamma, beta = self.film_generator(goal_latent).chunk(2, dim=1)
        h = gamma * phi_s + beta
        return self.post_film(h)
