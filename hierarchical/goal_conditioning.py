"""Goal-conditioned FiLM feature extractor for structured masked goals."""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from feature_extractor.memory.state_spec import StateSpec
from hierarchical.goals import GOAL_DIM, GOAL_TARGET_DIM


class GoalConditionedModulationExtractor(BaseFeaturesExtractor):
    """Encode state and goal structure, then apply FiLM modulation.

    Observation layout expected by this extractor:
        [state, goal_target(k), goal_mask(k)]

    State encoder input is structured as:
        s_aug = concat(state, masked_error, goal_mask, goal_alignment)

    where:
        goal_error = f(state) - goal_target
        masked_error = goal_mask * goal_error
        goal_alignment = goal_mask * sign(goal_error) * ||v_player||
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        obs_dim = int(observation_space.shape[0])
        if obs_dim <= GOAL_DIM:
            raise ValueError(f"Observation dim {obs_dim} must be > GOAL_DIM={GOAL_DIM}")

        self.state_dim = obs_dim - GOAL_DIM
        self.goal_dim = GOAL_DIM
        self.goal_target_dim = GOAL_TARGET_DIM

        self.idx_player_vx = StateSpec.index("player_vx")
        self.idx_player_vy = StateSpec.index("player_vy")

        self.goal_feature_indices = [
            StateSpec.index("dist_to_stage_center"),
            StateSpec.index("dist_to_nearest_ledge"),
            StateSpec.index("in_strike_range"),
            StateSpec.index("player_grounded"),
            StateSpec.index("player_is_offstage"),
            StateSpec.index("rel_distance"),
            StateSpec.index("frame_advantage_estimate"),
        ]

        if len(self.goal_feature_indices) != self.goal_target_dim:
            raise ValueError(
                f"Goal feature index count ({len(self.goal_feature_indices)}) must match GOAL_TARGET_DIM ({self.goal_target_dim})"
            )

        self.register_buffer("feat_min", torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=torch.float32))
        self.register_buffer("feat_max", torch.tensor([2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0], dtype=torch.float32))

        aug_state_dim = self.state_dim + (3 * self.goal_target_dim)

        self.state_encoder = nn.Sequential(
            nn.Linear(aug_state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.goal_encoder = nn.Sequential(
            nn.Linear(self.goal_dim, 128),
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

    def _normalize_goal_features(self, raw_feats: torch.Tensor) -> torch.Tensor:
        denom = (self.feat_max - self.feat_min).clamp_min(1e-6)
        z = (raw_feats - self.feat_min) / denom
        return z.clamp(0.0, 1.0)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        state = observations[:, : self.state_dim]
        goal_full = observations[:, self.state_dim :]
        goal_target = goal_full[:, : self.goal_target_dim]
        goal_mask = goal_full[:, self.goal_target_dim :]

        raw_goal_feats = state[:, self.goal_feature_indices]
        goal_feats = self._normalize_goal_features(raw_goal_feats)

        player_vx = state[:, self.idx_player_vx]
        player_vy = state[:, self.idx_player_vy]

        goal_error = goal_feats - goal_target
        masked_error = goal_mask * goal_error

        speed = torch.tanh(torch.sqrt(player_vx.square() + player_vy.square() + 1e-8)).unsqueeze(1)
        goal_alignment = goal_mask * torch.sign(goal_error) * speed

        state_aug = torch.cat([state, masked_error, goal_mask, goal_alignment], dim=1)

        phi_s = self.state_encoder(state_aug)
        goal_latent = self.goal_encoder(goal_full)
        gamma, beta = self.film_generator(goal_latent).chunk(2, dim=1)

        h = gamma * phi_s + beta
        h = h + phi_s
        return self.post_film(h)
