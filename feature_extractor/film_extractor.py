"""Variable-dimension FiLM feature extractor for stage LLC training.

Observation layout expected:
    [state(base_dim) | goal_target_norm(G) | mask(G)]

where G = len(feature_names), goal_target_norm = target / feature_scale,
and base_dim = StateSpec.dim() = 51.

The extractor:
  1. Extracts the G goal features from the raw state and normalises them.
  2. Computes goal_error = norm_feat - goal_target_norm   (aligned scale).
  3. Computes masked_error and goal_alignment (velocity-weighted directional cue).
  4. Encodes augmented state s_aug = [state, masked_error, mask, goal_alignment].
  5. Encodes goal vector and applies FiLM modulation with residual.

This mirrors GoalConditionedModulationExtractor (hierarchical/goal_conditioning.py)
but is parametrised by stage-specific feature names rather than the fixed 7-dim
combat goal, so it composes across all LLC stages.
"""

from __future__ import annotations

from typing import Sequence

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from feature_extractor.memory.state_spec import StateSpec
from train.llc_stage_common import FEATURE_SCALE

# Per-feature normalisation bounds used to bring extracted state features and
# the stored goal targets into the same [0, 1] (or [-1, 1] for signed) range.
_FEAT_BOUNDS: dict[str, tuple[float, float]] = {
    "player_x": (0.0, 1.0),
    "player_y": (0.0, 1.0),
    "player_vx": (-3.0, 3.0),
    "player_vy": (-3.0, 3.0),
    "player_grounded": (0.0, 1.0),
    "player_damage_pct": (0.0, 1.0),
    "player_jumps_norm": (0.0, 1.0),
    "player_is_offstage": (0.0, 1.0),
    "dist_to_stage_center": (0.0, 2.0),
    "dist_to_nearest_ledge": (0.0, 2.0),
    "dy_to_ledge": (-1.0, 1.0),
    "rel_dx": (-1.0, 1.0),
    "rel_dy": (-1.0, 1.0),
    "rel_distance": (0.0, 2.0),
    "in_strike_range": (0.0, 1.0),
    "opponent_hitstun": (0.0, 1.0),
    "frame_advantage_estimate": (-1.0, 1.0),
}

# Fall-back for any feature not listed: use FEATURE_SCALE as the max, min = 0.
def _get_bounds(name: str) -> tuple[float, float]:
    if name in _FEAT_BOUNDS:
        return _FEAT_BOUNDS[name]
    scale = FEATURE_SCALE.get(name, 1.0)
    return (0.0, scale)


class StageGoalFiLMExtractor(BaseFeaturesExtractor):
    """FiLM-modulated goal-conditioned extractor, parametrised per stage.

    Parameters
    ----------
    observation_space : gym.spaces.Box
        Must have shape (base_dim + 2*G,) where G = len(goal_feature_names).
    goal_feature_names : list[str]
        Ordered list of StateSpec feature names that correspond to the G goal
        dimensions.  Must match the names used in StageSpec.feature_names.
    features_dim : int
        Output dimension of this extractor (fed into policy/value head).
    state_hidden : int
        Hidden size for the state encoder MLP.
    goal_hidden : int
        Hidden size for the goal encoder MLP.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        goal_feature_names: Sequence[str],
        features_dim: int = 256,
        state_hidden: int = 256,
        goal_hidden: int = 128,
    ):
        super().__init__(observation_space, features_dim)

        self.goal_feature_names = list(goal_feature_names)
        self.G = len(self.goal_feature_names)
        self.base_dim = int(observation_space.shape[0]) - 2 * self.G

        if self.base_dim <= 0:
            raise ValueError(
                f"Observation dim {observation_space.shape[0]} is too small for "
                f"G={self.G} goal dims (need at least 2*G+1)."
            )

        # Indices into the raw state for the G goal features.
        self.goal_feat_indices = [StateSpec.index(n) for n in self.goal_feature_names]

        # Velocity indices for goal-alignment signal.
        self.idx_vx = StateSpec.index("player_vx")
        self.idx_vy = StateSpec.index("player_vy")

        # Per-feature normalisation bounds (as buffers so they move to GPU if needed).
        feat_min, feat_max = zip(*[_get_bounds(n) for n in self.goal_feature_names])
        self.feat_min: torch.Tensor
        self.feat_max: torch.Tensor
        self.register_buffer("feat_min", torch.tensor(feat_min, dtype=torch.float32))
        self.register_buffer("feat_max", torch.tensor(feat_max, dtype=torch.float32))

        # s_aug = [state, masked_error(G), mask(G), goal_alignment(G)]
        aug_state_dim = self.base_dim + 3 * self.G

        self.state_encoder = nn.Sequential(
            nn.Linear(aug_state_dim, state_hidden),
            nn.ReLU(),
            nn.Linear(state_hidden, state_hidden),
            nn.ReLU(),
        )

        # goal_vec = [goal_target_norm(G) | mask(G)] = 2*G dims
        goal_vec_dim = 2 * self.G
        self.goal_encoder = nn.Sequential(
            nn.Linear(goal_vec_dim, goal_hidden),
            nn.ReLU(),
            nn.Linear(goal_hidden, goal_hidden),
            nn.ReLU(),
        )

        # FiLM: produce gamma and beta of size state_hidden each.
        self.film_generator = nn.Linear(goal_hidden, 2 * state_hidden)

        self.post_film = nn.Sequential(
            nn.ReLU(),
            nn.Linear(state_hidden, features_dim),
            nn.ReLU(),
        )

    def _normalize_feats(self, raw: torch.Tensor) -> torch.Tensor:
        """Normalise G raw state features to [0, 1] using per-feature bounds."""
        denom = (self.feat_max - self.feat_min).clamp_min(1e-6)
        return ((raw - self.feat_min) / denom).clamp(0.0, 1.0)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Split observation into components.
        state = observations[:, : self.base_dim]                           # (B, base_dim)
        goal_target_norm = observations[:, self.base_dim : self.base_dim + self.G]  # (B, G)
        goal_mask = observations[:, self.base_dim + self.G :]              # (B, G)

        # Extract and normalise the G goal-relevant features from the raw state.
        raw_goal_feats = state[:, self.goal_feat_indices]   # (B, G)
        norm_goal_feats = self._normalize_feats(raw_goal_feats)

        # Goal error: normalised state feature minus target (both in [0,1] space).
        goal_error = norm_goal_feats - goal_target_norm      # (B, G)
        masked_error = goal_mask * goal_error                # (B, G)

        # Goal-alignment: direction of error weighted by player speed.
        player_vx = state[:, self.idx_vx]
        player_vy = state[:, self.idx_vy]
        speed = torch.tanh(
            torch.sqrt(player_vx.square() + player_vy.square() + 1e-8)
        ).unsqueeze(1)                                       # (B, 1)
        goal_alignment = goal_mask * torch.sign(goal_error) * speed  # (B, G)

        # Augmented state with goal context fused in.
        state_aug = torch.cat([state, masked_error, goal_mask, goal_alignment], dim=1)

        # Encode state and goal.
        phi_s = self.state_encoder(state_aug)                # (B, state_hidden)
        goal_vec = torch.cat([goal_target_norm, goal_mask], dim=1)
        goal_latent = self.goal_encoder(goal_vec)            # (B, goal_hidden)

        # FiLM modulation with residual.
        gamma, beta = self.film_generator(goal_latent).chunk(2, dim=1)
        h = gamma * phi_s + beta + phi_s                     # residual
        return self.post_film(h)                             # (B, features_dim)
