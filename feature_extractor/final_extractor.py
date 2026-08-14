"""Goal-independent representation used by the final combat policy.

The skill policy learns a state representation inside a FiLM extractor.  The
final policy keeps that representation boundary but removes every external goal
input.  Keeping ``encode_representation`` separate from ``adapt_representation``
also leaves a narrow seam where a future belief model can replace the flat
state encoder without changing the actor and critic heads.
"""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class FinalStateExtractor(BaseFeaturesExtractor):
    """Encode the canonical state/history vector without a semantic goal."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
        state_hidden: int = 256,
    ) -> None:
        super().__init__(observation_space, features_dim)
        shape = observation_space.shape
        if shape is None or len(shape) != 1 or int(shape[0]) <= 0:
            raise ValueError("FinalStateExtractor requires a non-empty 1D observation")

        self.input_dim = int(shape[0])
        self.state_hidden = int(state_hidden)
        self.state_encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.state_hidden),
            nn.ReLU(),
            nn.Linear(self.state_hidden, self.state_hidden),
            nn.ReLU(),
        )
        # This trainable adapter occupies the old post-FiLM boundary.  It can be
        # initialised from skill pretraining while remaining independent of any
        # target, mask, goal encoder, or modulation parameters.
        self.adapter = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.state_hidden, features_dim),
            nn.ReLU(),
        )

    def encode_representation(self, observations: torch.Tensor) -> torch.Tensor:
        """Return the state representation consumed by the policy adapter."""
        return self.state_encoder(observations)

    def adapt_representation(self, representation: torch.Tensor) -> torch.Tensor:
        """Map a state or future belief representation into actor features."""
        return self.adapter(representation)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.adapt_representation(self.encode_representation(observations))
