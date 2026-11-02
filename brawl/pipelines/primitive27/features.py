"""Fixed geometry preprocessing, serialized with the primitive27 policy."""
from __future__ import annotations

# Live entry points acquire capture before importing torch or this module.
# Offline BC must not acquire a display duplicator as an import side effect.
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from brawl.state.state_spec import StateSpec


class PrimitiveGeometry(BaseFeaturesExtractor):
    """Keep raw observations unchanged; expose small weapon geometry to the MLP.

    The 0.05-screen-distance interaction neighbourhood becomes order one. Its
    radius is derived from existing offsets, not a new sensor or action rule.
    This fixed transform needs no validation statistics and remains identical
    during BC, PPO and deployment (including checkpoint reload).
    """

    def __init__(self, observation_space):
        super().__init__(observation_space, observation_space.shape[0] + 1)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        dx = observations[:, StateSpec.index('weapon_dx')]
        dy = observations[:, StateSpec.index('weapon_dy')]
        features = observations.clone()
        features[:, StateSpec.index('weapon_dx')] = dx * 20.
        features[:, StateSpec.index('weapon_dy')] = dy * 20.
        return th.cat((features, th.hypot(dx, dy).unsqueeze(1) * 20.), dim=1)
