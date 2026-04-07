"""Simple MLP world model: predicts delta(s_{t+1} - s_t) from (s_t, a_t).

Architecture kept minimal and MBRL-ready: the forward() method returns predicted
next-state so it can later serve as a differentiable dynamics model inside a
model-based planner or Dyna-style loop.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

from feature_extractor.memory.state_spec import StateSpec

STATE_DIM = StateSpec.dim()       # 51
ACTION_DIMS = [4, 2, 2, 4]       # MultiDiscrete([4,2,2,4])
ACTION_OH_DIM = sum(ACTION_DIMS)  # 12
INPUT_DIM = STATE_DIM + ACTION_OH_DIM  # 63


def encode_actions(actions: np.ndarray) -> np.ndarray:
    """One-hot encode MultiDiscrete actions.

    Parameters
    ----------
    actions : (N, 4) int array with columns [move, jump, dodge, attack]

    Returns
    -------
    (N, 12) float32 one-hot encoded actions
    """
    n = actions.shape[0]
    oh = np.zeros((n, ACTION_OH_DIM), dtype=np.float32)
    offset = 0
    for col, dim in enumerate(ACTION_DIMS):
        oh[np.arange(n), offset + actions[:, col]] = 1.0
        offset += dim
    return oh


class WorldModel(nn.Module):
    """MLP that predicts state delta: s_{t+1} = s_t + model(s_t, one_hot(a_t))."""

    def __init__(self, hidden: int = 256, n_layers: int = 3, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = INPUT_DIM
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.LayerNorm(hidden))
            layers.append(nn.SiLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden
        layers.append(nn.Linear(hidden, STATE_DIM))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action_oh: torch.Tensor) -> torch.Tensor:
        """Predict next state.

        Parameters
        ----------
        state     : (B, 51) current state
        action_oh : (B, 12) one-hot encoded action

        Returns
        -------
        (B, 51) predicted next state  (s_t + predicted delta)
        """
        x = torch.cat([state, action_oh], dim=-1)
        delta = self.net(x)
        return state + delta

    def predict_delta(self, state: torch.Tensor, action_oh: torch.Tensor) -> torch.Tensor:
        """Return raw predicted delta (useful for loss computation)."""
        x = torch.cat([state, action_oh], dim=-1)
        return self.net(x)

    def predict_np(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Numpy convenience: (N,51) states + (N,4) int actions → (N,51) next states."""
        device = next(self.parameters()).device
        s = torch.as_tensor(states, dtype=torch.float32, device=device)
        a_oh = torch.as_tensor(encode_actions(actions), dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = self.forward(s, a_oh)
        return pred.cpu().numpy()
