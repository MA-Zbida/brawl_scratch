"""GoalEnv wrappers for SB3 HerReplayBuffer compatibility.

Provides two wrappers that sit on top of StageGoalEnv:

1. ``FlattenMultiDiscreteWrapper`` — encodes MultiDiscrete([4,2,2,4]) to
   Discrete(64) via mixed-radix encoding so the replay buffer stores a
   single integer action.

2. ``StageGoalDictEnv`` — converts the flat 65-dim augmented observation
   into a gymnasium ``Dict`` observation with keys ``observation``,
   ``achieved_goal``, and ``desired_goal``.  Also exposes a vectorised
   ``compute_reward()`` as required by ``HerReplayBuffer``.

Typical wrapping order (innermost first):
    BrawlDeepEnv → StageGoalEnv → FlattenMultiDiscreteWrapper → StageGoalDictEnv
"""

from __future__ import annotations

from functools import reduce
from typing import Any, Optional, Sequence

import gymnasium as gym
import numpy as np

from hierarchical.goals import GOAL_TARGET_DIM, extract_goal_features


# ---------------------------------------------------------------------------
# Action-space flattening: MultiDiscrete([4,2,2,4]) ↔ Discrete(64)
# ---------------------------------------------------------------------------

_NVEC = np.array([4, 2, 2, 4], dtype=np.int64)
_TOTAL_ACTIONS = int(reduce(lambda a, b: a * b, _NVEC))  # 64

# Mixed-radix multipliers: [1, 4, 8, 16]
_MULTIPLIERS = np.ones_like(_NVEC)
for _i in range(1, len(_NVEC)):
    _MULTIPLIERS[_i] = _MULTIPLIERS[_i - 1] * _NVEC[_i - 1]


def encode_action(multi: np.ndarray) -> int:
    """Encode a MultiDiscrete action array to a single Discrete index."""
    return int(np.dot(multi, _MULTIPLIERS))


def decode_action(flat: int) -> np.ndarray:
    """Decode a single Discrete index to a MultiDiscrete action array."""
    out = np.empty(len(_NVEC), dtype=np.int64)
    remainder = int(flat)
    for i in range(len(_NVEC)):
        out[i] = remainder % _NVEC[i]
        remainder //= _NVEC[i]
    return out


class FlattenMultiDiscreteWrapper(gym.ActionWrapper):
    """Flatten ``MultiDiscrete([4,2,2,4])`` → ``Discrete(64)``."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiDiscrete), (
            f"Expected MultiDiscrete, got {type(env.action_space)}"
        )
        self.action_space = gym.spaces.Discrete(_TOTAL_ACTIONS)

    def action(self, action: int | np.integer) -> np.ndarray:
        """Discrete(64) → MultiDiscrete([4,2,2,4])."""
        return decode_action(int(action))


# ---------------------------------------------------------------------------
# Dict observation wrapper for HerReplayBuffer
# ---------------------------------------------------------------------------

class StageGoalDictEnv(gym.Wrapper):
    """Convert flat StageGoalEnv obs to Dict obs for HER.

    Input obs layout (from StageGoalEnv):  [state(B) | goal_target(7) | mask(7)]
    Output Dict obs:
        - ``observation``: state vector (B-dim)
        - ``achieved_goal``: 7-dim goal features of current state
        - ``desired_goal``: 7-dim goal target

    Parameters
    ----------
    env : gym.Env
        Must be a (possibly action-wrapped) StageGoalEnv whose obs is
        flat (base_dim + 2*GOAL_TARGET_DIM).
    proximity_scale : float
        Reward penalty coefficient for distance to goal
        (used in ``compute_reward``).
    success_threshold : float
        Error below which the goal is considered achieved.
    success_bonus : float
        Bonus reward for achieving the goal.
    mask : np.ndarray
        7-dim mask selecting active goal dims.
    """

    def __init__(
        self,
        env: gym.Env,
        proximity_scale: float,
        success_threshold: float,
        success_bonus: float,
        mask: np.ndarray,
    ):
        super().__init__(env)

        flat_dim = int(env.observation_space.shape[0])
        self._base_dim = flat_dim - 2 * GOAL_TARGET_DIM  # 51
        self._goal_dim = GOAL_TARGET_DIM  # 7

        self._mask = np.asarray(mask, dtype=np.float32).reshape(self._goal_dim)
        self._proximity_scale = float(proximity_scale)
        self._success_threshold = float(success_threshold)
        self._success_bonus = float(success_bonus)

        self.observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(self._base_dim,), dtype=np.float32,
                ),
                "achieved_goal": gym.spaces.Box(
                    low=0.0, high=1.0,
                    shape=(self._goal_dim,), dtype=np.float32,
                ),
                "desired_goal": gym.spaces.Box(
                    low=0.0, high=1.0,
                    shape=(self._goal_dim,), dtype=np.float32,
                ),
            }
        )

    def _split_obs(self, flat_obs: np.ndarray) -> dict[str, np.ndarray]:
        state = flat_obs[: self._base_dim].astype(np.float32)
        desired = flat_obs[self._base_dim : self._base_dim + self._goal_dim].astype(np.float32)
        achieved = extract_goal_features(state)
        return {
            "observation": state,
            "achieved_goal": achieved,
            "desired_goal": desired,
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        flat_obs, info = self.env.reset(seed=seed, options=options)
        return self._split_obs(flat_obs), info

    def step(self, action):
        flat_obs, reward, terminated, truncated, info = self.env.step(action)
        return self._split_obs(flat_obs), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # compute_reward: REQUIRED by HerReplayBuffer (vectorised contract)
    # ------------------------------------------------------------------
    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict[str, Any],
    ) -> np.ndarray:
        """Vectorised reward matching StageGoalEnv shaping.

        Parameters
        ----------
        achieved_goal : np.ndarray, shape ``(batch, 7)`` or ``(7,)``
        desired_goal  : np.ndarray, shape ``(batch, 7)`` or ``(7,)``
        info          : dict (unused — mask/scales stored at init)

        Returns
        -------
        np.ndarray, shape ``(batch,)`` or scalar
        """
        achieved = np.asarray(achieved_goal, dtype=np.float32)
        desired = np.asarray(desired_goal, dtype=np.float32)

        # Error per sample: masked L1
        error = np.sum(self._mask * np.abs(achieved - desired), axis=-1)

        # Proximity reward (negative penalty for distance)
        reward = -self._proximity_scale * error

        # Success bonus
        success = error < self._success_threshold
        reward = reward + self._success_bonus * success.astype(np.float32)

        return reward.astype(np.float32)
