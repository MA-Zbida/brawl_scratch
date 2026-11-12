"""BrawlEnv: the one step function (research/DESIGN.md section 4).

Each step: the action becomes keys for ``k`` game frames, the clock plays them, one frame is
captured and perceived once, and that one Sense gives the observation and the reward. The task
continues without end: ``terminated`` is never true, and ``truncated`` only marks the end of a
window so that learners keep bootstrapping.
"""
from __future__ import annotations

import time
from typing import Any

import gymnasium as gym
import numpy as np

from brawl import actions, observation
from brawl.ledger import reward as reward_of


class BrawlEnv(gym.Env):
    def __init__(self, clock, screen, perception, *, k: int = 3, window: int = 1200, damage_weight: float = 0.01,
                 ko_weight: float = 1.0, now=time.perf_counter) -> None:
        """``clock``: brawl.game.clock (stepped or realtime), already started. ``screen``:
        brawl.game.capture.Screen. ``perception``: brawl.sense.perception.Perception built with
        the same ``k``. ``window``: decisions per window (1,200 is one minute at k = 3)."""
        self.clock, self.screen, self.perception, self.k, self.window = clock, screen, perception, k, window
        self.weights, self.now = (damage_weight, ko_weight), now
        self.builder = observation.Observation()
        self.observation_space = gym.spaces.Box(-5.0, 5.0, (observation.SIZE,), np.float32)
        self.action_space = gym.spaces.MultiDiscrete(list(actions.SHAPE))
        self.steps = 0
        self.calls = {"advance": 0, "grab": 0, "perceive": 0, "build": 0}  # gate L1: each once per step
        self._seen = False

    def _observe(self) -> tuple[np.ndarray, Any, float]:
        t0 = self.now()
        frame = self.screen.grab(fresh=False)
        sense = self.perception(frame.image, self.clock.frame)
        obs = self.builder.build(sense)
        self.calls["grab"] += 1
        self.calls["perceive"] += 1
        self.calls["build"] += 1
        return obs, sense, self.now() - t0

    def reset(self, *, seed=None, options=None):
        """Start a window where the game is: the game itself is never reset."""
        super().reset(seed=seed)
        if self._seen:  # a new window continues from the last observation's state
            obs, sense, _ = self._observe()
        else:
            self.perception.reset()
            self.builder.reset()
            obs, sense, _ = self._observe()
            self._seen = True
        return obs, {"sense": sense}

    def step(self, action):
        action = (int(action[0]), int(action[1]))
        timing = self.clock.advance(actions.to_plan(action, self.k))
        self.calls["advance"] += 1
        self.builder.acted(action)
        obs, sense, sense_s = self._observe()
        self.steps += 1
        e = sense.events
        info = {"events": e, "sense": sense, "timing": timing, "sense_s": sense_s, "frame": self.clock.frame}
        return obs, reward_of(e, *self.weights), False, self.steps % self.window == 0, info

    def close(self) -> None:
        self.clock.stop()
