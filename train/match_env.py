"""Match accounting without curriculum goals or automatic menu key presses."""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from feature_extractor.memory.state_spec import StateSpec
from reward.match_objective import MatchObjective


class MatchTrainingEnv(gym.Wrapper):
    def __init__(self, env, *, mode="manual_matches", max_steps=3000,
                 matches_path: Path | None = None, steps_path: Path | None = None):
        super().__init__(env)
        self.mode = mode
        self.max_steps = max_steps
        self.objective = MatchObjective(match_bonus=mode == "manual_matches")
        self.matches_path = matches_path
        self.steps_path = steps_path
        self.usage = "train"
        self._last = None
        self._needs_restart = False
        self._first = True
        self._started = time.perf_counter()
        self._step_time = self._started
        self.total_interactions = 0
        self.last_summary = None
        self.force_boundary = False
        self.policy_version = 0
        self._handles: dict[Path, Any] = {}
        self._rows_written = 0

    def _write(self, path, row):
        if path is None:
            return
        path = Path(path)
        if path not in self._handles:
            path.parent.mkdir(parents=True, exist_ok=True)
            exists = path.exists() and path.stat().st_size > 0
            if exists:
                with path.open(encoding="utf-8", newline="") as stream:
                    if next(csv.reader(stream)) != list(row):
                        raise ValueError(f"Refusing incompatible CSV append: {path}")
            handle = path.open("a", newline="", encoding="utf-8")
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            self._handles[path] = handle, writer
        handle, writer = self._handles[path]
        writer.writerow(row)
        self._rows_written += 1
        if path == self.matches_path or self._rows_written % 128 == 0:
            handle.flush()

    def release_inputs(self):
        self.unwrapped.input_controller.reset()

    def _rearm_stocks(self):
        # The training arena has unlimited lives, but the pixel provider counts
        # down from three. Re-arm counters, not temporal state or cooldown locks.
        base = self.unwrapped
        if not hasattr(base, "memory"):
            return
        for name in ("self_stocks_left", "op_stocks_left",
                     "prev_self_stocks_left", "prev_op_stocks_left"):
            setattr(base.memory, name, 3.)
        provider = base.stocks_health_provider
        for name in ("self_stocks_left", "op_stocks_left"):
            if provider is not None:
                setattr(provider, name, 3.)
        base.memory.player.stocks = base.memory.opponent.stocks = 3.

    def reset(self, *, seed=None, options=None):
        if self._first or self._needs_restart:
            if self._needs_restart:
                self.release_inputs()
                input("Match ended. Start a NEW 3-stock match, then press Enter here.")
                time.sleep(3)
                self._rearm_stocks()
                provider = self.unwrapped.stocks_health_provider
                if provider is not None:
                    provider.reset(preserve_match_state=False)
            self._last = self.env.reset(seed=seed, options=options)
            self.objective.reset()
        # A reporting time window is not a game reset. Preserve history, damage
        # budgets, and the live stock exchange across these truncations.
        self._first = self._needs_restart = False
        self._count = 0
        self._window_start = time.perf_counter()
        self._window_usage = self.usage
        self._sums = dict(reward=0., damage_dealt=0., damage_received=0.,
                          stocks_taken=0., stocks_lost=0.)
        return self._last[0].copy(), dict(self._last[1])

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        memory = getattr(self.unwrapped, "memory", None)
        if memory is not None:
            info["damage_valid"] = bool(
                memory.player.exists and memory.opponent.exists
                and memory.player_respawn_timer <= 0
                and memory.opponent_respawn_timer <= 0)
        terms = self.objective.step(info)
        info.update({f"match_reward_{key}": value for key, value in terms.items()})
        reward = terms["total_reward"]
        self._count += 1
        self.total_interactions += 1
        now = time.perf_counter()
        info.update(reward_breakdown=terms, transition_usage=self.usage,
                    update_gap=float(self.usage == "update_gap"),
                    evaluation=float(self.usage == "evaluation"),
                    policy_version=self.policy_version,
                    observation_interval_s=now - self._step_time)
        self._step_time = now
        self._sums["reward"] += reward
        for target, source in (
            ("damage_dealt", "op_delta_damage"), ("damage_received", "self_delta_damage"),
            ("stocks_taken", "op_stock_lost_step"), ("stocks_lost", "self_stock_lost_step")):
            self._sums[target] += float(info.get(source, 0))
        won = float(info.get("op_stocks_left", 3)) <= 0
        lost = float(info.get("self_stocks_left", 3)) <= 0
        if self.mode == "training_room":
            terminated = False
            self._rearm_stocks()
            obs = np.array(obs, copy=True)
            # Remaining match lives are always full in unlimited-respawn mode.
            if obs.size >= StateSpec.dim():
                for name in ("self_stocks_norm", "op_stocks_norm"):
                    obs[StateSpec.index(name)] = 1.
            if hasattr(self.unwrapped, "_last_obs"):
                self.unwrapped._last_obs = obs.copy()
            self._needs_restart = False
        else:
            terminated = bool(terminated or won or lost)
            self._needs_restart = bool(terminated)
        truncated = bool((truncated or self.force_boundary or self._count >= self.max_steps) and not terminated)
        self.force_boundary = False
        outcome = ("draw" if won and lost else "won" if won else "lost") if terminated else "timeout"
        info["match_outcome"] = outcome if (terminated or truncated) else "ongoing"
        self._write(self.steps_path, dict(
            interaction=self.total_interactions, usage=self.usage,
            wall_seconds=now - self._started, observation_interval_s=info["observation_interval_s"],
            action_requested=int(np.asarray(action).item()),
            action_executed=int(info.get("effective_action", np.asarray(action).item())),
            terminated=int(terminated), truncated=int(truncated),
            **terms,
            damage_dealt=float(info.get("op_delta_damage", 0)),
            damage_received=float(info.get("self_delta_damage", 0)),
            stocks_taken=float(info.get("op_stock_lost_step", 0)),
            stocks_lost=float(info.get("self_stock_lost_step", 0)),
        ))
        if terminated or truncated:
            self.last_summary = dict(
                mode=self.mode, usage=self._window_usage, outcome=outcome,
                interactions=self._count, seconds=now-self._window_start,
                **self._sums,
                stock_differential=self._sums["stocks_taken"]-self._sums["stocks_lost"],
                damage_trade=self._sums["damage_dealt"]-self._sums["damage_received"])
            self._write(self.matches_path, self.last_summary)
        self._last = obs.copy(), info.copy()
        return obs, reward, terminated, truncated, info

    def close(self):
        try:
            self.release_inputs()
            self.env.close()
        finally:
            for handle, _ in self._handles.values():
                handle.close()
            self._handles.clear()


class ReplayGymWrapper(gym.Wrapper):
    """Adapt the lightweight replay recorder to SB3's Gymnasium contract."""
    def __init__(self, env, writer):
        super().__init__(env)
        from world_model.replay_log import WorldReplayRecorder
        self.recorder = WorldReplayRecorder(env, writer)

    def step(self, action):
        return self.recorder.step(action)

    def reset(self, **kwargs):
        return self.recorder.reset(**kwargs)

    def close(self):
        return self.recorder.close()
