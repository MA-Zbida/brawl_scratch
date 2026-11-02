"""Checkpoint, evaluate, and refresh the live state between PPO rollouts."""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from brawl.training.checkpoints import save_typed_checkpoint


class FinalCampaign(BaseCallback):
    def __init__(self, cfg, match_env, directory: Path, metadata: dict):
        super().__init__()
        self.cfg = cfg
        self.match_env = match_env
        self.directory = directory
        self.metadata = metadata
        self.next_save = cfg.checkpoint_every
        self.next_eval = cfg.eval_every
        self.best = None
        self.baseline_done = False
        self._gap_started = None
        self.gap_seconds = 0.

    def save(self, name: str):
        return save_typed_checkpoint(
            self.model, self.directory / f"{name}.zip",
            self.metadata | dict(
                training_stage="final_agent", objective="stock_exchanges" if
                self.cfg.mode == "training_room" else "match",
                mode=self.cfg.mode, environment_steps=int(self.model.num_timesteps),
                recorded_interactions=self.match_env.total_interactions,
                optimizer_gap_seconds=self.gap_seconds))

    def _refresh(self, *, boundary=False):
        # Released inputs during optimizer downtime form a neutral bridge, not
        # an effect of the next PPO action. Keep it in replay and outside PPO.
        self.match_env.usage = "update_gap"
        self.match_env.force_boundary = boundary
        obs, _, dones, _ = self.training_env.step(np.array([0]))
        self.model._last_obs = obs
        self.model._last_episode_starts = dones.copy()
        self.match_env.usage = "train"
        if boundary or dones[0]:
            self.match_env._window_usage = "train"

    def evaluate(self, episodes: int):
        self._refresh(boundary=True)
        self.match_env.usage = self.match_env._window_usage = "evaluation"
        summaries = []
        obs = self.model._last_obs
        while len(summaries) < episodes:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, dones, _ = self.training_env.step(action)
            if dones[0]:
                summaries.append(dict(self.match_env.last_summary))
        self.match_env.release_inputs()
        self.model._last_obs = obs
        self.model._last_episode_starts = dones.copy()
        self.match_env.usage = self.match_env._window_usage = "train"
        seconds = sum(s["seconds"] for s in summaries)
        stock_diff = sum(s["stock_differential"] for s in summaries)
        damage_trade = sum(s["damage_trade"] for s in summaries)
        row = dict(
            training_step=int(self.model.num_timesteps), mode=self.cfg.mode,
            windows=len(summaries), seconds=seconds,
            stocks_taken=sum(s["stocks_taken"] for s in summaries),
            stocks_lost=sum(s["stocks_lost"] for s in summaries),
            stock_differential=stock_diff,
            net_stocks_per_minute=60 * stock_diff / max(seconds, 1e-6),
            damage_trade=damage_trade,
            wins=sum(s["outcome"] == "won" for s in summaries),
            losses=sum(s["outcome"] == "lost" for s in summaries),
        )
        path = self.directory / "evaluations.csv"
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        key = ((row["net_stocks_per_minute"], damage_trade / max(seconds, 1e-6))
               if self.cfg.mode == "training_room"
               else (row["wins"] / episodes, row["net_stocks_per_minute"]))
        if self.best is None or key > self.best:
            self.best = key
            self.save("best")
        print(f"[final eval @{self.model.num_timesteps}] {self.cfg.mode}: "
              f"stocks {row['stocks_taken']:.0f}-{row['stocks_lost']:.0f}, "
              f"net/min={row['net_stocks_per_minute']:+.3f}, trade={damage_trade:+.1f}")

    def _on_rollout_start(self):
        if self._gap_started is not None:
            self.gap_seconds += time.perf_counter() - self._gap_started
            self._gap_started = None
            self._refresh()
        if not self.baseline_done:
            self.save("initializer" if not self.cfg.resume else "resume_start")
            if self.cfg.eval_every:
                self.evaluate(min(3, self.cfg.eval_episodes))
            self.baseline_done = True
            self.next_save = self.model.num_timesteps + self.cfg.checkpoint_every
            self.next_eval = self.model.num_timesteps + self.cfg.eval_every
        elif self.cfg.eval_every and self.model.num_timesteps >= self.next_eval:
            self.evaluate(self.cfg.eval_episodes)
            self.next_eval = self.model.num_timesteps + self.cfg.eval_every
        if self.model.num_timesteps >= self.next_save:
            self.save(f"step_{self.model.num_timesteps:09d}")
            self.save("latest")
            self.next_save = self.model.num_timesteps + self.cfg.checkpoint_every

        self.match_env.release_inputs()
        self._refresh()

    def _on_rollout_end(self):
        self.match_env.release_inputs()
        self._gap_started = time.perf_counter()

    def _on_step(self):
        self.match_env.policy_version = int(self.model.num_timesteps // self.cfg.n_steps)
        return True
