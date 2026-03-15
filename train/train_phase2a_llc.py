#!/usr/bin/env python
"""Train LLC with continuous goals (new hierarchical design).

Stage 1 focuses more on movement/recovery (higher recovery sampling).
Stage 2 increases combat/engagement pressure.

This script saves a PNG dashboard during training with:
- reward per sampled goal segment
- episode coverage per sampled goal
- episode reward / goal error / reward-component trends
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env import BrawlDeepEnv, EnvConfig
from debug.target_cursor import CursorBounds, TargetCursor
from feature_extractor.memory.state_spec import StateSpec
from hierarchical.goal_conditioning import GoalConditionedModulationExtractor
from hierarchical.goals import GoalSampler
from hierarchical.llc_env import LLCEnv

# Platform geometry (normalised)
PLATFORM_X_MIN = 0.315
PLATFORM_X_MAX = 0.683
PLATFORM_Y_MIN = 0.5527
PLATFORM_Y_MAX = 0.8149
MID_X = (PLATFORM_X_MAX + PLATFORM_X_MIN) / 2.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LLC with continuous goals")
    p.add_argument("--stage", type=int, choices=[1, 2], default=1)
    p.add_argument("--timesteps", type=int, default=400_000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.15)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--max-episode-steps", type=int, default=1200)
    p.add_argument("--save-dir", type=str, default="train/models")
    p.add_argument("--model-name", type=str, default="llc")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--plot-every", type=int, default=5)
    p.add_argument("--moving-avg", type=int, default=30)
    p.add_argument("--top-goals", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--delay", type=float, default=3.0)
    p.add_argument("--disable-target-cursor", action="store_true",
                   help="Disable moving mouse cursor to target_x,target_y")
    p.add_argument("--cursor-update-every", type=int, default=1,
                   help="Move cursor once every N env steps")
    p.add_argument("--cursor-left", type=int, default=0)
    p.add_argument("--cursor-top", type=int, default=0)
    p.add_argument("--cursor-right", type=int, default=0,
                   help="0 means full screen width")
    p.add_argument("--cursor-bottom", type=int, default=0,
                   help="0 means full screen height")
    return p.parse_args()


def make_llc_env(stage: int, max_episode_steps: int):
    class Stage1MixedGoalSampler:
        def __init__(self, seed: int | None = None):
            self.rng = np.random.default_rng(seed)
            self.last_lane = "navigation"

        def sample(self, obs: np.ndarray) -> np.ndarray:
            # Stage-1 mixed curriculum with XY targets:
            # goal = [target_x, target_y, _, _, _, _]
            # - navigation lane: center/ledge locomotion
            # - combat_proxy lane: target around opponent (if visible)
            op_exists = StateSpec.get(obs, "opponent_exists") > 0.5

            if self.rng.random() < 0.30 and op_exists:
                self.last_lane = "combat_proxy"
                op_x = StateSpec.get(obs, "opponent_x")
                op_y = StateSpec.get(obs, "opponent_y")
                target_x = np.clip(op_x + self.rng.normal(0.0, 0.05), 0.05, 0.95)
                target_y = np.clip(op_y + self.rng.normal(0.0, 0.05), 0.35, 0.90)
            else:
                self.last_lane = "navigation"
                if self.rng.random() < 0.60:
                    # stay on platform (between ledges, below Y_MIN since Y is inverted)
                    target_x = np.clip(MID_X + self.rng.normal(0.0, 0.06), PLATFORM_X_MIN + 0.02, PLATFORM_X_MAX - 0.02)
                    target_y = np.clip(PLATFORM_Y_MIN - abs(self.rng.normal(0.04, 0.02)), 0.0, PLATFORM_Y_MIN - 0.01)
                else:
                    # ledge mobility
                    side = -1.0 if self.rng.random() < 0.5 else 1.0

                    if side < 0:
                        target_x = PLATFORM_X_MIN
                    else:
                        target_x = PLATFORM_X_MAX

                    target_y = np.clip(PLATFORM_Y_MIN + self.rng.normal(0.0, 0.05), PLATFORM_Y_MIN, PLATFORM_Y_MAX)

            return np.array([target_x, target_y, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    config = EnvConfig(
        terminate_on_stock_out=(stage != 1),
        max_episode_steps=(min(max_episode_steps, 500) if stage == 1 else max_episode_steps),
        yolo_infer_every_n_steps=3,
        action_repeat_steps=1,
        tap_latch_steps=1,
    )
    base = BrawlDeepEnv(config=config)

    if stage == 1:
        sampler = Stage1MixedGoalSampler()
        return LLCEnv(
            env=base,
            goal_sampler=sampler,
            min_goal_duration=100,
            max_goal_duration=100,
            goal_error_scale=0.14,
            goal_progress_scale=5.0,
            goal_progress_clip_min=-0.10,
            goal_progress_clip_max=0.30,
            reward_clip=2.0,
            damage_reward_scale=0.0,
            damage_penalty_scale=0.0,
            hitstun_bonus_scale=0.0,
            stock_event_scale=0.0,
            goal_success_threshold=0.20,
            goal_success_bonus=0.25,
            adaptive_success_bonus=False,
            success_bonus_multiplier=3.0,
            success_bonus_min=0.25,
            success_bonus_max=0.25,
            success_enter_threshold=0.18,
            success_exit_threshold=0.24,
            success_cooldown_steps=4,
            segment_end_success_bonus=0.0,
            segment_end_fail_penalty=0.0,
            segment_magnitude_ema_alpha=0.05,
            goal_error_mode="xy",
            mask_opponent_features=False,
            resample_goal_on_timer=False,
            terminate_on_goal_success=True,
            terminate_on_death=True,
            min_alive_steps_before_goal_termination=20,
            success_terminal_bonus=0.60,
            death_terminal_penalty=-0.25,
            suppress_reward_when_player_missing=True,
        )

    sampler = GoalSampler(recovery_prob=0.16, engage_prob=0.30, spacing_prob=0.20)
    return LLCEnv(
        env=base,
        goal_sampler=sampler,
        min_goal_duration=16,
        max_goal_duration=16,
        goal_error_scale=0.18,
        goal_progress_scale=1.05,
        reward_clip=2.2,
        damage_reward_scale=0.015,
        damage_penalty_scale=0.012,
        hitstun_bonus_scale=0.012,
        goal_success_threshold=0.20,
        goal_success_bonus=0.8,
        adaptive_success_bonus=True,
        success_bonus_multiplier=2.5,
        success_bonus_min=0.4,
        success_bonus_max=1.0,
        success_enter_threshold=0.18,
        success_exit_threshold=0.24,
        success_cooldown_steps=4,
        segment_end_success_bonus=0.20,
        segment_end_fail_penalty=-0.04,
        segment_magnitude_ema_alpha=0.05,
    )


class LLCGoalDashboardCallback(BaseCallback):
    """Save PNG dashboards with per-goal and per-episode diagnostics."""

    def __init__(
        self,
        save_dir: Path,
        stage: int,
        plot_every: int = 5,
        moving_avg: int = 30,
        top_goals: int = 12,
        target_cursor: TargetCursor | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.stage = stage
        self.plot_every = max(1, int(plot_every))
        self.moving_avg = max(2, int(moving_avg))
        self.top_goals = max(4, int(top_goals))
        self.target_cursor = target_cursor

        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_goal_errors: list[float] = []
        self.episode_goal_rewards: list[float] = []
        self.episode_combat_rewards: list[float] = []
        self.episode_goal_success_rate: list[float] = []
        self.episode_goal_success_events: list[float] = []
        self.episode_adaptive_bonus_mean: list[float] = []
        self.episode_segment_terminal_bonus_mean: list[float] = []

        self.goal_segment_rewards: dict[str, list[float]] = defaultdict(list)
        self.goal_segment_lengths: dict[str, list[int]] = defaultdict(list)
        self.goal_episode_counts: dict[str, int] = defaultdict(int)
        self.goal_bucket_success_count: dict[str, float] = defaultdict(float)
        self.goal_bucket_total_count: dict[str, int] = defaultdict(int)
        self.goal_lane_reward_sum: dict[str, float] = defaultdict(float)
        self.goal_lane_count: dict[str, int] = defaultdict(int)
        self.goal_lane_success_sum: dict[str, float] = defaultdict(float)
        self._raw_goal_keys: set[str] = set()

        self._ep_reward = 0.0
        self._ep_len = 0
        self._ep_goal_error_sum = 0.0
        self._ep_goal_reward_sum = 0.0
        self._ep_combat_sum = 0.0
        self._ep_goal_success_sum = 0.0
        self._ep_goal_success_event_sum = 0.0
        self._ep_adaptive_bonus_sum = 0.0
        self._ep_segment_terminal_bonus_sum = 0.0
        self._ep_seen_goals: set[str] = set()

        self._seg_goal_uid: int | None = None
        self._seg_goal_key: str | None = None
        self._seg_reward_sum = 0.0
        self._seg_len = 0

    @staticmethod
    def _goal_key(goal: np.ndarray) -> str:
        g = np.asarray(goal, dtype=np.float32)
        return "[" + ", ".join(f"{v:.2f}" for v in g) + "]"

    @staticmethod
    def _goal_bucket(goal: np.ndarray, goal_lane: str = "unknown", goal_mode: str = "full") -> str:
        g = np.asarray(goal, dtype=np.float32)

        # Stage-1 XY mode: bucket by lane + spatial target region.
        if goal_mode == "xy":
            tx = float(g[0])
            ty = float(g[1])

            if tx < (PLATFORM_X_MIN + (PLATFORM_X_MAX - PLATFORM_X_MIN) / 3.0):
                x_region = "left"
            elif tx > (PLATFORM_X_MAX - (PLATFORM_X_MAX - PLATFORM_X_MIN) / 3.0):
                x_region = "right"
            else:
                x_region = "center"

            y_region = "above_stage" if ty < PLATFORM_Y_MIN else "stage_or_below"
            return f"{goal_lane}:{x_region}:{y_region}"

        rel_dx, rel_dy, _, _, ledge_dx, ledge_dy = g.tolist()
        if abs(ledge_dx) < 0.16 and abs(ledge_dy) < 0.16:
            return "recover_ledge"
        if abs(rel_dx) < 0.12 and abs(rel_dy) < 0.10:
            return "engage_close"
        if abs(rel_dx) > 0.22 and abs(rel_dy) < 0.18:
            return "spacing"
        if rel_dy < -0.12:
            return "edgeguard_low"
        if rel_dy > 0.12:
            return "juggle_high"
        return "maneuver"

    def _finalize_segment(self) -> None:
        if self._seg_goal_key is None or self._seg_len <= 0:
            return
        self.goal_segment_rewards[self._seg_goal_key].append(float(self._seg_reward_sum))
        self.goal_segment_lengths[self._seg_goal_key].append(int(self._seg_len))
        self._seg_goal_uid = None
        self._seg_goal_key = None
        self._seg_reward_sum = 0.0
        self._seg_len = 0

    def _moving_average(self, arr: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if len(arr) < k:
            return np.array([]), np.array([])
        kernel = np.ones(k, dtype=np.float32) / float(k)
        ma = np.convolve(arr, kernel, mode="valid")
        x = np.arange(k, len(arr) + 1)
        return x, ma

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, info in enumerate(infos):
            goal = np.asarray(info.get("goal", np.zeros(6, dtype=np.float32)), dtype=np.float32)
            goal_key = self._goal_key(goal)
            goal_bucket = self._goal_bucket(goal)
            goal_lane = str(info.get("goal_lane", "unknown"))
            goal_mode = str(info.get("goal_error_mode", "full"))
            goal_bucket = self._goal_bucket(goal, goal_lane=goal_lane, goal_mode=goal_mode)
            goal_uid = int(info.get("goal_uid", -1))
            reward = float(info.get("llc_reward", 0.0))

            if self.target_cursor is not None and goal_mode == "xy":
                self.target_cursor.update_target_norm(float(goal[0]), float(goal[1]), visible=True)
            self._raw_goal_keys.add(goal_key)

            self._ep_reward += reward
            self._ep_len += 1
            self._ep_goal_error_sum += float(info.get("goal_error", 0.0))
            self._ep_goal_reward_sum += float(info.get("goal_reward", 0.0))
            self._ep_combat_sum += float(info.get("combat_aux_reward", 0.0))
            goal_success = float(info.get("goal_success", 0.0))
            self._ep_goal_success_sum += goal_success
            self._ep_goal_success_event_sum += float(info.get("goal_success_event", 0.0))
            self._ep_adaptive_bonus_sum += float(info.get("goal_success_bonus_adaptive", 0.0))
            self._ep_segment_terminal_bonus_sum += float(info.get("segment_terminal_bonus", 0.0))
            self._ep_seen_goals.add(goal_bucket)
            self.goal_bucket_success_count[goal_bucket] += goal_success
            self.goal_bucket_total_count[goal_bucket] += 1
            self.goal_lane_reward_sum[goal_lane] += reward
            self.goal_lane_success_sum[goal_lane] += goal_success
            self.goal_lane_count[goal_lane] += 1

            if self._seg_goal_uid is None:
                self._seg_goal_uid = goal_uid
                self._seg_goal_key = goal_bucket

            if goal_uid != self._seg_goal_uid:
                self._finalize_segment()
                self._seg_goal_uid = goal_uid
                self._seg_goal_key = goal_bucket

            self._seg_reward_sum += reward
            self._seg_len += 1

            done_flag = bool(dones[i]) if i < len(dones) else False
            if info.get("terminal_observation") is not None:
                done_flag = True

            if done_flag:
                self._finalize_segment()
                self.episode_rewards.append(float(self._ep_reward))
                self.episode_lengths.append(int(self._ep_len))

                denom = max(1, self._ep_len)
                self.episode_goal_errors.append(float(self._ep_goal_error_sum / denom))
                self.episode_goal_rewards.append(float(self._ep_goal_reward_sum / denom))
                self.episode_combat_rewards.append(float(self._ep_combat_sum / denom))
                self.episode_goal_success_rate.append(float(self._ep_goal_success_sum / denom))
                self.episode_goal_success_events.append(float(self._ep_goal_success_event_sum / denom))
                self.episode_adaptive_bonus_mean.append(float(self._ep_adaptive_bonus_sum / denom))
                self.episode_segment_terminal_bonus_mean.append(float(self._ep_segment_terminal_bonus_sum / denom))

                for gk in self._ep_seen_goals:
                    self.goal_episode_counts[gk] += 1

                ep_num = len(self.episode_rewards)
                if ep_num % self.plot_every == 0:
                    self._save_dashboard()

                self._ep_reward = 0.0
                self._ep_len = 0
                self._ep_goal_error_sum = 0.0
                self._ep_goal_reward_sum = 0.0
                self._ep_combat_sum = 0.0
                self._ep_goal_success_sum = 0.0
                self._ep_goal_success_event_sum = 0.0
                self._ep_adaptive_bonus_sum = 0.0
                self._ep_segment_terminal_bonus_sum = 0.0
                self._ep_seen_goals = set()

        return True

    def _on_training_end(self) -> None:
        if self.target_cursor is not None:
            self.target_cursor.stop()

    def _save_dashboard(self) -> None:
        if len(self.episode_rewards) < 2:
            return

        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
        except Exception:
            return

        fig, axes = plt.subplots(2, 3, figsize=(19, 11))
        fig.suptitle(
            f"LLC Stage {self.stage} Goal Debug Dashboard - ep {len(self.episode_rewards)} - step {self.num_timesteps:,}",
            fontsize=14,
            fontweight="bold",
        )

        episodes = np.arange(1, len(self.episode_rewards) + 1)

        ax = axes[0, 0]
        rewards = np.array(self.episode_rewards, dtype=np.float32)
        ax.plot(episodes, rewards, alpha=0.35, color="steelblue", label="reward")
        x_ma, y_ma = self._moving_average(rewards, self.moving_avg)
        if len(x_ma) > 0:
            ax.plot(x_ma, y_ma, color="navy", linewidth=2, label=f"MA({self.moving_avg})")
        ax.set_title("Episode Reward")
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[0, 1]
        ranked_goals = sorted(self.goal_segment_rewards.items(), key=lambda kv: len(kv[1]), reverse=True)
        top = ranked_goals[: self.top_goals]
        if top:
            labels = [k for k, _ in top]
            means = [float(np.mean(v)) for _, v in top]
            counts = [len(v) for _, v in top]
            succ = [
                100.0 * float(self.goal_bucket_success_count.get(lbl, 0.0))
                / float(max(1, self.goal_bucket_total_count.get(lbl, 1)))
                for lbl in labels
            ]
            y = np.arange(len(labels))
            ax.barh(y, means, color="teal", alpha=0.85)
            ax.set_yticks(y)
            ax.set_yticklabels([f"{labels[i]} ({counts[i]}, {succ[i]:.1f}% succ)" for i in range(len(labels))])
            ax.set_xlabel("mean segment reward")
            ax.set_title("Reward Per Goal Bucket")
            ax.grid(True, axis="x", alpha=0.25)
        else:
            ax.set_title("Reward Per Goal Bucket")
            ax.text(0.1, 0.5, "No sampled goals yet", transform=ax.transAxes)

        ax = axes[0, 2]
        ranked_eps = sorted(self.goal_episode_counts.items(), key=lambda kv: kv[1], reverse=True)[: self.top_goals]
        if ranked_eps:
            labels = [k for k, _ in ranked_eps]
            counts = [v for _, v in ranked_eps]
            y = np.arange(len(labels))
            ax.barh(y, counts, color="darkorange", alpha=0.85)
            ax.set_yticks(y)
            ax.set_yticklabels(labels)
            ax.set_xlabel("episodes containing goal")
            ax.set_title("Episodes Per Goal Bucket")
            ax.grid(True, axis="x", alpha=0.25)
        else:
            ax.set_title("Episodes Per Goal Bucket")
            ax.text(0.1, 0.5, "No episode-goal stats yet", transform=ax.transAxes)

        ax = axes[1, 0]
        goal_err = np.array(self.episode_goal_errors, dtype=np.float32)
        ax.plot(episodes, goal_err, alpha=0.35, color="crimson", label="mean goal error")
        x_ma, y_ma = self._moving_average(goal_err, self.moving_avg)
        if len(x_ma) > 0:
            ax.plot(x_ma, y_ma, color="darkred", linewidth=2, label=f"MA({self.moving_avg})")
        ax.set_title("Goal Tracking Error")
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[1, 1]
        g_comp = np.array(self.episode_goal_rewards, dtype=np.float32)
        c_comp = np.array(self.episode_combat_rewards, dtype=np.float32)
        xg, yg = self._moving_average(g_comp, self.moving_avg)
        xc, yc = self._moving_average(c_comp, self.moving_avg)
        if len(xg) > 0:
            ax.plot(xg, yg, color="purple", linewidth=2, label="goal reward MA")
        if len(xc) > 0:
            ax.plot(xc, yc, color="green", linewidth=2, label="combat aux MA")
        if len(xg) == 0 and len(xc) == 0:
            ax.plot(episodes, g_comp, alpha=0.35, color="purple", label="goal reward")
            ax.plot(episodes, c_comp, alpha=0.35, color="green", label="combat aux")
        ax.set_title("Reward Components")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Overlay success-rate MA for quick curriculum feedback
        succ = np.array(self.episode_goal_success_rate, dtype=np.float32)
        xs, ys = self._moving_average(succ, self.moving_avg)
        if len(xs) > 0:
            ax2 = ax.twinx()
            ax2.plot(xs, ys, color="black", linewidth=1.8, linestyle="--", label="goal success MA")
            ax2.set_ylim(0.0, 1.0)
            ax2.set_ylabel("success rate")

        # Add success-event MA to detect threshold flicker vs meaningful entries
        succ_evt = np.array(self.episode_goal_success_events, dtype=np.float32)
        xse, yse = self._moving_average(succ_evt, self.moving_avg)
        if len(xse) > 0:
            ax2 = ax.twinx()
            ax2.spines["right"].set_position(("axes", 1.10))
            ax2.plot(xse, yse, color="gray", linewidth=1.4, linestyle=":", label="success events MA")
            ax2.set_ylim(0.0, max(0.2, float(np.max(yse) * 1.5)))
            ax2.set_ylabel("success events/step")

        ax = axes[1, 2]
        ax.axis("off")
        seg_counts = [len(v) for v in self.goal_segment_rewards.values()]
        seg_lens = [l for vals in self.goal_segment_lengths.values() for l in vals]
        unique_buckets = len(self.goal_segment_rewards)
        unique_raw_goals = len(self._raw_goal_keys)
        last_n = min(20, len(self.episode_rewards))
        summary = (
            f"Episodes: {len(self.episode_rewards)}\n"
            f"Timesteps: {self.num_timesteps:,}\n"
            f"Unique raw goals sampled: {unique_raw_goals}\n"
            f"Unique buckets sampled: {unique_buckets}\n"
            f"Goal segments logged: {sum(seg_counts)}\n"
            f"Avg segment len: {float(np.mean(seg_lens)) if seg_lens else 0.0:.2f}\n"
            f"Avg episode len: {float(np.mean(self.episode_lengths[-last_n:])) if last_n else 0.0:.1f}\n"
            f"Recent ({last_n}) reward: {float(np.mean(self.episode_rewards[-last_n:])) if last_n else 0.0:.3f}\n"
            f"Recent ({last_n}) goal error: {float(np.mean(self.episode_goal_errors[-last_n:])) if last_n else 0.0:.3f}\n"
            f"Recent ({last_n}) goal success: {100.0 * float(np.mean(self.episode_goal_success_rate[-last_n:])) if last_n else 0.0:.1f}%\n"
            f"Recent ({last_n}) adaptive bonus: {float(np.mean(self.episode_adaptive_bonus_mean[-last_n:])) if last_n else 0.0:.3f}\n"
            f"Recent ({last_n}) segment end bonus: {float(np.mean(self.episode_segment_terminal_bonus_mean[-last_n:])) if last_n else 0.0:.3f}\n"
            f"Lane nav reward/success: {self._lane_summary('navigation')}\n"
            f"Lane combat reward/success: {self._lane_summary('combat_proxy')}\n"
            f"Lane shares (nav/combat): {self._lane_share('navigation'):.1f}% / {self._lane_share('combat_proxy'):.1f}%\n"
            f"\nXY mode buckets = lane:x_region:y_region\n"
            f"(e.g. navigation:center:above_stage)."
        )
        ax.text(0.02, 0.98, summary, va="top", family="monospace", fontsize=11)

        plt.tight_layout()
        out_path = self.save_dir / f"llc_stage{self.stage}_goal_debug.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)

    def _lane_summary(self, lane: str) -> str:
        count = max(1, int(self.goal_lane_count.get(lane, 0)))
        mean_r = float(self.goal_lane_reward_sum.get(lane, 0.0)) / float(count)
        succ = 100.0 * float(self.goal_lane_success_sum.get(lane, 0.0)) / float(count)
        return f"{mean_r:.3f}/{succ:.1f}%"

    def _lane_share(self, lane: str) -> float:
        total = sum(self.goal_lane_count.values())
        if total <= 0:
            return 0.0
        return 100.0 * float(self.goal_lane_count.get(lane, 0)) / float(total)


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    vec_env = VecMonitor(DummyVecEnv([lambda: make_llc_env(args.stage, args.max_episode_steps)]))

    policy_kwargs = dict(
        features_extractor_class=GoalConditionedModulationExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    if args.resume:
        print(f"[LLC] Resuming from {args.resume}")
        model = PPO.load(
            args.resume,
            env=vec_env,
            learning_rate=args.learning_rate,
            clip_range=args.clip_range,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            seed=args.seed,
            device="cpu",
        )
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            seed=args.seed,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device="cpu",
        )

    print(f"[LLC] Stage {args.stage} training for {args.timesteps:,} timesteps")
    print(f"[LLC] Dashboard PNG: {save_dir / f'llc_stage{args.stage}_goal_debug.png'}")
    print(f"[LLC] Starting in {args.delay:.0f}s - switch to Brawlhalla")
    time.sleep(args.delay)

    callback = LLCGoalDashboardCallback(
        save_dir=save_dir,
        stage=args.stage,
        plot_every=args.plot_every,
        moving_avg=args.moving_avg,
        top_goals=args.top_goals,
        target_cursor=None,
    )

    if not args.disable_target_cursor:
        bounds = None
        if args.cursor_right > 0 and args.cursor_bottom > 0:
            bounds = CursorBounds(
                left=int(args.cursor_left),
                top=int(args.cursor_top),
                right=int(args.cursor_right),
                bottom=int(args.cursor_bottom),
            )
        target_cursor = TargetCursor(bounds=bounds, update_every_steps=int(args.cursor_update_every))
        callback.target_cursor = target_cursor

    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=True)

    model_path = save_dir / f"{args.model_name}_stage{args.stage}.zip"
    model.save(str(model_path))
    print(f"[LLC] Saved model to {model_path}")


if __name__ == "__main__":
    main()
