from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from env import BrawlDeepEnv, EnvConfig
from feature_extractor.memory.state_spec import StateSpec


ActionAdapter = Callable[[np.ndarray], np.ndarray]
TargetSampler = Callable[[np.ndarray], np.ndarray]


FEATURE_SCALE: dict[str, float] = {
    "dist_to_stage_center": 2.0,
    "dist_to_nearest_ledge": 2.0,
    "player_x": 1.0,
    "player_y": 1.0,
    "player_is_offstage": 1.0,
    "player_jumps_norm": 1.0,
    "dy_to_ledge": 1.0,
    "rel_dx": 2.0,
    "rel_dy": 2.0,
    "rel_distance": 2.0,
    "in_strike_range": 1.0,
    "opponent_hitstun": 1.0,
    "frame_advantage_estimate": 2.0,
}


@dataclass
class StageSpec:
    stage_id: int
    name: str
    feature_names: list[str]
    mask: np.ndarray
    target_sampler: TargetSampler
    min_goal_duration: int = 16
    max_goal_duration: int = 32
    progress_scale: float = 1.0
    progress_clip_min: float = -0.1
    progress_clip_max: float = 0.3
    success_threshold: float = 0.12
    success_bonus: float = 0.25
    reward_clip: float = 1.0
    disable_attack: bool = False
    reset_perturb_steps: int = 0


class StageGoalEnv(gym.Wrapper):
    """Stage-specific LLC training wrapper with masked goal-error shaping."""

    def __init__(
        self,
        env: gym.Env,
        spec: StageSpec,
        action_adapter: Optional[ActionAdapter] = None,
    ):
        super().__init__(env)
        self.spec = spec
        self.action_adapter = action_adapter

        self.feature_names = list(spec.feature_names)
        self.feature_indices = [StateSpec.index(n) for n in self.feature_names]
        self.feature_scales = np.array([FEATURE_SCALE[n] for n in self.feature_names], dtype=np.float32)

        self.goal_dim = len(self.feature_names)
        self.mask = np.asarray(spec.mask, dtype=np.float32).reshape(self.goal_dim)
        self.mask = np.clip(self.mask, 0.0, 1.0)

        base_dim = int(env.observation_space.shape[0])
        self._base_dim = base_dim
        self._aug_dim = base_dim + (2 * self.goal_dim)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._aug_dim,),
            dtype=np.float32,
        )

        self._obs_buf = np.zeros((self._aug_dim,), dtype=np.float32)
        self._goal_target = np.zeros((self.goal_dim,), dtype=np.float32)
        self._goal_steps_left = 0
        self._prev_error: float | None = None

    def _extract(self, obs: np.ndarray) -> np.ndarray:
        return obs[self.feature_indices].astype(np.float32)

    def _error(self, obs: np.ndarray, target: np.ndarray) -> float:
        feats = self._extract(obs)
        per_dim = np.abs((feats - target) / np.maximum(self.feature_scales, 1e-6))
        return float(np.sum(self.mask * per_dim))

    def _sample_goal(self, obs: np.ndarray) -> None:
        self._goal_target = self.spec.target_sampler(obs).astype(np.float32)
        self._goal_steps_left = int(np.random.randint(self.spec.min_goal_duration, self.spec.max_goal_duration + 1))

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        feats = self._extract(obs)
        signed = (feats - self._goal_target) / np.maximum(self.feature_scales, 1e-6)
        masked_error = self.mask * signed

        np.copyto(self._obs_buf[: self._base_dim], obs)
        np.copyto(self._obs_buf[self._base_dim : self._base_dim + self.goal_dim], masked_error)
        np.copyto(self._obs_buf[self._base_dim + self.goal_dim :], self.mask)
        return self._obs_buf

    def _perturb_reset(self) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset()
        if self.spec.reset_perturb_steps <= 0:
            return obs, info

        direction = 0 if np.random.rand() < 0.5 else 1
        for _ in range(self.spec.reset_perturb_steps):
            jump = 1 if np.random.rand() < 0.35 else 0
            action = np.array([direction, jump, 0, 0], dtype=np.int64)
            obs, _, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                obs, info = self.env.reset()
                break
        return obs, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            obs, info = self.env.reset(seed=seed, options=options)
        else:
            obs, info = self._perturb_reset()
        obs = np.asarray(obs, dtype=np.float32)
        self._sample_goal(obs)
        self._prev_error = None

        info["stage_name"] = self.spec.name
        info["goal_target"] = self._goal_target.copy()
        info["goal_mask"] = self.mask.copy()
        return self._augment(obs), info

    def step(self, action: Sequence[int]):
        action_arr = np.asarray(action, dtype=np.int64).copy()
        if self.spec.disable_attack:
            action_arr[3] = 0
        if self.action_adapter is not None:
            action_arr = self.action_adapter(action_arr)

        obs, _, terminated, truncated, info = self.env.step(action_arr)
        obs = np.asarray(obs, dtype=np.float32)

        if self._goal_steps_left <= 0:
            self._sample_goal(obs)
            self._prev_error = None
        else:
            self._goal_steps_left -= 1

        curr_error = self._error(obs, self._goal_target)
        progress = 0.0 if self._prev_error is None else (self._prev_error - curr_error)
        reward = self.spec.progress_scale * progress
        reward = float(np.clip(reward, self.spec.progress_clip_min, self.spec.progress_clip_max))

        success = bool(curr_error < self.spec.success_threshold)
        if success:
            reward += self.spec.success_bonus

        reward = float(np.clip(reward, -self.spec.reward_clip, self.spec.reward_clip))
        self._prev_error = curr_error

        info["stage_name"] = self.spec.name
        info["goal_target"] = self._goal_target.copy()
        info["goal_mask"] = self.mask.copy()
        info["goal_error"] = float(curr_error)
        info["goal_progress"] = float(progress)
        info["goal_success"] = float(1.0 if success else 0.0)
        info["goal_steps_left"] = int(self._goal_steps_left)
        info["llc_reward"] = float(reward)
        info["stage_feature_names"] = list(self.feature_names)

        return self._augment(obs), reward, terminated, truncated, info


class StageDashboardCallback(BaseCallback):
    """Informative stage dashboard for learning diagnostics.

    Produces:
    - Step-level CSV (reward, goal error, progress, success and optional combat signals)
    - Episode-level CSV (returns, lengths, success ratios, average errors)
    - PNG dashboard with moving averages and trend diagnostics
    """

    def __init__(
        self,
        save_dir: Path,
        model_name: str,
        plot_every_episodes: int = 5,
        moving_avg_window: int = 300,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.model_name = model_name
        self.plot_every_episodes = max(1, int(plot_every_episodes))
        self.moving_avg_window = max(10, int(moving_avg_window))

        self.step_csv = self.save_dir / f"{self.model_name}_steps.csv"
        self.episode_csv = self.save_dir / f"{self.model_name}_episodes.csv"
        self.plot_path = self.save_dir / f"{self.model_name}_dashboard.png"

        self.step_reward: list[float] = []
        self.step_goal_error: list[float] = []
        self.step_goal_progress: list[float] = []
        self.step_goal_success: list[float] = []
        self.step_op_delta: list[float] = []
        self.step_self_delta: list[float] = []
        self.step_time_index: list[int] = []
        self.stage_name: str = "unknown"
        self.stage_features: list[str] = []

        self.ep_return: list[float] = []
        self.ep_length: list[int] = []
        self.ep_goal_error_mean: list[float] = []
        self.ep_success_ratio: list[float] = []
        self.ep_op_delta_sum: list[float] = []
        self.ep_self_delta_sum: list[float] = []

        self._cur_ep_reward = 0.0
        self._cur_ep_len = 0
        self._cur_ep_goal_error_sum = 0.0
        self._cur_ep_success_sum = 0.0
        self._cur_ep_op_delta_sum = 0.0
        self._cur_ep_self_delta_sum = 0.0

        self._step_writer: Optional[csv.DictWriter] = None
        self._episode_writer: Optional[csv.DictWriter] = None
        self._step_fh = None
        self._episode_fh = None

    def _on_training_start(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._step_fh = open(self.step_csv, "w", newline="", encoding="utf-8")
        self._episode_fh = open(self.episode_csv, "w", newline="", encoding="utf-8")

        self._step_writer = csv.DictWriter(
            self._step_fh,
            fieldnames=[
                "step",
                "reward",
                "goal_error",
                "goal_progress",
                "goal_success",
                "op_delta_damage",
                "self_delta_damage",
                "stage_name",
            ],
        )
        self._step_writer.writeheader()

        self._episode_writer = csv.DictWriter(
            self._episode_fh,
            fieldnames=[
                "episode",
                "return",
                "length",
                "mean_goal_error",
                "success_ratio",
                "op_delta_damage_sum",
                "self_delta_damage_sum",
            ],
        )
        self._episode_writer.writeheader()

    @staticmethod
    def _moving_average(arr: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
        if arr.size < window:
            return np.array([]), np.array([])
        kernel = np.ones(window, dtype=np.float32) / float(window)
        ma = np.convolve(arr, kernel, mode="valid")
        x = np.arange(window, arr.size + 1)
        return x, ma

    @staticmethod
    def _trend(arr: np.ndarray) -> float:
        if arr.size < 2:
            return 0.0
        x = np.arange(arr.size, dtype=np.float32)
        x = x - x.mean()
        y = arr.astype(np.float32) - arr.mean()
        denom = float(np.dot(x, x))
        if denom <= 1e-8:
            return 0.0
        return float(np.dot(x, y) / denom)

    def _plot_dashboard(self) -> None:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return

        if len(self.step_reward) < 10:
            return

        step_idx = np.asarray(self.step_time_index, dtype=np.int32)
        rewards = np.asarray(self.step_reward, dtype=np.float32)
        errors = np.asarray(self.step_goal_error, dtype=np.float32)
        progress = np.asarray(self.step_goal_progress, dtype=np.float32)
        success = np.asarray(self.step_goal_success, dtype=np.float32)
        op_delta = np.asarray(self.step_op_delta, dtype=np.float32)
        self_delta = np.asarray(self.step_self_delta, dtype=np.float32)

        win = int(min(self.moving_avg_window, max(20, len(rewards) // 8)))

        fig, axes = plt.subplots(3, 2, figsize=(16, 11))
        axes = axes.ravel()

        axes[0].plot(step_idx, rewards, alpha=0.25, label="reward/step")
        x_ma, y_ma = self._moving_average(rewards, win)
        if x_ma.size:
            axes[0].plot(x_ma, y_ma, linewidth=2, label=f"reward MA({win})")
        axes[0].set_title("Reward Dynamics")
        axes[0].set_xlabel("Step")
        axes[0].legend(loc="best")
        axes[0].grid(alpha=0.25)

        axes[1].plot(step_idx, errors, alpha=0.25, color="tab:red", label="goal_error/step")
        x_ma, y_ma = self._moving_average(errors, win)
        if x_ma.size:
            axes[1].plot(x_ma, y_ma, color="tab:red", linewidth=2, label=f"goal_error MA({win})")
        axes[1].set_title("Goal Error")
        axes[1].set_xlabel("Step")
        axes[1].legend(loc="best")
        axes[1].grid(alpha=0.25)

        axes[2].plot(step_idx, progress, alpha=0.25, color="tab:green", label="goal_progress/step")
        x_ma, y_ma = self._moving_average(progress, win)
        if x_ma.size:
            axes[2].plot(x_ma, y_ma, color="tab:green", linewidth=2, label=f"progress MA({win})")
        axes[2].axhline(0.0, linestyle="--", linewidth=1)
        axes[2].set_title("Goal Progress")
        axes[2].set_xlabel("Step")
        axes[2].legend(loc="best")
        axes[2].grid(alpha=0.25)

        x_ma, y_ma = self._moving_average(success, win)
        if x_ma.size:
            axes[3].plot(x_ma, y_ma, linewidth=2, color="tab:purple", label=f"success_rate MA({win})")
        cumulative_success = np.cumsum(success) / np.maximum(1.0, np.arange(1, len(success) + 1, dtype=np.float32))
        axes[3].plot(step_idx, cumulative_success, alpha=0.65, label="cumulative success rate")
        axes[3].set_ylim(-0.02, 1.02)
        axes[3].set_title("Success Rate")
        axes[3].set_xlabel("Step")
        axes[3].legend(loc="best")
        axes[3].grid(alpha=0.25)

        if len(self.ep_return) > 0:
            epi = np.arange(1, len(self.ep_return) + 1)
            axes[4].plot(epi, np.asarray(self.ep_return, dtype=np.float32), alpha=0.30, label="episode return")
            ep_ret = np.asarray(self.ep_return, dtype=np.float32)
            ep_win = int(min(30, max(5, len(ep_ret) // 6)))
            ex, ey = self._moving_average(ep_ret, ep_win)
            if ex.size:
                axes[4].plot(ex, ey, linewidth=2, label=f"return MA({ep_win})")
            axes[4].set_title("Episode Return")
            axes[4].set_xlabel("Episode")
            axes[4].legend(loc="best")
            axes[4].grid(alpha=0.25)

        reward_trend = self._trend(np.asarray(rewards[-win:], dtype=np.float32))
        error_trend = self._trend(np.asarray(errors[-win:], dtype=np.float32))
        progress_trend = self._trend(np.asarray(progress[-win:], dtype=np.float32))
        success_recent = float(np.mean(success[-win:])) if len(success) >= win else float(np.mean(success))
        op_recent = float(np.mean(op_delta[-win:])) if len(op_delta) >= win else float(np.mean(op_delta))
        self_recent = float(np.mean(self_delta[-win:])) if len(self_delta) >= win else float(np.mean(self_delta))

        diagnostics: list[str] = []
        diagnostics.append(f"Stage: {self.stage_name}")
        diagnostics.append(f"Features: {', '.join(self.stage_features) if self.stage_features else 'n/a'}")
        diagnostics.append(f"Recent success rate: {success_recent:.3f}")
        diagnostics.append(f"Reward trend (recent): {reward_trend:+.5f}")
        diagnostics.append(f"Error trend (recent): {error_trend:+.5f}")
        diagnostics.append(f"Progress trend (recent): {progress_trend:+.5f}")
        diagnostics.append(f"Recent op/self damage delta: {op_recent:.4f} / {self_recent:.4f}")

        if reward_trend < 0 and error_trend > 0:
            diagnostics.append("Warning: reward declining while error rises (likely policy drift).")
        if success_recent < 0.05:
            diagnostics.append("Warning: low success rate (consider easier goals / larger success region).")
        if abs(progress_trend) < 1e-4:
            diagnostics.append("Warning: near-zero progress trend (possible local optimum or weak exploration).")

        axes[5].axis("off")
        axes[5].set_title("Critical Diagnostics", loc="left")
        axes[5].text(
            0.01,
            0.98,
            "\n".join(diagnostics),
            va="top",
            ha="left",
            family="monospace",
            fontsize=10,
        )

        fig.tight_layout()
        fig.savefig(self.plot_path, dpi=120)
        plt.close(fig)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])

        if len(infos) == 0:
            return True

        for i, info in enumerate(infos):
            reward = float(rewards[i]) if i < len(rewards) else float(info.get("llc_reward", 0.0))
            goal_error = float(info.get("goal_error", 0.0))
            goal_progress = float(info.get("goal_progress", 0.0))
            goal_success = float(info.get("goal_success", 0.0))
            op_delta = float(info.get("op_delta_damage", 0.0))
            self_delta = float(info.get("self_delta_damage", 0.0))

            self.stage_name = str(info.get("stage_name", self.stage_name))
            stage_features = info.get("stage_feature_names")
            if isinstance(stage_features, list):
                self.stage_features = [str(v) for v in stage_features]

            step_number = len(self.step_reward) + 1
            self.step_time_index.append(step_number)
            self.step_reward.append(reward)
            self.step_goal_error.append(goal_error)
            self.step_goal_progress.append(goal_progress)
            self.step_goal_success.append(goal_success)
            self.step_op_delta.append(op_delta)
            self.step_self_delta.append(self_delta)

            if self._step_writer is not None:
                self._step_writer.writerow(
                    {
                        "step": step_number,
                        "reward": reward,
                        "goal_error": goal_error,
                        "goal_progress": goal_progress,
                        "goal_success": goal_success,
                        "op_delta_damage": op_delta,
                        "self_delta_damage": self_delta,
                        "stage_name": self.stage_name,
                    }
                )

            self._cur_ep_reward += reward
            self._cur_ep_len += 1
            self._cur_ep_goal_error_sum += goal_error
            self._cur_ep_success_sum += goal_success
            self._cur_ep_op_delta_sum += op_delta
            self._cur_ep_self_delta_sum += self_delta

            done = bool(dones[i]) if i < len(dones) else False
            if done:
                ep_idx = len(self.ep_return) + 1
                ep_len = max(1, self._cur_ep_len)
                mean_err = self._cur_ep_goal_error_sum / float(ep_len)
                success_ratio = self._cur_ep_success_sum / float(ep_len)

                self.ep_return.append(float(self._cur_ep_reward))
                self.ep_length.append(int(self._cur_ep_len))
                self.ep_goal_error_mean.append(float(mean_err))
                self.ep_success_ratio.append(float(success_ratio))
                self.ep_op_delta_sum.append(float(self._cur_ep_op_delta_sum))
                self.ep_self_delta_sum.append(float(self._cur_ep_self_delta_sum))

                if self._episode_writer is not None:
                    self._episode_writer.writerow(
                        {
                            "episode": ep_idx,
                            "return": self._cur_ep_reward,
                            "length": self._cur_ep_len,
                            "mean_goal_error": mean_err,
                            "success_ratio": success_ratio,
                            "op_delta_damage_sum": self._cur_ep_op_delta_sum,
                            "self_delta_damage_sum": self._cur_ep_self_delta_sum,
                        }
                    )

                self._cur_ep_reward = 0.0
                self._cur_ep_len = 0
                self._cur_ep_goal_error_sum = 0.0
                self._cur_ep_success_sum = 0.0
                self._cur_ep_op_delta_sum = 0.0
                self._cur_ep_self_delta_sum = 0.0

                if ep_idx % self.plot_every_episodes == 0:
                    self._plot_dashboard()

        if self._step_fh is not None:
            self._step_fh.flush()
        if self._episode_fh is not None:
            self._episode_fh.flush()
        return True

    def _on_training_end(self) -> None:
        self._plot_dashboard()
        if self._step_fh is not None:
            self._step_fh.close()
        if self._episode_fh is not None:
            self._episode_fh.close()


def default_env_config(max_episode_steps: int, terminate_on_stock_out: bool = False) -> EnvConfig:
    return EnvConfig(
        terminate_on_stock_out=terminate_on_stock_out,
        max_episode_steps=max_episode_steps,
        yolo_infer_every_n_steps=3,
        action_repeat_steps=1,
        tap_latch_steps=1,
    )


def parse_train_args(default_name: str, default_steps: int) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"Train {default_name}")
    p.add_argument("--timesteps", type=int, default=default_steps)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.15)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--max-episode-steps", type=int, default=1200)
    p.add_argument("--save-dir", type=str, default="train/models")
    p.add_argument("--model-name", type=str, default=default_name)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--plot-every", type=int, default=5)
    p.add_argument("--moving-avg", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--delay", type=float, default=3.0)
    return p.parse_args()


def train_stage_model(args: argparse.Namespace, make_env: Callable[[], gym.Env]) -> None:
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    base_vec = VecMonitor(DummyVecEnv([make_env]))
    vecnorm_path = save_dir / f"{args.model_name}.vecnormalize.pkl"

    if args.resume and vecnorm_path.exists():
        vec_env = VecNormalize.load(str(vecnorm_path), base_vec)
    else:
        vec_env = VecNormalize(base_vec, norm_obs=True, norm_reward=False, clip_obs=10.0)

    if args.resume:
        print(f"[{args.model_name}] Resuming from {args.resume}")
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
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            verbose=1,
            device="cpu",
        )

    print(f"[{args.model_name}] Training for {args.timesteps:,} timesteps")
    print(f"[{args.model_name}] Starting in {args.delay:.0f}s - switch to Brawlhalla")
    time.sleep(args.delay)

    dashboard_cb = StageDashboardCallback(
        save_dir=save_dir,
        model_name=args.model_name,
        plot_every_episodes=args.plot_every,
        moving_avg_window=args.moving_avg,
    )

    interrupted = False
    try:
        model.learn(total_timesteps=args.timesteps, progress_bar=True, callback=dashboard_cb)
    except KeyboardInterrupt:
        interrupted = True
        interrupted_model = save_dir / f"{args.model_name}_interrupted.zip"
        interrupted_norm = save_dir / f"{args.model_name}_interrupted.vecnormalize.pkl"
        model.save(str(interrupted_model))
        vecn = model.get_vec_normalize_env()
        if vecn is not None:
            vecn.save(str(interrupted_norm))
        print(f"[{args.model_name}] Interrupted checkpoint saved: {interrupted_model}")
    finally:
        try:
            dashboard_cb._on_training_end()
        except Exception:
            pass

    final_model = save_dir / f"{args.model_name}.zip"
    model.save(str(final_model))
    vecn = model.get_vec_normalize_env()
    if vecn is not None:
        vecn.save(str(vecnorm_path))
    print(f"[{args.model_name}] Saved model to {final_model}")
    if interrupted:
        print(f"[{args.model_name}] Final model also saved after interruption.")


def make_base_env(max_episode_steps: int, terminate_on_stock_out: bool = False) -> BrawlDeepEnv:
    return BrawlDeepEnv(config=default_env_config(max_episode_steps=max_episode_steps, terminate_on_stock_out=terminate_on_stock_out))
