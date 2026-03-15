from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from config import UI_REGIONS
from env import BrawlDeepEnv, EnvConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RecurrentPPO on BrawlDeepEnv")
    parser.add_argument("--timesteps", type=int, default=500_000, help="Total training timesteps")
    parser.add_argument("--save-dir", type=str, default="models", help="Directory to save model artifacts")
    parser.add_argument("--model-name", type=str, default="recurrent_ppo_brawl", help="Saved model filename")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.15)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--ent-coef-final", type=float, default=0.001)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--moving-avg-window", type=int, default=50, help="Window size for moving-average reward plot")
    parser.add_argument("--plot-every-episodes", type=int, default=5, help="Save reward plots every N completed episodes")
    parser.add_argument("--resume", type=str, default=None, help="Path to a saved .zip model to resume training from")
    return parser.parse_args()


class RewardPlotCallback(BaseCallback):
    REWARD_COMPONENTS = ("dmg_dealt", "ko_reward", "ko_penalty", "game_win", "game_loss", "weapon_held", "approach", "proximity_bonus", "edge")

    def __init__(self, save_dir: Path, moving_avg_window: int = 1000, plot_every_episodes: int = 50,
                 ent_coef_start: float = 0.01, ent_coef_final: float = 0.001, verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.moving_avg_window = max(1, int(moving_avg_window))
        self.plot_every_episodes = max(1, int(plot_every_episodes))
        self.ent_coef_start = ent_coef_start
        self.ent_coef_final = ent_coef_final
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[float] = []
        self.episode_ko_counts: list[float] = []
        self.episode_had_ko: list[float] = []
        self.action_counter: Counter[str] = Counter()
        self._ongoing_ko_per_env: dict[int, float] = {}
        # Per-episode cumulative reward components
        self._ongoing_components: dict[int, dict[str, float]] = {}
        self.episode_components: dict[str, list[float]] = {k: [] for k in self.REWARD_COMPONENTS}

    def _save_episode_plots(self) -> None:
        if not self.episode_rewards:
            return

        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
        except Exception as exc:
            raise RuntimeError("matplotlib is required to generate reward plots") from exc

        rewards = np.asarray(self.episode_rewards, dtype=np.float32)
        episodes = np.arange(1, rewards.shape[0] + 1)

        if rewards.shape[0] >= self.moving_avg_window:
            kernel = np.ones(self.moving_avg_window, dtype=np.float32) / float(self.moving_avg_window)
            moving_avg = np.convolve(rewards, kernel, mode="valid")
            moving_episodes = np.arange(self.moving_avg_window, rewards.shape[0] + 1)
        else:
            moving_avg = rewards.copy()
            moving_episodes = episodes

        lengths = np.asarray(self.episode_lengths, dtype=np.float32)
        ko_flags = np.asarray(self.episode_had_ko, dtype=np.float32)
        if ko_flags.size >= self.moving_avg_window:
            ko_kernel = np.ones(self.moving_avg_window, dtype=np.float32) / float(self.moving_avg_window)
            ko_rate = np.convolve(ko_flags, ko_kernel, mode="valid")
            ko_episodes = np.arange(self.moving_avg_window, ko_flags.shape[0] + 1)
        else:
            ko_rate = ko_flags.copy()
            ko_episodes = np.arange(1, ko_flags.shape[0] + 1)

        fig, axes = plt.subplots(3, 2, figsize=(16, 14))

        # ── [0,0] Reward per episode ──
        ax = axes[0, 0]
        ax.plot(episodes, rewards, linewidth=0.8, alpha=0.5, label="Reward")
        ax.plot(moving_episodes, moving_avg, linewidth=1.2, label=f"Moving Avg ({self.moving_avg_window})")
        ax.set_title("Reward per Episode")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.legend(loc="best")

        # ── [0,1] Episode length ──
        ax = axes[0, 1]
        if lengths.size > 0:
            length_episodes = np.arange(1, lengths.shape[0] + 1)
            ax.plot(length_episodes, lengths, linewidth=0.8)
        ax.set_title("Episode Length")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Length (steps)")

        # ── [1,0] KO Rate ──
        ax = axes[1, 0]
        if ko_rate.size > 0:
            ax.plot(ko_episodes, ko_rate, linewidth=1.2)
        ax.set_ylim(0.0, 1.0)
        ax.set_title("KO Rate (rolling)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Rate")

        # ── [1,1] Action Frequency ──
        ax = axes[1, 1]
        if self.action_counter:
            top_actions = self.action_counter.most_common(10)
            labels = [label for label, _ in top_actions]
            values = [count for _, count in top_actions]
            ax.barh(labels, values)
            ax.invert_yaxis()
        ax.set_title("Action Frequency (Top 10)")
        ax.set_xlabel("Count")
        ax.set_ylabel("Action [move,jump,dodge,attack]")

        # ── [2,0] Reward Component Breakdown (rolling average) ──
        ax = axes[2, 0]
        comp_colors = {
            "dmg_dealt": "#2ecc71",
            "ko_reward": "#3498db", "ko_penalty": "#c0392b",
            "game_win": "#27ae60", "game_loss": "#8e44ad",
            "weapon_held": "#f39c12", "approach": "#e67e22",
            "proximity_bonus": "#16a085", "edge": "#1abc9c",
        }
        for comp_name in self.REWARD_COMPONENTS:
            comp_data = self.episode_components.get(comp_name, [])
            if len(comp_data) > 0:
                arr = np.asarray(comp_data, dtype=np.float32)
                comp_ep = np.arange(1, arr.shape[0] + 1)
                if arr.shape[0] >= self.moving_avg_window:
                    k = np.ones(self.moving_avg_window, dtype=np.float32) / float(self.moving_avg_window)
                    smoothed = np.convolve(arr, k, mode="valid")
                    smoothed_ep = np.arange(self.moving_avg_window, arr.shape[0] + 1)
                else:
                    smoothed = arr
                    smoothed_ep = comp_ep
                ax.plot(smoothed_ep, smoothed, linewidth=1.0, label=comp_name,
                        color=comp_colors.get(comp_name))
        ax.set_title("Reward Components (rolling avg)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative per Episode")
        ax.legend(loc="best", fontsize=7)

        # ── [2,1] Reward Component Bar (latest N episodes average) ──
        ax = axes[2, 1]
        recent_n = min(self.moving_avg_window, len(self.episode_rewards))
        if recent_n > 0:
            comp_means = {}
            for comp_name in self.REWARD_COMPONENTS:
                comp_data = self.episode_components.get(comp_name, [])
                if len(comp_data) >= recent_n:
                    comp_means[comp_name] = float(np.mean(comp_data[-recent_n:]))
            if comp_means:
                bar_labels = list(comp_means.keys())
                bar_values = [comp_means[k] for k in bar_labels]
                bar_colors = [comp_colors.get(k, "#7f8c8d") for k in bar_labels]
                ax.barh(bar_labels, bar_values, color=bar_colors)
                ax.invert_yaxis()
        ax.set_title(f"Avg Component (last {recent_n} eps)")
        ax.set_xlabel("Avg Cumulative Reward")

        plt.tight_layout()
        plt.savefig(self.save_dir / "training_dashboard.png", dpi=140)
        plt.close(fig)

    def _on_step(self) -> bool:
        # ── Entropy coefficient annealing ──
        progress = self.model.num_timesteps / self.model._total_timesteps
        new_ent = self.ent_coef_start + progress * (self.ent_coef_final - self.ent_coef_start)
        self.model.ent_coef = new_ent

        infos = self.locals.get("infos")
        if infos is None:
            return True

        for env_idx, info in enumerate(infos):
            if not isinstance(info, dict):
                continue

            action = info.get("effective_action")
            if isinstance(action, (list, tuple)) and len(action) == 4:
                try:
                    action_key = f"[{int(action[0])},{int(action[1])},{int(action[2])},{int(action[3])}]"
                    self.action_counter[action_key] += 1
                except Exception:
                    pass

            op_stock_lost_step = info.get("op_stock_lost_step")
            if op_stock_lost_step is not None:
                self._ongoing_ko_per_env[env_idx] = self._ongoing_ko_per_env.get(env_idx, 0.0) + float(op_stock_lost_step)

            # Accumulate reward components for this episode
            rb = info.get("reward_breakdown")
            if isinstance(rb, dict):
                if env_idx not in self._ongoing_components:
                    self._ongoing_components[env_idx] = {k: 0.0 for k in self.REWARD_COMPONENTS}
                for comp in self.REWARD_COMPONENTS:
                    self._ongoing_components[env_idx][comp] = (
                        self._ongoing_components[env_idx].get(comp, 0.0) + float(rb.get(comp, 0.0))
                    )

            episode_info = info.get("episode")
            if isinstance(episode_info, dict) and "r" in episode_info:
                self.episode_rewards.append(float(episode_info["r"]))
                episode_length = episode_info.get("l")
                if episode_length is not None:
                    self.episode_lengths.append(float(episode_length))
                else:
                    self.episode_lengths.append(float("nan"))

                ko_count = float(self._ongoing_ko_per_env.get(env_idx, 0.0))
                self.episode_ko_counts.append(ko_count)
                self.episode_had_ko.append(1.0 if ko_count > 0.0 else 0.0)
                self._ongoing_ko_per_env[env_idx] = 0.0

                # Store cumulative reward components for this episode
                ep_comps = self._ongoing_components.pop(env_idx, {})
                for comp in self.REWARD_COMPONENTS:
                    self.episode_components[comp].append(float(ep_comps.get(comp, 0.0)))

                if len(self.episode_rewards) % self.plot_every_episodes == 0:
                    self._save_episode_plots()
        return True

    def _on_training_end(self) -> None:
        self._save_episode_plots()


def make_env() -> BrawlDeepEnv:
    config = EnvConfig(
        max_vel=0.15,
        max_weapon_missing=5,
        vy_ground_threshold=0.01,
        terminate_on_stock_out=True,
        ui_regions=dict(UI_REGIONS),
        yolo_infer_every_n_steps=3,
        yolo_max_det=5,
        yolo_verbose=False,
        yolo_infer_width=640,
        yolo_infer_height=360,
        use_tracker_layer=True,
        tracker_max_missing=8,
        tracker_iou_threshold=0.1,
        tracker_smooth_alpha=0.6,
        emit_detailed_info=False,
        profile_step_timing=True,
        profile_window_size=120,
        action_repeat_steps=1,
        tap_latch_steps=1,
        max_episode_steps=1200,
    )
    return BrawlDeepEnv(config=config)


def validate_observation_contract(env: BrawlDeepEnv) -> None:
    feature_names = env.get_observation_spec()
    obs_shape = env.observation_space.shape
    if obs_shape is None:
        raise ValueError("Observation space shape is undefined")
    obs_dim = obs_shape[0]

    if len(feature_names) != obs_dim:
        raise ValueError(
            f"Observation spec mismatch: {len(feature_names)} names vs observation dim {obs_dim}"
        )


def validate_runtime_sanity(env: BrawlDeepEnv, steps: int = 5) -> None:
    obs, _ = env.reset(seed=0)
    if not np.all(np.isfinite(obs)):
        raise ValueError("Initial observation contains non-finite values")

    if env.memory.self_stocks_left != 3.0 or env.memory.op_stocks_left != 3.0:
        raise ValueError("Initial stock state is expected to start at 3 for both players")

    for _ in range(max(1, steps)):
        obs, reward, terminated, truncated, _ = env.step((0, 0, 0, 0))
        if not np.all(np.isfinite(obs)):
            raise ValueError("Runtime observation contains non-finite values")
        if not np.isfinite(reward):
            raise ValueError("Runtime reward contains non-finite values")

        if terminated or truncated:
            obs, _ = env.reset(seed=0)
            if not np.all(np.isfinite(obs)):
                raise ValueError("Observation after reset contains non-finite values")


def train() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    env = make_env()
    validate_observation_contract(env)
    validate_runtime_sanity(env)
    vec_env = VecMonitor(DummyVecEnv([lambda: env]))
    reward_plot_callback = RewardPlotCallback(
        save_dir=save_dir,
        moving_avg_window=args.moving_avg_window,
        plot_every_episodes=args.plot_every_episodes,
        ent_coef_start=float(args.ent_coef),
        ent_coef_final=float(args.ent_coef_final),
    )

    if args.resume:
        resume_path = Path(args.resume)
        # sb3 .load() auto-appends .zip if missing
        if not resume_path.exists() and not resume_path.with_suffix(".zip").exists():
            raise FileNotFoundError(f"Resume model not found: {resume_path}")
        print(f"Resuming training from: {resume_path}")
        model = RecurrentPPO.load(
            str(resume_path),
            env=vec_env,
            learning_rate=args.learning_rate,
            clip_range=args.clip_range,
            ent_coef=float(args.ent_coef),
            vf_coef=args.vf_coef,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            verbose=1,
            seed=args.seed,
        )
    else:
        model = RecurrentPPO(
            policy="MlpLstmPolicy",
            env=vec_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=float(args.ent_coef),
            vf_coef=args.vf_coef,
            verbose=1,
            seed=args.seed,
        )

    interrupted = False
    try:
        model.learn(total_timesteps=args.timesteps, progress_bar=True, callback=reward_plot_callback)
    except KeyboardInterrupt:
        interrupted = True
        print("\nKeyboardInterrupt detected. Saving interrupted checkpoint...")
        interrupted_path = save_dir / f"{args.model_name}_interrupted"
        model.save(str(interrupted_path))
        print(f"Interrupted checkpoint saved to: {interrupted_path}")

    model_path = save_dir / args.model_name
    model.save(str(model_path))
    if interrupted:
        print(f"Final model also saved to: {model_path}")

    vec_env.close()


if __name__ == "__main__":
    train()

