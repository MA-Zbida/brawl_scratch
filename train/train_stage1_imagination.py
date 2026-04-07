#!/usr/bin/env python
"""Stage 1 imagination training: train PPO inside the world model.

Instead of the real game at ~12fps, this runs 16 parallel WorldModelEnv
instances purely on CPU/GPU — typically 1000x+ faster.

Usage:
    python -m train.train_stage1_imagination \
        --world-model world_model/results/world_model.pt \
        --transitions world_model/data/transitions.npz \
        --timesteps 2000000 \
        --n-envs 16 \
        --out-dir train/models

After imagination pre-training, fine-tune on the real game:
    python -m train.train_stage1_homing_missile --algo ppo --resume train/models/stage1_imagination.zip
"""
from __future__ import annotations
 
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from feature_extractor.film_extractor import StageGoalFiLMExtractor
from hierarchical.goals import GOAL_FEATURE_NAMES, GOAL_STATE_SPEC_NAMES
from train.llc_stage_common import (
    DiagnosticCallback,
    StageDashboardCallback,
    StageGoalEnv,
    StageSpec,
)
from world_model.imagination_env import WorldModelEnv


# ── curriculum ────────────────────────────────────────────────────────
# _curriculum_step counts goal resamples, not env timesteps.
# With 16 envs, 80-step eps, goal_duration ~22 → ~58 resamples/rollout.
# ~244 rollouts in 2M steps → ~14K resamples.  Ramp over 15K.
_curriculum_step: int = 0


def _target_sampler(obs: np.ndarray) -> np.ndarray:
    global _curriculum_step
    _curriculum_step += 1
    # Sample a random x position on the platform.
    # Platform x range ≈ [0.315, 0.683].  Use slight inset to avoid edges.
    target_x = np.random.uniform(0.34, 0.66)
    return np.array([target_x, 0, 0, 0, 0, 0, 0], dtype=np.float32)


def _make_spec() -> StageSpec:
    return StageSpec(
        stage_id=1,
        name="stage1_homing_missile",
        mask=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        target_sampler=_target_sampler,
        min_goal_duration=16,
        max_goal_duration=28,
        progress_scale=0.0,
        progress_clip_min=-0.10,
        progress_clip_max=0.80,
        success_threshold=0.03,
        success_bonus=2.0,
        proximity_scale=8.0,
        death_penalty=2.0,
        velocity_penalty_scale=0.0,
        stay_bonus=0.0,
        reward_clip=3.0,
        disable_attack=True,
        disable_dodge=True,
        disable_jump=True,
        reset_perturb_steps=0,  # no perturbation in imagination (already diverse initial states)
        feature_names=list(GOAL_FEATURE_NAMES),
    )


def make_imagination_env(
    model_path: str,
    data_path: str,
    max_episode_steps: int,
    device: str,
) -> StageGoalEnv:
    """Create a single WorldModelEnv wrapped in StageGoalEnv."""
    base = WorldModelEnv(
        model_path=model_path,
        data_path=data_path,
        max_episode_steps=max_episode_steps,
        device=device,
    )
    spec = _make_spec()
    return StageGoalEnv(base, spec)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1 imagination training (world model)")
    # Required paths
    p.add_argument("--world-model", type=str, required=True,
                   help="Path to trained world_model.pt")
    p.add_argument("--transitions", type=str, required=True,
                   help="Path to transitions.npz (collected data)")
    # Training
    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--max-episode-steps", type=int, default=80,
                   help="Short episodes to limit world model drift")
    # PPO hyperparams
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.15)
    p.add_argument("--ent-coef", type=float, default=0.02)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    # Output
    p.add_argument("--out-dir", type=str, default="train/models")
    p.add_argument("--model-name", type=str, default="stage1_imagination")
    p.add_argument("--resume", type=str, default=None)
    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda",
                   help="Device for world model inference and PPO training")
    p.add_argument("--plot-every", type=int, default=20)
    p.add_argument("--moving-avg", type=int, default=500)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = Path(args.out_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── create parallel imagination envs ─────────────────────────────
    def _make_env():
        return make_imagination_env(
            model_path=args.world_model,
            data_path=args.transitions,
            max_episode_steps=args.max_episode_steps,
            device=args.device,
        )

    vec_env = VecMonitor(DummyVecEnv([_make_env for _ in range(args.n_envs)]))

    vecnorm_path = save_dir / f"{args.model_name}.vecnormalize.pkl"
    if args.resume and vecnorm_path.exists():
        vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
    else:
        vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=False, clip_obs=10.0)

    # ── build PPO ────────────────────────────────────────────────────
    policy_kwargs = dict(
        features_extractor_class=StageGoalFiLMExtractor,
        features_extractor_kwargs=dict(
            goal_feature_names=GOAL_STATE_SPEC_NAMES,
            features_dim=256,
        ),
        net_arch=dict(pi=[128], vf=[128]),
    )

    if args.resume:
        print(f"Resuming PPO from {args.resume}")
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
            device=args.device,
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
            device=args.device,
        )

    # ── callbacks ────────────────────────────────────────────────────
    spec = _make_spec()
    # NOTE: pass stage_spec=None to disable on-policy HER.  The HER code
    # indexes rollout_buffer rows with a flat counter that assumes n_envs=1.
    # With 16 parallel envs the counter overflows the buffer.  HER is also
    # less critical here because imagination gives us massive throughput.
    dashboard_cb = StageDashboardCallback(
        save_dir=save_dir,
        model_name=args.model_name,
        stage_spec=None,
        plot_every_episodes=args.plot_every,
        moving_avg_window=args.moving_avg,
    )
    diag_cb = DiagnosticCallback(report_every=500, flat_actions=False)
    callbacks = CallbackList([dashboard_cb, diag_cb])

    # ── train ────────────────────────────────────────────────────────
    n_envs = args.n_envs
    steps_per_rollout = args.n_steps * n_envs
    est_rollouts = args.timesteps / steps_per_rollout

    print("=" * 60)
    print("IMAGINATION TRAINING — Stage 1 (Homing Missile)")
    print("=" * 60)
    print(f"  World model  : {args.world_model}")
    print(f"  Transitions  : {args.transitions}")
    print(f"  Parallel envs: {n_envs}")
    print(f"  Total steps  : {args.timesteps:,}")
    print(f"  Steps/rollout: {steps_per_rollout:,}")
    print(f"  Est. rollouts: {est_rollouts:.0f}")
    print(f"  Episode len  : {args.max_episode_steps}")
    print(f"  Device       : {args.device}")
    print("=" * 60)

    t0 = time.perf_counter()

    interrupted = False
    try:
        model.learn(
            total_timesteps=args.timesteps,
            progress_bar=True,
            callback=callbacks,
        )
    except KeyboardInterrupt:
        interrupted = True
        int_path = save_dir / f"{args.model_name}_interrupted.zip"
        model.save(str(int_path))
        print(f"Interrupted — saved to {int_path}")
    finally:
        try:
            dashboard_cb._on_training_end()
        except Exception:
            pass

    elapsed = time.perf_counter() - t0

    # ── save ─────────────────────────────────────────────────────────
    final_path = save_dir / f"{args.model_name}.zip"
    model.save(str(final_path))
    vecn = model.get_vec_normalize_env()
    if vecn is not None:
        vecn.save(str(vecnorm_path))

    print(f"\nSaved to {final_path}")
    print(f"Training took {elapsed:.1f}s ({args.timesteps / elapsed:.0f} steps/s)")
    if not interrupted:
        print(f"\nNext: fine-tune on real env:")
        print(f"  python -m train.train_stage1_homing_missile --algo ppo --resume {final_path}")


if __name__ == "__main__":
    main()
