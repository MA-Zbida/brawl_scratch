#!/usr/bin/env python
"""Train HSP that outputs continuous goals for a frozen LLC."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from env import BrawlDeepEnv, EnvConfig
from hierarchical.hsp_env import HSPEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train HSP (continuous goals)")
    p.add_argument("--llc", type=str, required=True, help="Path to trained LLC .zip")
    p.add_argument("--timesteps", type=int, default=250_000)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--n-steps", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--macro-steps", type=int, default=12)
    p.add_argument("--max-episode-steps", type=int, default=1200)
    p.add_argument("--save-dir", type=str, default="train/models")
    p.add_argument("--model-name", type=str, default="hsp")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--delay", type=float, default=3.0)
    return p.parse_args()


def build_hsp_env(llc_model: PPO, macro_steps: int, max_episode_steps: int) -> HSPEnv:
    config = EnvConfig(
        terminate_on_stock_out=True,
        max_episode_steps=max_episode_steps,
        yolo_infer_every_n_steps=3,
        action_repeat_steps=1,
        tap_latch_steps=1,
    )
    base = BrawlDeepEnv(config=config)
    return HSPEnv(
        base_env=base,
        llc_model=llc_model,
        macro_steps=macro_steps,
        deterministic_llc=True,
    )


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[HSP] Loading frozen LLC from {args.llc}")
    llc = PPO.load(args.llc)

    def make_env():
        return build_hsp_env(llc_model=llc, macro_steps=args.macro_steps, max_episode_steps=args.max_episode_steps)

    vec_env = VecMonitor(DummyVecEnv([make_env]))

    if args.resume:
        print(f"[HSP] Resuming from {args.resume}")
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

    print(f"[HSP] Training for {args.timesteps:,} timesteps")
    print(f"[HSP] Starting in {args.delay:.0f}s - switch to Brawlhalla")
    time.sleep(args.delay)

    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    path = save_dir / f"{args.model_name}.zip"
    model.save(str(path))
    print(f"[HSP] Saved model to {path}")


if __name__ == "__main__":
    main()
