"""Unified training configuration for LLC curriculum phases.

Replaces scattered CLI arguments with a single typed config.
Phase-specific defaults are applied automatically by ``make_config()``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    """All training hyperparameters in one place."""

    # Core
    phase: str = "locomotion_grounded"
    timesteps: int = 500_000
    max_episode_steps: int = 1200
    save_dir: str = "train/models"
    model_name: str = ""
    resume: Optional[str] = None
    seed: int = 42
    delay: float = 3.0
    device: str = "cpu"

    # PPO
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.15
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Replay & Anchoring (paper §2)
    replay_ratio: float = 0.30
    replay_capacity: int = 262_144
    replay_warmup_updates: int = 1
    anchor_kl_coef: float = 0.02
    anchor_snapshot_count: int = 5
    anchor_update_interval: int = 8
    bc_loss_coef: float = 0.05
    bc_batch_size: int = 128
    bc_demos_path: Optional[str] = None
    pcgrad: bool = True

    # Logging
    plot_every: int = 2500
    log_csv: bool = False
    diag_report_every: int = 0
    moving_avg: int = 300
    eval_every_steps: int = 0
    eval_episodes: int = 3

    # Environment
    death_penalty: float = 1.0
    terminate_on_death: bool = True
    move_mouse_to_goal: bool = False
    yolo_every: int = 2
    yolo_blend_alpha: float = 0.95
    tracker_max_missing: int = 2
    tracker_smooth_alpha: float = 0.85


# Phase-specific overrides applied on top of defaults.
_PHASE_DEFAULTS: dict[str, dict] = {
    "locomotion_grounded": {
        "learning_rate": 3e-4,
        "n_steps": 1024,
        "clip_range": 0.2,
        "ent_coef": 0.03,
        "move_mouse_to_goal": True,
    },
    "locomotion_airborne": {
        "learning_rate": 3e-4,
        "n_steps": 1024,
        "clip_range": 0.2,
        "ent_coef": 0.03,
        "move_mouse_to_goal": True,
    },
    "locomotion_recovery": {
        "learning_rate": 3e-4,
        "n_steps": 1024,
        "clip_range": 0.2,
        "ent_coef": 0.03,
        "move_mouse_to_goal": True,
    },
    "locomotion": {
        "learning_rate": 3e-4,
        "n_steps": 1024,
        "clip_range": 0.2,
        "ent_coef": 0.03,
        "move_mouse_to_goal": True,
    },
}


def make_config(phase: str, **overrides) -> TrainConfig:
    """Create TrainConfig with phase-specific defaults + explicit overrides."""
    cfg = TrainConfig(phase=phase)
    for key, val in _PHASE_DEFAULTS.get(phase, {}).items():
        setattr(cfg, key, val)
    for key, val in overrides.items():
        if val is not None and hasattr(cfg, key):
            setattr(cfg, key, val)
    if not cfg.model_name:
        cfg.model_name = f"llc_{phase}"
    return cfg


def parse_args() -> TrainConfig:
    """Minimal CLI — only what you'd actually change between runs."""
    from train.curriculum_config import PHASES

    p = argparse.ArgumentParser(description="Train LLC curriculum phase")
    p.add_argument("--phase", required=True, choices=list(PHASES))
    p.add_argument("--timesteps", type=int)
    p.add_argument("--resume", type=str)
    p.add_argument("--device", type=str)
    p.add_argument("--save-dir", type=str)
    p.add_argument("--model-name", type=str)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--delay", type=float)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--n-steps", type=int)
    p.add_argument("--max-episode-steps", type=int)
    p.add_argument("--plot-every", type=int)
    p.add_argument("--log-csv", action="store_true", default=None)
    p.add_argument("--eval-every-steps", type=int)
    p.add_argument("--eval-episodes", type=int)
    p.add_argument("--death-penalty", type=float)
    p.add_argument("--no-terminate-on-death", action="store_true")

    args = p.parse_args()
    kw = {k: v for k, v in vars(args).items()
          if v is not None and k not in ("phase", "no_terminate_on_death")}
    if args.no_terminate_on_death:
        kw["terminate_on_death"] = False
    return make_config(args.phase, **kw)
