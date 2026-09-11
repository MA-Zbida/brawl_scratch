"""Configuration for goal-independent final-agent match training."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class FinalAgentConfig:
    """Hyperparameters for the single policy that directly plays the match."""

    init_from: str | None
    resume: str | None
    allow_untyped_skill_checkpoint: bool = False
    timesteps: int = 500_000
    max_episode_steps: int = 3000
    save_dir: str = "train/models/final_agent"
    model_name: str = "final_agent"
    seed: int = 42
    delay: float = 3.0
    device: str = "cpu"
    learning_rate: float = 1e-4
    n_steps: int = 512
    batch_size: int = 256
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_range: float = 0.1
    ent_coef: float = 0.01
    n_epochs: int = 4
    target_kl: float = 0.01
    mode: str = "training_room"
    checkpoint_every: int = 10240
    eval_every: int = 51200
    eval_episodes: int = 10
    world_replay: str = ""
    forever: bool = False
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    yolo_every: int = 1
    log_csv: bool = True

    @property
    def goal_conditioning(self) -> bool:
        return False

    @property
    def objective(self) -> str:
        return "match"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one goal-independent final combat policy on match reward"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--init-from",
        type=str,
        help="Typed goal-conditioned skill checkpoint used only for weight initialization",
    )
    source.add_argument(
        "--resume",
        type=str,
        help="Typed final-agent checkpoint to continue training",
    )
    parser.add_argument(
        "--allow-untyped-skill-checkpoint",
        action="store_true",
        help=(
            "Explicitly accept a legacy skill archive without role metadata after its "
            "FiLM observation contract is verified"
        ),
    )
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--max-episode-steps", type=int, default=3000)
    parser.add_argument("--save-dir", type=str, default="train/models/final_agent")
    parser.add_argument("--model-name", type=str, default="final_agent")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--target-kl", type=float, default=0.01)
    parser.add_argument("--mode", choices=["training_room", "manual_matches"], default="training_room")
    parser.add_argument("--checkpoint-every", type=int, default=10240)
    parser.add_argument("--eval-every", type=int, default=51200, help="0 disables scheduled evaluation")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Fixed-duration windows in training room")
    parser.add_argument("--world-replay", default="", help="Opt-in compressed transition directory")
    parser.add_argument("--forever", action="store_true", help="Repeat training budgets until Ctrl+C")
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--yolo-every", type=int, default=1)
    parser.add_argument("--log-csv", dest="log_csv", action="store_true", default=True)
    parser.add_argument("--no-log-csv", dest="log_csv", action="store_false")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> FinalAgentConfig:
    args = build_parser().parse_args(argv)
    if int(args.timesteps) < 0:
        raise SystemExit("--timesteps must be non-negative")
    if int(args.n_steps) <= 0 or int(args.batch_size) <= 0:
        raise SystemExit("--n-steps and --batch-size must be positive")
    if int(args.max_episode_steps) <= 0:
        raise SystemExit("--max-episode-steps must be positive")
    if args.n_steps % args.batch_size:
        raise SystemExit("--batch-size must divide --n-steps")
    if args.n_epochs <= 0 or args.target_kl <= 0 or args.learning_rate <= 0:
        raise SystemExit("epochs, KL limit, and learning rate must be positive")
    if not 0 < args.gamma <= 1 or not 0 <= args.gae_lambda <= 1:
        raise SystemExit("gamma must be in (0, 1] and GAE lambda in [0, 1]")
    if args.checkpoint_every <= 0 or args.eval_every < 0 or args.eval_episodes <= 0:
        raise SystemExit("checkpoint/evaluation intervals must be valid positive counts (eval-every may be 0)")
    if args.forever and args.timesteps == 0:
        raise SystemExit("--forever requires positive --timesteps per training block")
    return FinalAgentConfig(**vars(args))
