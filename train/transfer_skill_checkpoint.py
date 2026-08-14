"""Convert a typed goal-conditioned skill checkpoint into a final initializer."""

from __future__ import annotations

import argparse
from typing import Sequence

from train.final_policy import convert_skill_checkpoint


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transfer a goal-conditioned skill policy into a goal-free final policy"
    )
    parser.add_argument("--skill-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-untyped-skill-checkpoint", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    report = convert_skill_checkpoint(
        args.skill_checkpoint,
        args.output,
        device=args.device,
        allow_untyped_skill_checkpoint=bool(args.allow_untyped_skill_checkpoint),
    )
    print(report.format())
    print(f"Saved goal-free final initializer: {args.output}")


if __name__ == "__main__":
    main()
