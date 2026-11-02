"""Convert a typed goal-conditioned skill checkpoint into a final initializer."""

from __future__ import annotations

# Support both ``python -m brawl.pipelines.final_agent.transfer_skill_checkpoint`` and the repository's
# documented direct-script entry point while preserving DXGI-before-torch order.

import brawl.capture_first  # noqa: E402,F401

import argparse
from typing import Sequence

from brawl.pipelines.final_agent.policy import convert_skill_checkpoint


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
