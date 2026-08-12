#!/usr/bin/env python
"""Episode-success learning curves, one line per curriculum phase.

    python -m analysis.plot_learning_curves
    python -m analysis.plot_learning_curves --metric return --window 25

Reads train/models/llc_<phase>_episodes.csv and writes
assets/figures/learning-curves-<metric>.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import io, style

METRICS: dict[str, tuple[str, str]] = {
    # key -> (csv column, y-axis label)
    "success": ("episode_success", "Episode success rate"),
    "return": ("return", "Episode return"),
    "goal_error": ("mean_goal_error", "Mean goal error"),
    "entropy": ("action_entropy", "Action entropy (normalised)"),
    "idle": ("idle_rate", "Idle-step rate"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metric", choices=sorted(METRICS), default="success")
    p.add_argument("--window", type=int, default=20, help="Moving-average window in episodes")
    p.add_argument("--models-dir", type=Path, default=io.MODELS_DIR)
    p.add_argument("--figures-dir", type=Path, default=io.FIGURES_DIR)
    p.add_argument("--phases", type=str, default="", help="Comma-separated subset; default = all found")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    style.apply()

    column, ylabel = METRICS[args.metric]

    phases = io.available_phases(args.models_dir, kind="episodes")
    if args.phases.strip():
        wanted = {p.strip() for p in args.phases.split(",") if p.strip()}
        phases = [p for p in phases if p in wanted]

    if not phases:
        print(f"No episode CSVs found under {args.models_dir}. Run training first.", file=sys.stderr)
        return 1

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    drawn: list[str] = []
    end_labels: list[tuple[float, float, str, str]] = []

    for phase in phases:
        data = io.read_csv(io.episodes_csv(phase, args.models_dir))
        if column not in data:
            print(f"  skip {phase}: no column '{column}'", file=sys.stderr)
            continue

        x, y = io.moving_average(data[column], args.window)
        if x.size == 0:
            print(f"  skip {phase}: fewer than {args.window} episodes", file=sys.stderr)
            continue

        colour = style.PHASE_COLOR[phase]
        pretty = phase.replace("_", " ")
        ax.plot(x, y, color=colour, solid_capstyle="round", label=pretty)
        end_labels.append((float(x[-1]), float(y[-1]), pretty, colour))
        drawn.append(pretty)

    if not drawn:
        print("Nothing plottable.", file=sys.stderr)
        return 1

    if args.metric in ("success", "entropy", "idle"):
        ax.set_ylim(-0.02, 1.02)

    # Legend is always present for >= 2 series, in addition to direct labels.
    if len(drawn) >= 2:
        ax.legend(loc="upper left", ncols=min(3, len(drawn)))

    # Direct labels last, so y-limits are settled before de-collision.
    style.label_line_ends(ax, end_labels)
    plotted = len(drawn)

    style.finish(
        fig, ax,
        title=f"LLC {ylabel.lower()} by curriculum phase",
        subtitle=f"{args.window}-episode moving average",
        xlabel="Episode",
        ylabel=ylabel,
    )
    # Leave room for the right-hand direct labels.
    ax.margins(x=0.16)

    out = io.ensure_figures_dir(args.figures_dir) / f"learning-curves-{args.metric}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}  ({plotted} phase(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
