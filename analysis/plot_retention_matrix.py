#!/usr/bin/env python
"""Retention matrix: phase trained (rows) against phase evaluated (columns).

This is the project's central catastrophic-forgetting figure. Cell (i, j) is the
retention of skill j at the end of training phase i, where

    retention = current_score / best_score_so_far

Values at or above the 0.85 gate read blue; values below it read red. The
midpoint of the colour scale is the gate itself, so "passing" and "failing" are
visually opposite rather than merely different in intensity.

    python -m analysis.plot_retention_matrix

Reads train/models/llc_<phase>_eval.csv and writes
assets/figures/retention-matrix.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import io, style

GATE = 0.85


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models-dir", type=Path, default=io.MODELS_DIR)
    p.add_argument("--figures-dir", type=Path, default=io.FIGURES_DIR)
    p.add_argument("--gate", type=float, default=GATE)
    return p.parse_args()


def build_matrix(models_dir: Path) -> tuple[np.ndarray, list[str], list[str]]:
    """Final-eval retention for every (trained phase, evaluated phase) pair."""
    trained = io.available_phases(models_dir, kind="eval")
    if not trained:
        raise FileNotFoundError(f"No eval CSVs found under {models_dir}")

    matrix = np.full((len(trained), len(io.PHASES)), np.nan, dtype=np.float64)

    for row, train_phase in enumerate(trained):
        data = io.read_csv(io.eval_csv(train_phase, models_dir))
        if "phase" not in data or "retention" not in data:
            continue
        eval_phase_col = np.asarray(data["phase"], dtype=object)
        retention = np.asarray(data["retention"], dtype=np.float64)
        steps = np.asarray(data.get("train_steps", np.arange(retention.size)), dtype=np.float64)

        for col, eval_phase in enumerate(io.PHASES):
            hits = np.flatnonzero(eval_phase_col == eval_phase)
            if hits.size == 0:
                continue
            # Take the latest evaluation for that phase.
            latest = hits[int(np.argmax(steps[hits]))]
            matrix[row, col] = retention[latest]

    keep = [c for c in range(len(io.PHASES)) if not np.all(np.isnan(matrix[:, c]))]
    return matrix[:, keep], trained, [io.PHASES[c] for c in keep]


def main() -> int:
    args = parse_args()
    style.apply()

    try:
        matrix, rows, cols = build_matrix(args.models_dir)
    except FileNotFoundError as exc:
        print(f"{exc}. Run training with --eval-every-steps first.", file=sys.stderr)
        return 1

    # Diverging scale with the retention gate as the neutral midpoint.
    cmap = LinearSegmentedColormap.from_list(
        "retention",
        [style.DIVERGING_LOW, style.DIVERGING_MID, style.DIVERGING_HIGH],
    )
    finite = matrix[np.isfinite(matrix)]
    lo = float(min(0.0, finite.min())) if finite.size else 0.0
    hi = float(max(1.0, finite.max())) if finite.size else 1.0
    norm = TwoSlopeNorm(vmin=lo, vcenter=args.gate, vmax=hi)

    fig, ax = plt.subplots(figsize=(1.35 * len(cols) + 3.2, 0.85 * len(rows) + 2.8))
    ax.grid(False)
    mesh = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(cols)), [c.replace("_", "\n") for c in cols], fontsize=9)
    ax.set_yticks(range(len(rows)), [r.replace("_", " ") for r in rows], fontsize=9)
    ax.set_xlabel("Skill evaluated")
    ax.set_ylabel("After training phase")

    # 2px surface gap between cells.
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.tick_params(which="minor", length=0)
    ax.grid(which="minor", color=style.SURFACE, linewidth=2)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Value labels: the numbers are the data, so they are always visible.
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            value = matrix[r, c]
            if not np.isfinite(value):
                ax.text(c, r, "--", ha="center", va="center", fontsize=9, color=style.INK_MUTED)
                continue
            # Ink stays in text tokens; the cell fill carries magnitude.
            dark_fill = value < args.gate * 0.6 or value > args.gate + (hi - args.gate) * 0.6
            ax.text(
                c, r, f"{value:.2f}",
                ha="center", va="center", fontsize=9.5,
                color="#ffffff" if dark_fill else style.INK_PRIMARY,
            )

    bar = fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.03)
    bar.set_label(f"Retention  (gate = {args.gate:.2f})", color=style.INK_SECONDARY, fontsize=9)
    bar.outline.set_visible(False)
    bar.ax.tick_params(color=style.INK_MUTED, labelcolor=style.INK_MUTED, labelsize=8)

    style.finish(
        fig, ax,
        title="Skill retention across the curriculum",
        subtitle=f"Final evaluation per phase pair; below {args.gate:.2f} fails the advancement gate",
    )

    out = io.ensure_figures_dir(args.figures_dir) / "retention-matrix.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}  ({matrix.shape[0]}x{matrix.shape[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
