"""Shared figure style for all generated plots.

Every figure in the report is produced by a script in this package so that the
paper can be regenerated from run CSVs. Nothing is drawn by hand.

The categorical order below is fixed and must never be cycled or reordered: it
was validated for colour-vision-deficient separation on the adjacent-pair list
(worst adjacent CVD dE 9.1, worst adjacent normal-vision dE 19.6, light surface).
Reordering the slots invalidates that result.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# ── Categorical slots — fixed order, assign by entity, never by rank ──────────
SERIES: tuple[str, ...] = (
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
)

# ── Sequential ramp (single hue, light -> dark) for magnitude ────────────────
SEQUENTIAL: tuple[str, ...] = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
)

# ── Diverging poles + neutral midpoint, for signed quantities ────────────────
DIVERGING_LOW = "#d03b3b"   # below the gate
DIVERGING_MID = "#f0efec"   # neutral
DIVERGING_HIGH = "#2a78d6"  # above the gate

# ── Chrome and ink ──────────────────────────────────────────────────────────
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# Stable colour assignment per phase. Colour follows the entity, so filtering
# the phase list must never repaint the survivors.
PHASE_COLOR: dict[str, str] = {
    "recovery_mastery": SERIES[0],
    "movement_fluency": SERIES[1],
    "weapon_acquisition": SERIES[2],
    "spacing_neutral": SERIES[3],
    "combat_execution": SERIES[4],
    "all_skills_llc": SERIES[5],
}


def apply() -> None:
    """Install the shared rcParams. Call once at the top of every figure script."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",

        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 10,

        "text.color": INK_PRIMARY,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK_PRIMARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,

        # Recessive chrome: the data should be the darkest thing on the page.
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.axis": "y",

        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "legend.frameon": False,
        "legend.fontsize": 9,

        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
    })


def label_line_ends(ax, items, min_gap_frac: float = 0.055) -> None:
    """Direct-label several series at their right ends, without overlap.

    Three categorical slots sit below 3:1 contrast on the light surface, so the
    relief rule applies: every series carries a visible label, and identity is
    never communicated by colour alone. That only holds if the labels are
    actually legible, so anchors are spread apart when they would collide.

    ``items`` is a sequence of ``(x, y, text, color)``.
    """
    items = list(items)
    if not items:
        return

    y_lo, y_hi = ax.get_ylim()
    min_gap = abs(y_hi - y_lo) * float(min_gap_frac)

    # Place from the bottom up, pushing each label above the previous one.
    ordered = sorted(range(len(items)), key=lambda i: items[i][1])
    placed: dict[int, float] = {}
    cursor = -float("inf")
    for i in ordered:
        y = float(items[i][1])
        y = max(y, cursor + min_gap) if cursor > -float("inf") else y
        placed[i] = y
        cursor = y

    # If spreading pushed labels past the top, shift the whole stack down.
    overflow = cursor - y_hi
    if overflow > 0:
        for i in placed:
            placed[i] -= overflow

    for i, (x, y, text, color) in enumerate(items):
        y_label = placed[i]
        # Leader from the true line end to the de-collided label position.
        if abs(y_label - y) > 1e-9:
            ax.annotate(
                "",
                xy=(x, y),
                xytext=(x, y_label),
                arrowprops=dict(arrowstyle="-", color=color, linewidth=0.8, alpha=0.55),
                annotation_clip=False,
            )
        ax.annotate(
            text,
            xy=(x, y_label),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=9,
            color=INK_SECONDARY,
            annotation_clip=False,
        )


def finish(fig, ax, title: str, subtitle: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """Apply the standard title block and axis labels."""
    if subtitle:
        ax.set_title(f"{title}\n", loc="left")
        ax.text(
            0.0, 1.02, subtitle,
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9.5, color=INK_SECONDARY,
        )
    else:
        ax.set_title(title, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
