#!/usr/bin/env python
"""Generate provenance-safe evidence artifacts for the GCRL paper.

Only canonical current filenames are opened. Historical 3h runs, pilots, failed
runs, and alternate versions are never discovered or used as fallbacks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from brawl.control.action_space import ACTION_DIM, Action
from brawl.analysis import io, style
from brawl.paths import LEGACY_RUNS, RESEARCH


PRIMARY_PHASES: tuple[str, ...] = (
    "movement_fluency",
    "weapon_acquisition",
    "spacing_neutral",
    "combat_execution",
)
PHASE_LABEL = {
    "movement_fluency": "Movement",
    "weapon_acquisition": "Weapon",
    "spacing_neutral": "Spacing",
    "combat_execution": "Combat",
}
DEFAULT_MODELS_DIR = LEGACY_RUNS / "llc" / "easy_bot_v0"
DEFAULT_DEMOS_DIR = LEGACY_RUNS / "llc" / "recollected"
DEFAULT_OUTPUT_DIR = RESEARCH / "paper" / "evidence"
RETENTION_GATE = 0.85


@dataclass(frozen=True)
class LearningSeries:
    phase: str
    transitions: np.ndarray
    success: np.ndarray
    goal_error: np.ndarray
    raw_transitions: np.ndarray
    raw_success: np.ndarray
    raw_goal_error: np.ndarray
    total_transitions: int
    episodes: int
    window: int
    source_path: Path


@dataclass(frozen=True)
class RetentionMatrices:
    checkpoint_phases: tuple[str, ...]
    evaluated_phases: tuple[str, ...]
    skill_score: np.ndarray
    retention: np.ndarray
    episodes: np.ndarray
    source_paths: tuple[Path | None, ...]


@dataclass(frozen=True)
class DemoSummary:
    phase: str
    samples: int
    episodes: int
    attempts: int
    rejected_trivial: int
    distinct_actions: int
    normalized_action_entropy: float
    idle_rate: float
    movement_rate: float
    dodge_rate: float
    attack_rate: float
    pickup_rate: float
    median_episode_steps: float
    single_step_rate: float
    invalid_goal_rows: int
    source_path: Path


@dataclass(frozen=True)
class EvidenceBuildResult:
    outputs: tuple[Path, ...]
    issues: tuple[str, ...]
    publication_ready: bool


def _require_columns(data: dict[str, np.ndarray], required: Iterable[str], path: Path) -> None:
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def _rolling(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values.copy()
    width = min(max(1, int(window)), values.size)
    return np.convolve(values, np.full(width, 1.0 / width), mode="valid")


def load_learning_series(path: Path, *, phase: str, window: int = 20) -> LearningSeries:
    """Load an episode log with the x-axis measured in live transitions."""
    path = Path(path)
    data = io.read_csv(path)
    _require_columns(data, ("length", "episode_success", "mean_goal_error"), path)
    lengths = np.asarray(data["length"], dtype=np.float64)
    success = np.asarray(data["episode_success"], dtype=np.float64)
    goal_error = np.asarray(data["mean_goal_error"], dtype=np.float64)
    if not (lengths.size == success.size == goal_error.size):
        raise ValueError(f"{path} has misaligned episode columns")
    if lengths.size == 0 or np.any(~np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError(f"{path} contains an invalid episode length")
    if np.any(~np.isfinite(success)) or np.any(~np.isfinite(goal_error)):
        raise ValueError(f"{path} contains non-finite paper metrics")

    cumulative = np.cumsum(lengths).astype(np.int64)
    width = min(max(1, int(window)), lengths.size)
    return LearningSeries(
        phase=phase,
        transitions=cumulative[width - 1 :],
        success=_rolling(success, width),
        goal_error=_rolling(goal_error, width),
        raw_transitions=cumulative,
        raw_success=success,
        raw_goal_error=goal_error,
        total_transitions=int(cumulative[-1]),
        episodes=int(lengths.size),
        window=width,
        source_path=path,
    )


def build_retention_matrices(models_dir: Path) -> RetentionMatrices:
    """Load only exact current retention filenames into fixed four-phase matrices."""
    models_dir = Path(models_dir)
    shape = (len(PRIMARY_PHASES), len(PRIMARY_PHASES))
    scores = np.full(shape, np.nan, dtype=np.float64)
    retention = np.full(shape, np.nan, dtype=np.float64)
    episodes = np.full(shape, np.nan, dtype=np.float64)
    sources: list[Path | None] = []

    for row, checkpoint_phase in enumerate(PRIMARY_PHASES):
        path = models_dir / f"llc_{checkpoint_phase}_retention_eval.csv"
        if not path.exists():
            sources.append(None)
            continue
        sources.append(path)
        data = io.read_csv(path)
        _require_columns(data, ("phase", "episodes", "skill_score", "retention"), path)
        names = np.asarray(data["phase"], dtype=object)
        for col, evaluated_phase in enumerate(PRIMARY_PHASES):
            hits = np.flatnonzero(names == evaluated_phase)
            if not hits.size:
                continue
            index = int(hits[-1])
            scores[row, col] = float(data["skill_score"][index])
            retention[row, col] = float(data["retention"][index])
            episodes[row, col] = float(data["episodes"][index])

    return RetentionMatrices(
        checkpoint_phases=PRIMARY_PHASES,
        evaluated_phases=PRIMARY_PHASES,
        skill_score=scores,
        retention=retention,
        episodes=episodes,
        source_paths=tuple(sources),
    )


def _scalar(data: Any, key: str, default: Any) -> Any:
    if key not in data:
        return default
    values = np.asarray(data[key]).reshape(-1)
    return values[0].item() if values.size else default


def _episode_lengths(dones: np.ndarray) -> np.ndarray:
    dones = np.asarray(dones, dtype=bool).reshape(-1)
    if dones.size == 0:
        return np.asarray([], dtype=np.int64)
    ends = list((np.flatnonzero(dones) + 1).astype(int))
    if not ends or ends[-1] != dones.size:
        ends.append(int(dones.size))
    starts = [0, *ends[:-1]]
    return np.asarray([end - start for start, end in zip(starts, ends)], dtype=np.int64)


def summarize_demo(path: Path) -> DemoSummary:
    """Recompute archive composition instead of trusting copied validator output."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        required = ("actions", "dones", "goal_mask", "goal_target")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"{path} is missing required arrays: {', '.join(missing)}")
        actions = np.asarray(data["actions"], dtype=np.int64).reshape(-1)
        dones = np.asarray(data["dones"], dtype=bool).reshape(-1)
        masks = np.asarray(data["goal_mask"], dtype=np.float64)
        targets = np.asarray(data["goal_target"], dtype=np.float64)
        if actions.size != dones.size or masks.shape[0] != actions.size or targets.shape[0] != actions.size:
            raise ValueError(f"{path} contains misaligned demonstration arrays")
        if np.any(actions < 0) or np.any(actions >= ACTION_DIM):
            raise ValueError(f"{path} contains actions outside Discrete({ACTION_DIM})")

        counts = np.bincount(actions, minlength=ACTION_DIM).astype(np.float64)
        probabilities = counts[counts > 0] / max(1.0, float(actions.size))
        entropy = -float(np.sum(probabilities * np.log(probabilities)))
        normalized_entropy = entropy / math.log(ACTION_DIM) if actions.size else 0.0
        lengths = _episode_lengths(dones)
        active_goal = np.any(np.abs(masks) > 1e-12, axis=1)
        zero_target = np.all(np.abs(targets) <= 1e-12, axis=1)
        invalid_goal_rows = int(np.count_nonzero(active_goal & zero_target))
        phase = str(_scalar(data, "phase", path.stem.removesuffix("_demos")))
        episode_count = int(_scalar(data, "episodes_collected", lengths.size))
        attempts = int(_scalar(data, "episodes_attempted", episode_count))
        rejected_trivial = int(_scalar(data, "episodes_rejected_trivial", 0))

    samples = int(actions.size)
    denominator = float(max(1, samples))
    return DemoSummary(
        phase=phase,
        samples=samples,
        episodes=episode_count,
        attempts=attempts,
        rejected_trivial=rejected_trivial,
        distinct_actions=int(np.count_nonzero(counts)),
        normalized_action_entropy=float(normalized_entropy),
        idle_rate=float(counts[int(Action.NOOP)] / denominator),
        movement_rate=float(np.sum(counts[1:9]) / denominator),
        dodge_rate=float(np.sum(counts[9:18]) / denominator),
        attack_rate=float(np.sum(counts[18:26]) / denominator),
        pickup_rate=float(counts[int(Action.PICKUP)] / denominator),
        median_episode_steps=float(np.median(lengths)) if lengths.size else 0.0,
        single_step_rate=float(np.mean(lengths == 1)) if lengths.size else 0.0,
        invalid_goal_rows=invalid_goal_rows,
        source_path=path,
    )


def _source_record(path: Path, *, role: str) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": path.as_posix(),
        "role": role,
        "bytes": int(stat.st_size),
        "sha256": digest.hexdigest(),
        "modified_at_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    for path in paths:
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=style.SURFACE)
    plt.close(fig)
    return paths


def _plot_learning(
    series_by_phase: dict[str, LearningSeries],
    output_dir: Path,
    *,
    metric: str,
) -> list[Path]:
    if metric not in {"success", "goal_error"}:
        raise ValueError(f"Unsupported learning metric: {metric}")
    style.apply()
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.1), sharey=metric == "success")
    ylabel = "Episode success fraction" if metric == "success" else "Mean goal error"
    title = "Corrected skill-acquisition curves" if metric == "success" else "Corrected goal-error curves"

    for ax, phase in zip(axes.ravel(), PRIMARY_PHASES):
        ax.set_title(PHASE_LABEL[phase], loc="left", fontsize=11)
        series = series_by_phase.get(phase)
        if series is None:
            ax.text(
                0.5,
                0.5,
                "No current canonical log",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color=style.INK_MUTED,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        raw_y = series.raw_success if metric == "success" else series.raw_goal_error
        smooth_y = series.success if metric == "success" else series.goal_error
        colour = style.PHASE_COLOR[phase]
        ax.scatter(
            series.raw_transitions / 1000.0,
            raw_y,
            s=7,
            alpha=0.12,
            color=colour,
            linewidths=0,
        )
        ax.plot(series.transitions / 1000.0, smooth_y, color=colour, linewidth=2.1)
        ax.text(
            0.98,
            0.95,
            f"{series.total_transitions:,} transitions | {series.episodes:,} episodes\n"
            f"{series.window}-episode moving mean",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.2,
            color=style.INK_SECONDARY,
            bbox={"facecolor": style.SURFACE, "edgecolor": "none", "alpha": 0.82, "pad": 2.0},
        )
        if metric == "success":
            ax.set_ylim(-0.04, 1.04)
        else:
            ax.set_ylim(bottom=0.0)
        ax.set_xlabel("Cumulative live transitions (thousands)")
        ax.set_ylabel(ylabel)

    fig.suptitle(title, x=0.07, ha="left", fontsize=13, fontweight="semibold")
    fig.text(
        0.07,
        0.94,
        "Points are episodes; lines are lightly smoothed. Each phase keeps its actual interaction budget.",
        ha="left",
        fontsize=9.5,
        color=style.INK_SECONDARY,
    )
    fig.tight_layout(rect=(0.04, 0.02, 1.0, 0.92))
    stem = "corrected_learning_success" if metric == "success" else "corrected_goal_error"
    return _save_figure(fig, output_dir, stem)


def _plot_matrix(
    matrices: RetentionMatrices,
    output_dir: Path,
    *,
    metric: str,
    gate: float,
) -> list[Path]:
    style.apply()
    if metric == "skill_score":
        values = matrices.skill_score
        cmap = LinearSegmentedColormap.from_list("skill", style.SEQUENTIAL)
        norm: Any = plt.Normalize(0.0, 1.0)
        title = "Preliminary current skill scores"
        stem = "corrected_skill_score_matrix"
        colorbar_label = "Skill score"
    elif metric == "retention":
        values = matrices.retention
        cmap = LinearSegmentedColormap.from_list(
            "retention",
            [style.DIVERGING_LOW, style.DIVERGING_MID, style.DIVERGING_HIGH],
        )
        norm = TwoSlopeNorm(vmin=0.0, vcenter=gate, vmax=1.0)
        title = "Preliminary current retention"
        stem = "corrected_retention_matrix"
        colorbar_label = f"Retention (gate = {gate:.2f})"
    else:
        raise ValueError(f"Unsupported matrix metric: {metric}")

    fig, ax = plt.subplots(figsize=(7.6, 5.5))
    ax.grid(False)
    mesh = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=norm, aspect="equal")
    ax.set_xticks(
        range(len(PRIMARY_PHASES)),
        [PHASE_LABEL[phase] for phase in PRIMARY_PHASES],
        rotation=22,
        ha="right",
    )
    ax.set_yticks(range(len(PRIMARY_PHASES)), [PHASE_LABEL[phase] for phase in PRIMARY_PHASES])
    ax.set_xlabel("Skill evaluated")
    ax.set_ylabel("Checkpoint after phase")

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            count = matrices.episodes[row, col]
            if not np.isfinite(value):
                label = "not run" if col <= row else "—"
                ax.text(col, row, label, ha="center", va="center", fontsize=8.5, color=style.INK_MUTED)
                continue
            dark = (metric == "skill_score" and value > 0.58) or (metric == "retention" and value < 0.45)
            ax.text(
                col,
                row,
                f"{value:.2f}\nn={int(count)}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if dark else style.INK_PRIMARY,
            )

    ax.set_xticks(np.arange(-0.5, len(PRIMARY_PHASES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(PRIMARY_PHASES), 1), minor=True)
    ax.tick_params(which="minor", length=0)
    ax.grid(which="minor", color=style.SURFACE, linewidth=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(mesh, ax=ax, fraction=0.045, pad=0.04)
    colorbar.set_label(colorbar_label)
    colorbar.outline.set_visible(False)
    ax.set_title(f"{title}\n", loc="left", fontsize=13, fontweight="semibold")
    ax.text(
        0.0,
        1.02,
        "Current canonical evaluations only; n is printed in every measured cell.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color=style.INK_SECONDARY,
    )
    fig.tight_layout()
    return _save_figure(fig, output_dir, stem)


def _plot_demo_quality(summaries: Sequence[DemoSummary], output_dir: Path) -> list[Path]:
    style.apply()
    ordered = [next((item for item in summaries if item.phase == phase), None) for phase in PRIMARY_PHASES]
    ordered = [item for item in ordered if item is not None]
    labels = [PHASE_LABEL[item.phase] for item in ordered]
    colours = [style.PHASE_COLOR[item.phase] for item in ordered]
    y = np.arange(len(ordered))

    fig, (ax_samples, ax_mix) = plt.subplots(
        1,
        2,
        figsize=(11.0, 4.8),
        gridspec_kw={"width_ratios": [0.82, 1.35]},
    )
    samples = np.asarray([item.samples for item in ordered])
    ax_samples.barh(y, samples, color=colours, height=0.62)
    ax_samples.set_yticks(y, labels)
    ax_samples.invert_yaxis()
    ax_samples.set_xlabel("Demonstration transitions")
    ax_samples.set_title("Archive size and action coverage", loc="left")
    for index, item in enumerate(ordered):
        ax_samples.text(
            item.samples,
            index,
            f"  {item.samples:,} | {item.episodes} ep | {item.distinct_actions}/{ACTION_DIM} actions",
            va="center",
            fontsize=8.5,
            color=style.INK_SECONDARY,
        )
    ax_samples.margins(x=0.42)

    groups = (
        ("Idle", [item.idle_rate for item in ordered], "#9b9b93"),
        ("Locomotion", [item.movement_rate for item in ordered], style.SERIES[0]),
        ("Dodge/dash", [item.dodge_rate for item in ordered], style.SERIES[2]),
        ("Attack", [item.attack_rate for item in ordered], style.SERIES[4]),
        ("Pickup", [item.pickup_rate for item in ordered], style.SERIES[3]),
    )
    left = np.zeros(len(ordered), dtype=np.float64)
    for name, values, colour in groups:
        values_array = np.asarray(values, dtype=np.float64)
        ax_mix.barh(y, values_array, left=left, height=0.62, label=name, color=colour)
        left += values_array
    ax_mix.set_yticks(y, labels)
    ax_mix.invert_yaxis()
    ax_mix.set_xlim(0.0, 1.0)
    ax_mix.set_xlabel("Fraction of demonstration actions")
    ax_mix.set_title("Action-family composition", loc="left")
    handles, legend_labels = ax_mix.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.72, 0.015),
        ncols=5,
    )

    fig.suptitle(
        "Current behavior-cloning demonstration quality",
        x=0.055,
        ha="left",
        fontsize=13,
        fontweight="semibold",
    )
    fig.text(
        0.055,
        0.91,
        "Recomputed from canonical phase_demos.npz archives; pilots and alternate versions are excluded.",
        ha="left",
        fontsize=9.5,
        color=style.INK_SECONDARY,
    )
    fig.tight_layout(rect=(0.03, 0.14, 1.0, 0.88))
    return _save_figure(fig, output_dir, "demo_quality")


def _metrics_rows(
    learning: dict[str, LearningSeries],
    demos: Sequence[DemoSummary],
    matrices: RetentionMatrices,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase in PRIMARY_PHASES:
        series = learning.get(phase)
        if series is None:
            continue
        values = {
            "environment_transitions": series.total_transitions,
            "training_episodes": series.episodes,
            "overall_episode_success": float(np.mean(series.raw_success)),
            "final_window_episode_success": float(series.success[-1]),
            "overall_mean_goal_error": float(np.mean(series.raw_goal_error)),
            "final_window_mean_goal_error": float(series.goal_error[-1]),
        }
        for metric, value in values.items():
            rows.append(
                {
                    "category": "training",
                    "checkpoint_phase": phase,
                    "evaluated_phase": "",
                    "metric": metric,
                    "value": value,
                    "n": series.episodes,
                    "source": series.source_path.as_posix(),
                    "status": "descriptive",
                }
            )

    for summary in demos:
        for metric in (
            "samples",
            "episodes",
            "attempts",
            "rejected_trivial",
            "distinct_actions",
            "normalized_action_entropy",
            "idle_rate",
            "movement_rate",
            "dodge_rate",
            "attack_rate",
            "pickup_rate",
            "median_episode_steps",
            "single_step_rate",
            "invalid_goal_rows",
        ):
            rows.append(
                {
                    "category": "demonstration",
                    "checkpoint_phase": summary.phase,
                    "evaluated_phase": "",
                    "metric": metric,
                    "value": getattr(summary, metric),
                    "n": summary.samples,
                    "source": summary.source_path.as_posix(),
                    "status": "measured",
                }
            )

    for row, checkpoint in enumerate(PRIMARY_PHASES):
        source = matrices.source_paths[row]
        for col, evaluated in enumerate(PRIMARY_PHASES):
            count = matrices.episodes[row, col]
            for metric, values in (
                ("skill_score", matrices.skill_score),
                ("retention", matrices.retention),
            ):
                value = values[row, col]
                if not np.isfinite(value):
                    continue
                rows.append(
                    {
                        "category": "evaluation",
                        "checkpoint_phase": checkpoint,
                        "evaluated_phase": evaluated,
                        "metric": metric,
                        "value": float(value),
                        "n": int(count),
                        "source": source.as_posix() if source else "",
                        "status": "preliminary",
                    }
                )
    return rows


def _write_metrics_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = (
        "category",
        "checkpoint_phase",
        "evaluated_phase",
        "metric",
        "value",
        "n",
        "source",
        "status",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _latex_value(value: float, count: float) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:.3f} (n={int(count)})"


def _write_latex_tables(
    path: Path,
    demos: Sequence[DemoSummary],
    matrices: RetentionMatrices,
) -> None:
    demo_by_phase = {item.phase: item for item in demos}
    row_end = r" \\"
    lines = [
        "% Generated by python -m brawl.analysis.paper_evidence; do not edit by hand.",
        "% Descriptive measurements only; not causal or multi-seed estimates.",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        "Phase & Transitions & Episodes & Actions used & Entropy/$\\log 27$" + row_end,
        r"\midrule",
    ]
    for phase in PRIMARY_PHASES:
        item = demo_by_phase.get(phase)
        if item is None:
            lines.append(f"{PHASE_LABEL[phase]} & -- & -- & -- & --" + row_end)
        else:
            lines.append(
                f"{PHASE_LABEL[phase]} & {item.samples:,} & {item.episodes} & "
                f"{item.distinct_actions}/{ACTION_DIM} & {item.normalized_action_entropy:.3f}"
                + row_end
            )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            "Checkpoint after & Movement & Weapon & Spacing & Combat" + row_end,
            r"\midrule",
        ]
    )
    for row, checkpoint in enumerate(PRIMARY_PHASES):
        cells = [
            _latex_value(matrices.skill_score[row, col], matrices.episodes[row, col])
            for col in range(len(PRIMARY_PHASES))
        ]
        lines.append(f"{PHASE_LABEL[checkpoint]} & " + " & ".join(cells) + row_end)
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _expected_cells() -> Iterable[tuple[int, int]]:
    for row in range(len(PRIMARY_PHASES)):
        for col in range(row + 1):
            yield row, col


def _readiness_issues(
    learning: dict[str, LearningSeries],
    demos: Sequence[DemoSummary],
    matrices: RetentionMatrices,
    *,
    min_eval_episodes: int,
) -> list[str]:
    issues: list[str] = []
    demos_by_phase = {item.phase: item for item in demos}
    for phase in PRIMARY_PHASES:
        if phase not in learning:
            issues.append(f"{phase} current episode log is missing")
        if phase not in demos_by_phase:
            issues.append(f"{phase} canonical demonstration archive is missing")
        elif demos_by_phase[phase].invalid_goal_rows:
            count = demos_by_phase[phase].invalid_goal_rows
            issues.append(f"{phase} demonstrations contain {count} invalid active-goal rows")

    for row, checkpoint in enumerate(PRIMARY_PHASES):
        if matrices.source_paths[row] is None:
            issues.append(f"{checkpoint} retention evaluation is missing")
            continue
        finite_counts = matrices.episodes[row, : row + 1]
        measured = finite_counts[np.isfinite(finite_counts)]
        if measured.size and np.min(measured) < min_eval_episodes:
            issues.append(
                f"{checkpoint} retention evaluation uses only {int(np.min(measured))} "
                f"evaluation episodes; minimum is {min_eval_episodes}"
            )

    for row, col in _expected_cells():
        if not np.isfinite(matrices.skill_score[row, col]):
            issues.append(
                f"missing evaluation cell: {PRIMARY_PHASES[row]} checkpoint on "
                f"{PRIMARY_PHASES[col]}"
            )

    issues.extend(
        [
            "independent training seeds are not represented in this evidence bundle",
            "matched-budget scratch baselines are missing for transfer claims",
            "mechanism ablations are missing for FiLM/replay/KL/PCGrad claims",
            "goal-independent match-play benchmark is missing",
        ]
    )
    return issues


def _write_readiness(
    path: Path,
    *,
    learning: dict[str, LearningSeries],
    demos: Sequence[DemoSummary],
    issues: Sequence[str],
    min_eval_episodes: int,
) -> None:
    lines = [
        "# Paper evidence readiness",
        "",
        "Only current canonical filenames are used. Historical 3h runs, pilots, failed runs, "
        "and alternate versions are never selected.",
        "",
        "## What can be reported now",
        "",
        "- Current demonstration composition and integrity measurements.",
        "- Within-run learning diagnostics against cumulative live-game transitions.",
        "- Preliminary checkpoint skill scores and retention with sample counts in every cell.",
        "",
        "## What cannot yet be claimed",
        "",
        "- Causal improvement from transfer, FiLM, replay, KL anchoring, or PCGrad.",
        "- Optimization uncertainty across independent training seeds.",
        "- Final goal-independent match competence.",
        "",
        f"Evaluation readiness requires at least {min_eval_episodes} episodes per checkpoint/skill cell.",
        "",
        "## Blocking evidence gaps",
        "",
    ]
    lines.extend(f"- {issue}" for issue in issues)
    lines.extend(
        [
            "",
            "## Current training logs",
            "",
            "| Phase | Transitions | Episodes | Smoothing |",
            "|---|---:|---:|---:|",
        ]
    )
    for phase in PRIMARY_PHASES:
        series = learning.get(phase)
        if series is None:
            lines.append(f"| {PHASE_LABEL[phase]} | -- | -- | -- |")
        else:
            lines.append(
                f"| {PHASE_LABEL[phase]} | {series.total_transitions:,} | "
                f"{series.episodes:,} | {series.window} episodes |"
            )
    lines.extend(
        [
            "",
            "## Canonical demonstrations",
            "",
            "| Phase | Transitions | Episodes | Actions | Invalid goal rows |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    demos_by_phase = {item.phase: item for item in demos}
    for phase in PRIMARY_PHASES:
        item = demos_by_phase.get(phase)
        if item is None:
            lines.append(f"| {PHASE_LABEL[phase]} | -- | -- | -- | -- |")
        else:
            lines.append(
                f"| {PHASE_LABEL[phase]} | {item.samples:,} | {item.episodes} | "
                f"{item.distinct_actions}/{ACTION_DIM} | {item.invalid_goal_rows} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_evidence(
    *,
    models_dir: Path = DEFAULT_MODELS_DIR,
    demos_dir: Path = DEFAULT_DEMOS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    window: int = 20,
    min_eval_episodes: int = 30,
    retention_gate: float = RETENTION_GATE,
) -> EvidenceBuildResult:
    """Generate figures, long-form metrics, tables, hashes, and evidence gaps."""
    models_dir = Path(models_dir)
    demos_dir = Path(demos_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    learning: dict[str, LearningSeries] = {}
    demos: list[DemoSummary] = []
    source_roles: dict[Path, str] = {}

    for phase in PRIMARY_PHASES:
        episode_path = models_dir / f"llc_{phase}_episodes.csv"
        if episode_path.exists():
            learning[phase] = load_learning_series(
                episode_path,
                phase=phase,
                window=window,
            )
            source_roles[episode_path] = "current_training_episodes"

        demo_path = demos_dir / f"{phase}_demos.npz"
        if demo_path.exists():
            demos.append(summarize_demo(demo_path))
            source_roles[demo_path] = "current_canonical_demonstrations"

        checkpoint_path = models_dir / f"llc_{phase}.zip"
        metadata_path = models_dir / f"llc_{phase}.zip.metadata.json"
        if checkpoint_path.exists():
            source_roles[checkpoint_path] = "current_policy_checkpoint"
        if metadata_path.exists():
            source_roles[metadata_path] = "current_checkpoint_metadata"

    matrices = build_retention_matrices(models_dir)
    for path in matrices.source_paths:
        if path is not None:
            source_roles[path] = "current_retention_evaluation"

    issues = _readiness_issues(
        learning,
        demos,
        matrices,
        min_eval_episodes=int(min_eval_episodes),
    )
    publication_ready = not issues

    outputs: list[Path] = []
    outputs.extend(_plot_demo_quality(demos, output_dir))
    outputs.extend(_plot_learning(learning, output_dir, metric="success"))
    outputs.extend(_plot_learning(learning, output_dir, metric="goal_error"))
    outputs.extend(
        _plot_matrix(
            matrices,
            output_dir,
            metric="skill_score",
            gate=retention_gate,
        )
    )
    outputs.extend(
        _plot_matrix(
            matrices,
            output_dir,
            metric="retention",
            gate=retention_gate,
        )
    )

    metrics_path = output_dir / "paper_metrics.csv"
    _write_metrics_csv(metrics_path, _metrics_rows(learning, demos, matrices))
    outputs.append(metrics_path)

    tables_path = output_dir / "paper_tables.tex"
    _write_latex_tables(tables_path, demos, matrices)
    outputs.append(tables_path)

    readiness_path = output_dir / "evidence_readiness.md"
    _write_readiness(
        readiness_path,
        learning=learning,
        demos=demos,
        issues=issues,
        min_eval_episodes=int(min_eval_episodes),
    )
    outputs.append(readiness_path)

    sources = [
        _source_record(path, role=role)
        for path, role in sorted(source_roles.items(), key=lambda item: item[0].as_posix())
    ]
    generated = [_source_record(path, role="generated_evidence") for path in outputs]
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_policy": (
            "exact current canonical filenames only; "
            "no *_3h, pilot, failed-run, or alternate-version fallback"
        ),
        "models_dir": models_dir.as_posix(),
        "demos_dir": demos_dir.as_posix(),
        "primary_phases": list(PRIMARY_PHASES),
        "smoothing_window_episodes": int(window),
        "minimum_evaluation_episodes": int(min_eval_episodes),
        "retention_gate": float(retention_gate),
        "publication_ready": bool(publication_ready),
        "issues": issues,
        "sources": sources,
        "generated_outputs": generated,
        "demo_summaries": [
            {**asdict(summary), "source_path": summary.source_path.as_posix()}
            for summary in demos
        ],
        "training_summaries": {
            phase: {
                "transitions": series.total_transitions,
                "episodes": series.episodes,
                "final_window_success": float(series.success[-1]),
                "final_window_goal_error": float(series.goal_error[-1]),
            }
            for phase, series in learning.items()
        },
    }
    manifest_path = output_dir / "evidence_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    outputs.append(manifest_path)
    return EvidenceBuildResult(
        outputs=tuple(outputs),
        issues=tuple(issues),
        publication_ready=publication_ready,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--demos-dir", type=Path, default=DEFAULT_DEMOS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--min-eval-episodes", type=int, default=30)
    parser.add_argument("--retention-gate", type=float, default=RETENTION_GATE)
    parser.add_argument(
        "--strict-readiness",
        action="store_true",
        help="Return non-zero while the generated bundle has evidence gaps",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_evidence(
        models_dir=args.models_dir,
        demos_dir=args.demos_dir,
        output_dir=args.output_dir,
        window=args.window,
        min_eval_episodes=args.min_eval_episodes,
        retention_gate=args.retention_gate,
    )
    print(f"Generated {len(result.outputs)} evidence artifacts in {args.output_dir}")
    if result.issues:
        print(f"PRELIMINARY: {len(result.issues)} evidence gap(s) remain")
        for issue in result.issues:
            print(f"  - {issue}")
    else:
        print("READY: all configured evidence gates pass")
    return 1 if args.strict_readiness and result.issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
