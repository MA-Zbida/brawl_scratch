#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from train.curriculum_goals import CURRICULUM_GOAL_FEATURES
except Exception:
    CURRICULUM_GOAL_FEATURES = [
        "signed_dx_to_ledge",
        "dy_to_ledge",
        "player_x",
        "player_y",
        "player_has_weapon",
        "weapon_dx",
        "weapon_dy",
        "rel_distance",
        "rel_dy",
        "in_strike_range",
        "opponent_damage_pct",
    ]


GOAL_GROUPS: dict[str, tuple[str, ...]] = {
    "recovery": ("signed_dx_to_ledge", "dy_to_ledge"),
    "movement": ("player_x", "player_y"),
    "weapon": ("player_has_weapon", "weapon_dx", "weapon_dy"),
    "spacing": ("rel_distance", "rel_dy"),
    "combat": ("in_strike_range", "opponent_damage_pct"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot LLC skill, retention, and anti-collapse diagnostics from CSV logs")
    p.add_argument("--steps-csv", type=str, default="", help="Path to *_steps.csv")
    p.add_argument("--episodes-csv", type=str, default="", help="Path to *_episodes.csv")
    p.add_argument("--eval-csv", type=str, default="", help="Path to *_eval.csv")
    p.add_argument("--out-dir", type=str, default="", help="Output directory (default: alongside the first input CSV)")
    p.add_argument("--prefix", type=str, default="llc", help="Output filename prefix")
    p.add_argument("--ma-window", type=int, default=25, help="Moving average window for noisy curves")
    return p.parse_args()


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _as_float(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        text = str(row.get(key, "")).strip()
        return float(text) if text else float(default)
    except Exception:
        return float(default)


def _float_series(rows: Iterable[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([_as_float(row, key) for row in rows], dtype=np.float32)


def _json_array(text: str, *, expected: int | None = None) -> np.ndarray:
    try:
        raw = json.loads(str(text or "[]"))
        arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    except Exception:
        arr = np.zeros((0,), dtype=np.float32)
    if expected is not None and arr.shape[0] != expected:
        out = np.zeros((expected,), dtype=np.float32)
        n = min(expected, arr.shape[0])
        if n > 0:
            out[:n] = arr[:n]
        return out
    return arr


def _moving_average(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    win = int(max(1, min(window, arr.size)))
    if win == 1:
        return np.arange(arr.size, dtype=np.float32), arr
    kernel = np.ones((win,), dtype=np.float32) / float(win)
    smoothed = np.convolve(arr, kernel, mode="valid")
    x = np.arange(win - 1, win - 1 + smoothed.size, dtype=np.float32)
    return x, smoothed.astype(np.float32)


def _active_error_matrix(step_rows: list[dict[str, str]]) -> np.ndarray:
    expected = len(CURRICULUM_GOAL_FEATURES)
    arrays = [_json_array(row.get("active_feature_errors", ""), expected=expected) for row in step_rows]
    if not arrays:
        return np.zeros((0, expected), dtype=np.float32)
    return np.stack(arrays).astype(np.float32)


def _feature_matrix(step_rows: list[dict[str, str]], column: str) -> np.ndarray:
    expected = len(CURRICULUM_GOAL_FEATURES)
    arrays = [_json_array(row.get(column, ""), expected=expected) for row in step_rows]
    if not arrays:
        return np.zeros((0, expected), dtype=np.float32)
    return np.stack(arrays).astype(np.float32)


def _feature_indices(names: Iterable[str]) -> list[int]:
    index = {name: i for i, name in enumerate(CURRICULUM_GOAL_FEATURES)}
    return [index[name] for name in names if name in index]


def _active_rows_for_features(mask: np.ndarray, names: Iterable[str]) -> np.ndarray:
    idxs = _feature_indices(names)
    if mask.size == 0 or not idxs:
        return np.zeros((0,), dtype=bool)
    return np.any(np.asarray(mask[:, idxs], dtype=np.float32) > 1e-6, axis=1)


def _subsample_indices(count: int, max_points: int = 2500) -> np.ndarray:
    if count <= 0:
        return np.zeros((0,), dtype=np.int64)
    if count <= max_points:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, max_points, dtype=np.int64)


def _plot_episode_health(rows: list[dict[str, str]], out_path: Path, window: int) -> None:
    if not rows:
        return
    import matplotlib.pyplot as plt

    x = np.arange(1, len(rows) + 1)
    fields = [
        ("success", "Episode Success", (0.0, 1.0)),
        ("mean_error", "Mean Goal Error", None),
        ("damage_trade", "Damage Trade", None),
        ("action_entropy", "Action Entropy", (0.0, 1.0)),
        ("idle_rate", "Idle Rate", (0.0, 1.0)),
        ("whiff_rate", "Whiff Rate", (0.0, 1.0)),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(15, 11))
    for ax, (field, title, ylim) in zip(axes.ravel(), fields):
        y = _float_series(rows, field)
        ax.plot(x, y, alpha=0.25, label=field)
        ma_x, ma_y = _moving_average(y, window)
        if ma_x.size:
            ax.plot(ma_x + 1, ma_y, linewidth=2, label=f"MA({min(window, len(y))})")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_goal_family_errors(rows: list[dict[str, str]], out_path: Path, window: int) -> None:
    errors = _active_error_matrix(rows)
    if errors.size == 0:
        return
    import matplotlib.pyplot as plt

    x = np.arange(errors.shape[0], dtype=np.float32)
    fig, axes = plt.subplots(len(GOAL_GROUPS), 1, figsize=(15, 14), sharex=True)
    for ax, (family, names) in zip(np.asarray(axes).reshape(-1), GOAL_GROUPS.items()):
        idxs = _feature_indices(names)
        if not idxs:
            continue
        y = np.mean(errors[:, idxs], axis=1)
        ax.plot(x, y, alpha=0.20, label=f"{family} active error")
        ma_x, ma_y = _moving_average(y, window)
        if ma_x.size:
            ax.plot(ma_x, ma_y, linewidth=2, label=f"MA({min(window, y.size)})")
        ax.set_title(f"{family.title()} Goal Error")
        ax.set_ylabel("error")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Step")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_goal_feature_traces(rows: list[dict[str, str]], out_path: Path, window: int) -> None:
    raw = _feature_matrix(rows, "raw_goal_feats")
    target = _feature_matrix(rows, "goal_target")
    mask = _feature_matrix(rows, "goal_mask")
    if raw.size == 0 or target.size == 0 or mask.size == 0:
        return
    import matplotlib.pyplot as plt

    errors = np.abs(raw - target) * mask
    x = np.arange(errors.shape[0], dtype=np.float32)
    fig, axes = plt.subplots(len(GOAL_GROUPS), 1, figsize=(15, 14), sharex=True)
    for ax, (family, names) in zip(np.asarray(axes).reshape(-1), GOAL_GROUPS.items()):
        plotted = False
        for name in names:
            idxs = _feature_indices((name,))
            if not idxs:
                continue
            idx = idxs[0]
            y = errors[:, idx]
            if not np.any(np.isfinite(y)) or float(np.nanmax(y)) <= 1e-8:
                continue
            ax.plot(x, y, alpha=0.15, label=f"{name} error")
            ma_x, ma_y = _moving_average(y, window)
            if ma_x.size:
                ax.plot(ma_x, ma_y, linewidth=2, label=f"{name} MA")
            plotted = True
        if not plotted:
            ax.text(0.5, 0.5, "No active samples", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(f"{family.title()} Feature Error Traces")
        ax.set_ylabel("active error")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Step")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _scatter_raw_target(ax, raw: np.ndarray, target: np.ndarray, mask: np.ndarray, names: tuple[str, str], title: str) -> None:
    idxs = _feature_indices(names)
    if len(idxs) != 2:
        ax.axis("off")
        return
    active = _active_rows_for_features(mask, names)
    if active.size == 0 or not np.any(active):
        ax.text(0.5, 0.5, "No active samples", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        return
    active_idx = np.flatnonzero(active)
    take = active_idx[_subsample_indices(active_idx.size)]
    color = np.linspace(0.0, 1.0, take.size, dtype=np.float32)
    ax.scatter(raw[take, idxs[0]], raw[take, idxs[1]], c=color, cmap="viridis", s=8, alpha=0.45, label="raw")
    ax.scatter(target[take, idxs[0]], target[take, idxs[1]], marker="x", s=18, color="tab:red", alpha=0.55, label="target")
    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")


def _plot_goal_phase_spaces(rows: list[dict[str, str]], out_path: Path) -> None:
    raw = _feature_matrix(rows, "raw_goal_feats")
    target = _feature_matrix(rows, "goal_target")
    mask = _feature_matrix(rows, "goal_mask")
    if raw.size == 0 or target.size == 0 or mask.size == 0:
        return
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    axes_flat = axes.ravel()
    _scatter_raw_target(
        axes_flat[0],
        raw,
        target,
        mask,
        ("signed_dx_to_ledge", "dy_to_ledge"),
        "Recovery Ledge Offset Phase Space",
    )
    _scatter_raw_target(
        axes_flat[1],
        raw,
        target,
        mask,
        ("player_x", "player_y"),
        "Movement Trajectory To Target",
    )
    _scatter_raw_target(
        axes_flat[2],
        raw,
        target,
        mask,
        ("weapon_dx", "weapon_dy"),
        "Weapon Offset Convergence",
    )
    _scatter_raw_target(
        axes_flat[3],
        raw,
        target,
        mask,
        ("rel_distance", "rel_dy"),
        "Spacing Distance/Vertical Band",
    )

    frame_idx = _feature_indices(("opponent_damage_pct",))
    range_idx = _feature_indices(("in_strike_range",))
    combat_active = _active_rows_for_features(mask, ("in_strike_range", "opponent_damage_pct"))
    ax = axes_flat[4]
    if combat_active.size > 0 and np.any(combat_active) and frame_idx:
        vals = raw[combat_active, frame_idx[0]]
        ax.hist(vals[np.isfinite(vals)], bins=30, color="tab:purple", alpha=0.65, label="frame advantage")
        if range_idx:
            in_range = raw[combat_active, range_idx[0]]
            ax.axvline(float(np.nanmean(in_range)), color="tab:green", linestyle="--", label="mean in_range")
        ax.set_title("Combat Frame Advantage Histogram")
        ax.set_xlabel("normalized value")
        ax.set_ylabel("count")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, "No active combat samples", transform=ax.transAxes, ha="center", va="center")
        ax.set_title("Combat Frame Advantage Histogram")

    axes_flat[5].axis("off")
    axes_flat[5].text(
        0.01,
        0.98,
        "Goal-space diagnostics\n"
        "Recovery: ledge dx/dy\n"
        "Movement: player x/y trajectory\n"
        "Weapon: weapon dx/dy convergence\n"
        "Spacing: distance/vertical band\n"
        "Combat: frame advantage distribution",
        ha="left",
        va="top",
        family="monospace",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_eval_retention(rows: list[dict[str, str]], out_path: Path) -> None:
    if not rows:
        return
    import matplotlib.pyplot as plt

    phases = sorted({str(row.get("phase", "")).strip() for row in rows if str(row.get("phase", "")).strip()})
    if not phases:
        return

    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    for phase in phases:
        phase_rows = [row for row in rows if str(row.get("phase", "")).strip() == phase]
        x = _float_series(phase_rows, "train_steps")
        axes[0].plot(x, _float_series(phase_rows, "skill_score"), marker="o", label=phase)
        axes[1].plot(x, _float_series(phase_rows, "retention"), marker="o", label=phase)
        axes[2].plot(x, _float_series(phase_rows, "amnesia"), marker="o", label=phase)

    axes[0].set_title("Skill Score By Phase")
    axes[1].set_title("Retention By Phase")
    axes[2].set_title("Amnesia By Phase")
    axes[1].axhline(0.85, linestyle="--", color="tab:green", linewidth=1, label="85% retention gate")
    axes[2].axhline(0.15, linestyle="--", color="tab:red", linewidth=1, label="15% amnesia gate")
    for ax in axes:
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Training steps")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_combat(rows: list[dict[str, str]], out_path: Path, window: int) -> None:
    if not rows:
        return
    import matplotlib.pyplot as plt

    x = np.arange(1, len(rows) + 1)
    fields = [
        ("attack_precision", "Attack Precision"),
        ("whiff_rate", "Whiff Rate"),
        ("damage_trade", "Damage Trade"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    for ax, (field, title) in zip(axes, fields):
        y = _float_series(rows, field)
        ax.plot(x, y, alpha=0.25, label=field)
        ma_x, ma_y = _moving_average(y, window)
        if ma_x.size:
            ax.plot(ma_x + 1, ma_y, linewidth=2, label=f"MA({min(window, len(y))})")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Episode")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _default_out_dir(args: argparse.Namespace) -> Path:
    if args.out_dir:
        return Path(args.out_dir)
    for candidate in (args.eval_csv, args.episodes_csv, args.steps_csv):
        if candidate:
            return Path(candidate).resolve().parent
    return Path("train/models")


def main() -> None:
    args = parse_args()
    out_dir = _default_out_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(args.prefix or "llc").strip() or "llc"
    window = int(max(1, args.ma_window))

    step_rows = _read_csv(args.steps_csv) if args.steps_csv else []
    episode_rows = _read_csv(args.episodes_csv) if args.episodes_csv else []
    eval_rows = _read_csv(args.eval_csv) if args.eval_csv else []

    made: list[Path] = []
    try:
        if episode_rows:
            path = out_dir / f"{prefix}_episode_health.png"
            _plot_episode_health(episode_rows, path, window)
            made.append(path)
            path = out_dir / f"{prefix}_combat_precision.png"
            _plot_combat(episode_rows, path, window)
            made.append(path)
        if step_rows:
            path = out_dir / f"{prefix}_goal_family_errors.png"
            _plot_goal_family_errors(step_rows, path, window)
            made.append(path)
            path = out_dir / f"{prefix}_goal_feature_traces.png"
            _plot_goal_feature_traces(step_rows, path, window)
            made.append(path)
            path = out_dir / f"{prefix}_goal_phase_spaces.png"
            _plot_goal_phase_spaces(step_rows, path)
            made.append(path)
        if eval_rows:
            path = out_dir / f"{prefix}_retention_amnesia.png"
            _plot_eval_retention(eval_rows, path)
            made.append(path)
    except ImportError as exc:
        raise SystemExit(f"matplotlib is required for plotting: {exc}") from exc

    if not made:
        print("No plots produced. Pass at least one populated CSV path.")
        return
    for path in made:
        print(path)


if __name__ == "__main__":
    main()
