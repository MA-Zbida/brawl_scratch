"""CSV loading helpers for the figure scripts.

Uses the standard library plus numpy so that figure generation adds no
dependency beyond what training already requires.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from train.phase_registry import PHASE_ORDER

MODELS_DIR = Path("train/models")
FIGURES_DIR = Path("assets/figures")

PHASES = PHASE_ORDER


def read_csv(path: Path) -> dict[str, np.ndarray]:
    """Read a CSV into a dict of columns, coercing numeric columns to float."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such run CSV: {path}")

    with path.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        raise ValueError(f"Run CSV is empty: {path}")

    columns: dict[str, np.ndarray] = {}
    for key in rows[0]:
        raw = [row.get(key, "") for row in rows]
        try:
            columns[key] = np.asarray([float(v) if v != "" else np.nan for v in raw], dtype=np.float64)
        except ValueError:
            columns[key] = np.asarray(raw, dtype=object)
    return columns


def episodes_csv(phase: str, models_dir: Path = MODELS_DIR) -> Path:
    return Path(models_dir) / f"llc_{phase}_episodes.csv"


def eval_csv(phase: str, models_dir: Path = MODELS_DIR) -> Path:
    return Path(models_dir) / f"llc_{phase}_eval.csv"


def available_phases(models_dir: Path = MODELS_DIR, kind: str = "episodes") -> list[str]:
    """Phases that actually have a run CSV on disk."""
    picker = episodes_csv if kind == "episodes" else eval_csv
    return [p for p in PHASES if picker(p, models_dir).exists()]


def moving_average(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Centred-right moving average. Returns (x, y); empty if too few points."""
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    window = int(max(1, min(window, values.size)))
    if values.size < window or values.size == 0:
        return np.array([]), np.array([])
    kernel = np.ones(window) / float(window)
    smoothed = np.convolve(values, kernel, mode="valid")
    return np.arange(window, values.size + 1), smoothed


def ensure_figures_dir(figures_dir: Path = FIGURES_DIR) -> Path:
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir
