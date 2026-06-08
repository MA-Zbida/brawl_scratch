from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


PHASE_ORDER: tuple[str, ...] = (
    "recovery_mastery",
    "movement_fluency",
    "weapon_acquisition",
    "spacing_neutral",
    "combat_execution",
    "all_skills_llc",
)

PHASE_SCORE_THRESHOLDS: dict[str, float] = {
    "recovery_mastery": 0.65,
    "movement_fluency": 0.65,
    "weapon_acquisition": 0.60,
    "spacing_neutral": 0.55,
    "combat_execution": 0.50,
    "all_skills_llc": 0.60,
}


def phase_score_threshold(phase: str, override: float | None = None) -> float:
    if override is not None:
        return float(max(0.0, override))
    return float(PHASE_SCORE_THRESHOLDS.get(str(phase).strip().lower(), 0.0))


def previous_phases(phase: str, *, include_current: bool = True) -> list[str]:
    phase = str(phase).strip().lower()
    if phase == "all_skills_llc":
        end = len(PHASE_ORDER) if include_current else len(PHASE_ORDER) - 1
        return list(PHASE_ORDER[:end])
    if phase not in PHASE_ORDER:
        return [phase]
    idx = PHASE_ORDER.index(phase)
    end = idx + 1 if include_current else idx
    return list(PHASE_ORDER[:end])


def parse_phase_list(raw: str | None, current_phase: str, *, include_previous: bool) -> list[str]:
    text = "" if raw is None else str(raw).strip()
    if text.lower() in ("all", "*"):
        return list(PHASE_ORDER)
    if text:
        phases = [part.strip().lower() for part in text.replace(";", ",").split(",") if part.strip()]
        return list(dict.fromkeys(phases))
    if include_previous:
        return previous_phases(current_phase, include_current=True)
    return [str(current_phase).strip().lower()]


def _bounded_unit(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (float(value) - low) / (high - low)))


def skill_score_for_phase(phase: str, summary: Mapping[str, Any]) -> float:
    phase = str(phase).strip().lower()
    episode_success = _bounded_unit(float(summary.get("episode_success_rate", 0.0)))
    step_success = _bounded_unit(float(summary.get("mean_goal_success", 0.0)))
    win_rate = _bounded_unit(float(summary.get("win_rate", 0.0)))
    hit_rate = _bounded_unit(float(summary.get("hit_rate", 0.0)))
    pickup_rate = _bounded_unit(float(summary.get("weapon_pickup_rate", 0.0)))
    damage_trade = _bounded_unit(float(summary.get("mean_damage_trade", 0.0)), -0.05, 0.25)
    low_error = 1.0 - _bounded_unit(float(summary.get("mean_goal_error", 1.0)), 0.0, 0.25)

    if phase == "combat_execution":
        score = (0.30 * episode_success) + (0.25 * hit_rate) + (0.25 * damage_trade) + (0.20 * win_rate)
    elif phase == "weapon_acquisition":
        score = (0.45 * episode_success) + (0.35 * pickup_rate) + (0.20 * low_error)
    elif phase == "all_skills_llc":
        score = (0.35 * episode_success) + (0.25 * step_success) + (0.20 * low_error) + (0.20 * damage_trade)
    else:
        score = (0.60 * episode_success) + (0.20 * step_success) + (0.20 * low_error)
    return float(max(0.0, min(1.0, score)))


def retention_and_amnesia(current_score: float, best_score: float) -> tuple[float, float]:
    current = max(0.0, float(current_score))
    best = max(1e-8, float(best_score))
    retention = max(0.0, min(1.0, current / best))
    amnesia = max(0.0, 1.0 - retention)
    return float(retention), float(amnesia)


def update_best_scores(
    best_scores: Mapping[str, float],
    current_scores: Mapping[str, float],
) -> dict[str, float]:
    updated = {str(k): float(v) for k, v in best_scores.items()}
    for phase, score in current_scores.items():
        key = str(phase)
        updated[key] = max(float(score), float(updated.get(key, 0.0)))
    return updated


def load_best_scores(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    raw = data.get("best_scores", data) if isinstance(data, dict) else {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): float(v) for k, v in raw.items()}


def save_best_scores(path: Path, best_scores: Mapping[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"best_scores": {str(k): float(v) for k, v in best_scores.items()}}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
