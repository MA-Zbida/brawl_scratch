#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

# Running this file directly (`python tools/validate_llc_demos.py`) puts tools/
# on sys.path, not the repo root, so the bootstrap has to precede every
# first-party import -- not just the train.* ones.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from action_space import ACTION_DIM, components  # noqa: E402  (needs the path above)
from train.retention import PHASE_ORDER  # noqa: E402


DEMO_PHASES = tuple(phase for phase in PHASE_ORDER if phase != "all_skills_llc")
DEMO_FILENAMES: dict[str, str] = {phase: f"{phase}_demos.npz" for phase in DEMO_PHASES}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate LLC behavioral-cloning demo archives before training")
    p.add_argument("paths", nargs="*", help="NPZ demo paths. Semicolon/comma separated values are accepted.")
    p.add_argument("--phase", type=str, default="", choices=["", "all", *DEMO_PHASES], help="Expected phase or all default demo archives")
    p.add_argument("--models-dir", type=str, default="train/models")
    p.add_argument("--min-samples", type=int, default=100)
    p.add_argument("--min-action-entropy", type=float, default=0.02)
    p.add_argument("--max-idle-rate", type=float, default=0.60)
    p.add_argument("--strict-warnings", action="store_true", help="Treat warning-level demo issues as failures")
    return p.parse_args()


def split_paths(raw: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in raw:
        for part in str(item).replace(",", ";").split(";"):
            text = part.strip().strip('"')
            if text:
                paths.append(Path(text))
    return paths


def default_paths_for_phase(phase: str, models_dir: str) -> list[Path]:
    key = str(phase).strip().lower()
    if key == "all":
        return [Path(models_dir) / DEMO_FILENAMES[p] for p in DEMO_PHASES]
    if key in DEMO_FILENAMES:
        return [Path(models_dir) / DEMO_FILENAMES[key]]
    return []


def _metadata_scalar(data: Any, key: str, default: str = "") -> str:
    if key not in data.files:
        return str(default)
    try:
        arr = np.asarray(data[key]).reshape(-1)
        if arr.size <= 0:
            return str(default)
        return str(arr[0])
    except Exception:
        return str(default)


def _metadata_int(data: Any, key: str, default: int = 0) -> int:
    try:
        return int(float(_metadata_scalar(data, key, str(default))))
    except Exception:
        return int(default)


def _load_actions(data: Any) -> np.ndarray | None:
    """Return the stored actions with their shape intact.

    Shape is the evidence that distinguishes the current Discrete(27) encoding
    (N,) from the retired MultiDiscrete([4,2,2,4]) one (N,4), so this must not
    normalise it away -- an earlier version reshaped (N,) to (N,1) and the
    discrete check downstream then rejected every archive, valid ones included.
    """
    if "actions_discrete" in data.files:
        return np.asarray(data["actions_discrete"], dtype=np.int64)
    if "actions" in data.files:
        return np.asarray(data["actions"], dtype=np.int64)
    return None


def action_entropy(actions: np.ndarray) -> float:
    arr = np.asarray(actions, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[0] <= 0:
        return 0.0
    counts = Counter(tuple(row.tolist()) for row in arr)
    total = float(sum(counts.values()))
    if total <= 0.0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = float(count) / total
        entropy -= p * math.log(max(p, 1e-12))
    denom = math.log(max(2, min(64, len(counts) if len(counts) > 1 else 2)))
    return float(max(0.0, min(1.0, entropy / denom)))


def _component_tables() -> tuple[np.ndarray, np.ndarray]:
    """Per-action-id lookup: is this action idle, and is it an attack.

    Built from `action_space.components` so the meaning of an id lives in one
    place. Indexing a table beats decoding per sample -- an archive is tens of
    thousands of rows, the space is 27 actions.
    """
    idle = np.zeros(ACTION_DIM, dtype=bool)
    attack = np.zeros(ACTION_DIM, dtype=bool)
    for action_id in range(ACTION_DIM):
        comp = components(action_id)
        idle[action_id] = not any(
            (comp.hdir, comp.vdir, comp.jump, comp.dodge, comp.light, comp.heavy, comp.interact)
        )
        attack[action_id] = bool(comp.light or comp.heavy)
    return idle, attack


_IS_IDLE, _IS_ATTACK = _component_tables()


def _rate(actions: np.ndarray, table: np.ndarray) -> float:
    arr = np.asarray(actions, dtype=np.int64).reshape(-1)
    if arr.size <= 0:
        return 0.0
    valid = (arr >= 0) & (arr < ACTION_DIM)
    if not np.any(valid):
        return 0.0
    return float(np.mean(table[arr[valid]].astype(np.float32)))


def idle_rate(actions: np.ndarray) -> float:
    return _rate(actions, _IS_IDLE)


def attack_rate(actions: np.ndarray) -> float:
    return _rate(actions, _IS_ATTACK)


def _goal_active_ratio(data: Any, n: int) -> float:
    if "goal_mask" not in data.files:
        return 0.0
    try:
        mask = np.asarray(data["goal_mask"], dtype=np.float32)
        if mask.ndim != 2 or mask.shape[0] <= 0:
            return 0.0
        active = np.any(mask[:n] > 1e-6, axis=1)
        return float(np.mean(active.astype(np.float32)))
    except Exception:
        return 0.0


def validate_demo_archive(
    path: Path,
    *,
    expected_phase: str = "",
    min_samples: int = 100,
    min_action_entropy: float = 0.02,
    max_idle_rate: float = 0.60,
    strict_warnings: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "phase": "",
        "samples": 0,
        "obs_dim": 0,
        "action_dim": 0,
        "action_entropy": 0.0,
        "idle_rate": 0.0,
        "attack_rate": 0.0,
        "goal_active_ratio": 0.0,
        "episodes_collected": 0,
        "status": "PASS",
        "errors": [],
        "warnings": [],
    }

    errors: list[str] = result["errors"]
    warnings: list[str] = result["warnings"]
    if not path.exists():
        errors.append("file not found")
        result["status"] = "FAIL"
        return result

    try:
        with np.load(path, allow_pickle=False) as data:
            if "obs" not in data.files:
                errors.append("missing obs array")
                result["status"] = "FAIL"
                return result

            obs = np.asarray(data["obs"], dtype=np.float32)
            actions = _load_actions(data)
            if actions is None:
                errors.append("missing actions/actions_multidiscrete array")
                result["status"] = "FAIL"
                return result

            if obs.ndim != 2:
                errors.append(f"obs must be 2D [N,D], got {obs.shape}")
                result["status"] = "FAIL"
                return result
            if actions.ndim not in (1, 2):
                errors.append(f"actions must be [N] (discrete) or [N,A] (legacy), got {actions.shape}")
                result["status"] = "FAIL"
                return result

            n = int(min(obs.shape[0], actions.shape[0]))
            phase = _metadata_scalar(data, "phase", path.stem.replace("_demos", "")).strip().lower()
            result["phase"] = phase
            result["samples"] = n
            result["obs_dim"] = int(obs.shape[1])
            result["action_dim"] = ACTION_DIM if actions.ndim == 1 else int(actions.shape[1])
            result["action_entropy"] = action_entropy(actions[:n])
            result["idle_rate"] = idle_rate(actions[:n])
            result["attack_rate"] = attack_rate(actions[:n])
            result["goal_active_ratio"] = _goal_active_ratio(data, n)
            result["episodes_collected"] = _metadata_int(data, "episodes_collected", 0)

            if expected_phase and phase != str(expected_phase).strip().lower():
                errors.append(f"phase metadata mismatch: expected {expected_phase}, got {phase or 'unknown'}")
            if n < int(min_samples):
                errors.append(f"too few samples: {n} < {int(min_samples)}")
            if obs.shape[0] != actions.shape[0]:
                errors.append(f"obs/actions length mismatch: {obs.shape[0]} vs {actions.shape[0]}")
            if actions.ndim != 1:
                errors.append(
                    f"expected discrete actions of shape (N,), got {actions.shape}. "
                    "Archives from the old MultiDiscrete([4,2,2,4]) encoding are not "
                    "convertible -- that space could not express direction-modified "
                    "attacks at all. Recollect."
                )
            elif actions.size and (actions.max() >= ACTION_DIM or actions.min() < 0):
                errors.append(
                    f"actions outside the {ACTION_DIM}-action space "
                    f"(range {actions.min()}..{actions.max()})"
                )
            if not np.all(np.isfinite(obs[:n])):
                errors.append("obs contains NaN/Inf")
            if result["goal_active_ratio"] <= 0.0:
                warnings.append("goal_mask missing or never active")
            if result["action_entropy"] < float(min_action_entropy):
                warnings.append(
                    f"low action entropy: {float(result['action_entropy']):.3f} < {float(min_action_entropy):.3f}"
                )
            if result["idle_rate"] > float(max_idle_rate):
                warnings.append(f"high idle rate: {float(result['idle_rate']):.3f} > {float(max_idle_rate):.3f}")

            if phase == "combat_execution" and result["attack_rate"] <= 0.0:
                warnings.append("combat demo has zero attack inputs")
            if phase == "recovery_mastery" and "sequential_terminal_success" in data.files:
                seq_success = np.asarray(data["sequential_terminal_success"], dtype=bool)
                if seq_success.size > 0 and _metadata_int(data, "recovery_sequence_enforced", 0) > 0:
                    ratio = float(np.mean(seq_success[:n].astype(np.float32)))
                    if ratio <= 0.0:
                        warnings.append("recovery sequence gate was enabled but no terminal sequence successes were saved")
    except Exception as exc:
        errors.append(f"could not read npz: {exc}")

    if errors or (strict_warnings and warnings):
        result["status"] = "FAIL"
    elif warnings:
        result["status"] = "WARN"
    else:
        result["status"] = "PASS"
    return result


def _print_table(results: list[dict[str, Any]]) -> None:
    header = "status phase                samples obs action entropy idle   attack goal   episodes path"
    print(header)
    print("-" * len(header))
    for item in results:
        print(
            f"{str(item['status']):6s} "
            f"{str(item['phase'] or 'unknown'):20s} "
            f"{int(item['samples']):7d} "
            f"{int(item['obs_dim']):3d} "
            f"{int(item['action_dim']):6d} "
            f"{float(item['action_entropy']):7.3f} "
            f"{float(item['idle_rate']):5.3f} "
            f"{float(item['attack_rate']):6.3f} "
            f"{float(item['goal_active_ratio']):5.3f} "
            f"{int(item['episodes_collected']):8d} "
            f"{item['path']}"
        )
        for err in item["errors"]:
            print(f"  ERROR: {err}")
        for warning in item["warnings"]:
            print(f"  WARN : {warning}")


def main() -> None:
    args = parse_args()
    paths = split_paths(args.paths)
    if not paths:
        paths = default_paths_for_phase(str(args.phase), str(args.models_dir))
    if not paths:
        raise SystemExit("Pass demo paths or --phase {recovery_mastery|movement_fluency|weapon_acquisition|spacing_neutral|combat_execution|all}")

    results: list[dict[str, Any]] = []
    for path in paths:
        expected = ""
        if str(args.phase).strip().lower() in DEMO_PHASES and len(paths) == 1:
            expected = str(args.phase).strip().lower()
        results.append(
            validate_demo_archive(
                path,
                expected_phase=expected,
                min_samples=int(args.min_samples),
                min_action_entropy=float(args.min_action_entropy),
                max_idle_rate=float(args.max_idle_rate),
                strict_warnings=bool(args.strict_warnings),
            )
        )

    _print_table(results)
    failed = any(str(item["status"]) == "FAIL" for item in results)
    warned = any(str(item["status"]) == "WARN" for item in results)
    if failed:
        print("STOP: fix or recollect failed demo archives before BC pretraining.")
        raise SystemExit(2)
    if warned:
        print("CHECK: demos loaded, but warnings indicate possible weak BC anchors.")
        raise SystemExit(1)
    print("PASS: demo archives look usable for BC anchoring.")


if __name__ == "__main__":
    main()

