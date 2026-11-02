#!/usr/bin/env python
"""Exercise the live LLC pipeline contracts before spending gameplay hours."""

from __future__ import annotations

# Import order is load-bearing on hybrid-GPU laptops.  Standard-library imports
# are safe, but capture_first must acquire DXGI before SB3/torch/Ultralytics load.
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import brawl.capture_first  # noqa: E402,F401

import argparse
import gc
import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
from brawl.compat import load_sb3
from brawl.paths import LEGACY_CONFIGS


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = LEGACY_CONFIGS / "llc" / "easy_bot_v0" / "config.json"


class PreflightFailure(RuntimeError):
    """A fail-closed preflight error that identifies its pipeline stage."""

    def __init__(self, stage: str, detail: str) -> None:
        self.stage = str(stage)
        self.detail = str(detail)
        super().__init__(f"PREFLIGHT FAIL [{self.stage}]: {self.detail}")


def _resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def validate_allowed_actions(
    phases: Sequence[str],
    spec_builder: Callable[[str], Any],
    *,
    action_dim: int,
) -> None:
    """Validate both explicit and effective action allowlists for every phase."""
    from brawl.pipelines.llc.phase_action_policy import phase_allowed_action_ids

    for phase in phases:
        stage = f"allowed_actions:{phase}"
        try:
            spec = spec_builder(phase)
            explicit = getattr(spec, "allowed_actions", None)
            if explicit is not None:
                raw = tuple(int(action) for action in explicit)
                if not raw:
                    raise PreflightFailure(stage, "allowed_actions must be non-empty")
                invalid = sorted({action for action in raw if action < 0 or action >= action_dim})
                if invalid:
                    raise PreflightFailure(
                        stage,
                        f"action ids {invalid} fall outside [0, {action_dim - 1}]",
                    )
            effective = tuple(int(action) for action in phase_allowed_action_ids(spec))
            if not effective:
                raise PreflightFailure(stage, "effective allowed_actions must be non-empty")
            invalid = sorted(
                {action for action in effective if action < 0 or action >= action_dim}
            )
            if invalid:
                raise PreflightFailure(
                    stage,
                    f"effective action ids {invalid} fall outside [0, {action_dim - 1}]",
                )
        except PreflightFailure:
            raise
        except Exception as exc:
            raise PreflightFailure(stage, str(exc)) from exc


def _close_environment(env: Any, stage: str) -> None:
    try:
        env.close()
    except Exception as exc:
        raise PreflightFailure(stage, f"close failed: {exc}") from exc


def exercise_environment_matrix(
    phases: Sequence[str],
    *,
    classifier_modes: Sequence[bool],
    environment_factory: Callable[[str, bool], Any],
) -> None:
    """Construct and close the exact base/wrapper pair for each phase/mode."""
    for phase in phases:
        for classifier_enabled in classifier_modes:
            mode = "on" if classifier_enabled else "off"
            stage = f"environment:{phase}:classifier={mode}"
            env = None
            try:
                env = environment_factory(phase, bool(classifier_enabled))
            except Exception as exc:
                raise PreflightFailure(stage, str(exc)) from exc
            finally:
                if env is not None:
                    _close_environment(env, stage)
                    del env
                    gc.collect()


def exercise_sequential_lifecycle(
    environment_factory: Callable[[], Any],
    *,
    count: int = 3,
) -> None:
    """Reproduce retention's create/close/create lifecycle in one process."""
    for index in range(max(1, int(count))):
        stage = f"sequential_environment:{index + 1}/{count}"
        env = None
        try:
            env = environment_factory()
        except Exception as exc:
            raise PreflightFailure(stage, str(exc)) from exc
        finally:
            if env is not None:
                _close_environment(env, stage)
                del env
                gc.collect()


def _read_checkpoint_metadata(path: Path) -> dict[str, Any]:
    from brawl.training.checkpoints import read_checkpoint_metadata

    return read_checkpoint_metadata(path)


def _load_skill_checkpoint(path: Path) -> Any:
    from stable_baselines3 import PPO

    # The policies are MLPs and deployment is CPU-only. Loading them on CUDA in
    # preflight would add contention with TensorRT without testing the real path.
    return load_sb3(PPO, str(path), device="cpu")


def validate_checkpoint_references(
    references: Iterable[Mapping[str, Any]],
    *,
    repo_root: Path,
    checkpoint_loader: Callable[[Path], Any] = _load_skill_checkpoint,
) -> None:
    """Deserialize each required checkpoint and compare archive/sidecar contracts."""
    for reference in references:
        name = str(reference.get("name", "unnamed"))
        stage = f"checkpoint:{name}"
        path = _resolve(repo_root, str(reference.get("path", "")))
        if not path.is_file():
            raise PreflightFailure(stage, f"missing checkpoint: {path}")
        try:
            metadata = _read_checkpoint_metadata(path)
            expected_role = str(reference.get("role", "goal_conditioned_skill"))
            actual_role = str(metadata.get("policy_type", ""))
            if actual_role != expected_role:
                raise PreflightFailure(
                    stage,
                    f"role mismatch: expected {expected_role}, got {actual_role}",
                )
            expected_obs = int(reference["observation_dim"])
            if int(metadata.get("observation_dim", -1)) != expected_obs:
                raise PreflightFailure(
                    stage,
                    "metadata observation_dim mismatch: "
                    f"expected {expected_obs}, got {metadata.get('observation_dim')}",
                )
            model = checkpoint_loader(path)
            model_shape = getattr(getattr(model, "observation_space", None), "shape", None)
            if model_shape != (expected_obs,):
                raise PreflightFailure(
                    stage,
                    f"archive observation shape mismatch: expected {(expected_obs,)}, got {model_shape}",
                )
            expected_actions = int(reference.get("action_dim", 27))
            actual_actions = int(getattr(getattr(model, "action_space", None), "n", -1))
            if actual_actions != expected_actions:
                raise PreflightFailure(
                    stage,
                    f"archive action_dim mismatch: expected {expected_actions}, got {actual_actions}",
                )
            del model
            gc.collect()
        except PreflightFailure:
            raise
        except Exception as exc:
            raise PreflightFailure(stage, str(exc)) from exc


def validate_demo_references(
    references: Iterable[Mapping[str, Any]],
    *,
    repo_root: Path,
    demo_validator: Callable[..., dict[str, Any]],
) -> None:
    """Run the same archive validator that gates BC collection."""
    for reference in references:
        phase = str(reference.get("phase", "unknown"))
        stage = f"demo:{phase}"
        path = _resolve(repo_root, str(reference.get("path", "")))
        if not path.is_file():
            raise PreflightFailure(stage, f"missing demo archive: {path}")
        try:
            result = demo_validator(
                path,
                expected_phase=phase,
                min_samples=int(reference.get("min_samples", 100)),
            )
        except Exception as exc:
            raise PreflightFailure(stage, str(exc)) from exc
        if str(result.get("status", "FAIL")) != "PASS":
            details = [*result.get("errors", []), *result.get("warnings", [])]
            raise PreflightFailure(stage, "; ".join(map(str, details)) or "validator failed")


def _checkpoint_strings(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for child in value.values():
            found.update(_checkpoint_strings(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_checkpoint_strings(child))
    elif isinstance(value, str) and value.lower().endswith(".zip"):
        found.add(value.replace("\\", "/"))
    return found


def _validate_checkpoint_reference_coverage(config: Mapping[str, Any]) -> None:
    pipeline = config.get("pipeline", {})
    required = {
        str(item["path"]).replace("\\", "/")
        for item in pipeline.get("required_checkpoints", [])
    }
    deferred = {
        str(path).replace("\\", "/")
        for path in pipeline.get("deferred_checkpoint_outputs", [])
    }
    uncovered = sorted(_checkpoint_strings(config) - required - deferred)
    if uncovered:
        raise PreflightFailure(
            "checkpoint_references",
            "checkpoint paths are neither required nor explicitly deferred: "
            + ", ".join(uncovered),
        )


def _real_environment_factory(
    config: Mapping[str, Any],
) -> Callable[[str, bool], Any]:
    from brawl.env.live import BrawlDeepEnv, EnvConfig
    from brawl.pipelines.llc.curriculum_config import build_phase_spec
    from brawl.pipelines.llc.stage_common import StageGoalEnv

    perception = config.get("perception", {})

    def build(phase: str, classifier_enabled: bool) -> Any:
        base = None
        try:
            env_config = EnvConfig(
                ui_regions=None,
                yolo_infer_every_n_steps=max(
                    1, int(perception.get("yolo_infer_every_n_steps", 1))
                ),
                weapon_classifier_enabled=bool(classifier_enabled),
                weapon_classifier_checkpoint=str(
                    _resolve(
                        REPO_ROOT,
                        perception.get(
                            "weapon_classifier_checkpoint",
                            "data/models/weapon_classifier_r18.pt",
                        ),
                    )
                ),
                weapon_classifier_interval=max(
                    1, int(perception.get("weapon_classifier_interval", 4))
                ),
                weapon_classifier_min_confidence=float(
                    perception.get("weapon_classifier_min_confidence", 0.7)
                ),
            )
            base = BrawlDeepEnv(config=env_config)
            spec = build_phase_spec(phase, terminate_on_death=False)
            return StageGoalEnv(base, spec)
        except Exception:
            if base is not None:
                base.close()
            raise

    return build


def run_preflight(config_path: Path, *, repo_root: Path = REPO_ROOT) -> None:
    from brawl.control.action_space import ACTION_DIM
    from scripts.llc.validate_llc_demos import validate_demo_archive
    from brawl.pipelines.llc.curriculum_config import PHASES, build_phase_spec

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PreflightFailure("config", f"could not read {config_path}: {exc}") from exc
    pipeline = config.get("pipeline")
    if not isinstance(pipeline, Mapping):
        raise PreflightFailure("config", "missing pipeline contract")
    if bool(config.get("perception", {}).get("weapon_classifier_enabled", False)):
        raise PreflightFailure(
            "config",
            "weapon classifier must remain disabled for live training until retrained on live crops",
        )

    _validate_checkpoint_reference_coverage(config)
    validate_allowed_actions(PHASES, build_phase_spec, action_dim=ACTION_DIM)
    validate_checkpoint_references(
        pipeline.get("required_checkpoints", []),
        repo_root=repo_root,
    )
    validate_demo_references(
        pipeline.get("demo_archives", []),
        repo_root=repo_root,
        demo_validator=validate_demo_archive,
    )

    factory = _real_environment_factory(config)
    exercise_environment_matrix(
        PHASES,
        classifier_modes=(False, True),
        environment_factory=factory,
    )
    # This explicit second pass makes the camera-singleton regression impossible
    # to hide inside a phase/mode matrix whose ordering later gets refactored.
    exercise_sequential_lifecycle(
        lambda: factory("movement_fluency", False),
        count=3,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-fast lifecycle/data/checkpoint preflight for the Easy-bot LLC chain"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    try:
        run_preflight(args.config.resolve())
    except PreflightFailure as exc:
        print(str(exc), file=_sys.stderr)
        raise SystemExit(2) from exc
    elapsed = time.perf_counter() - start
    print(f"PREFLIGHT PASS ({elapsed:.1f}s): all pipeline contracts and lifecycles are usable")


if __name__ == "__main__":
    main()
