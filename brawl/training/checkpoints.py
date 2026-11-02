"""Typed checkpoint metadata for skill-pretraining and final-agent policies."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


GOAL_CONDITIONED_SKILL = "goal_conditioned_skill"
FINAL_AGENT = "final_agent"
_VALID_POLICY_TYPES = {GOAL_CONDITIONED_SKILL, FINAL_AGENT}
_OBSOLETE_POLICY_TYPES = {"hsp", "high_level", "hierarchical_agent", "llc_hsp"}
CHECKPOINT_METADATA_SCHEMA_VERSION = 1


class CheckpointRoleError(ValueError):
    """Raised when a checkpoint is missing or violates its declared role."""


def metadata_path(checkpoint: str | Path) -> Path:
    path = Path(checkpoint)
    return path.with_name(path.name + ".metadata.json")


def make_checkpoint_metadata(
    *,
    policy_type: str,
    observation_dim: int,
    action_dim: int,
    goal_dim: int | None = None,
    initialized_from: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    role = str(policy_type).strip().lower()
    if role not in _VALID_POLICY_TYPES:
        raise CheckpointRoleError(f"unsupported policy_type={policy_type!r}")
    if int(observation_dim) <= 0 or int(action_dim) <= 0:
        raise CheckpointRoleError("observation_dim and action_dim must be positive")
    if role == GOAL_CONDITIONED_SKILL and (goal_dim is None or int(goal_dim) <= 0):
        raise CheckpointRoleError("goal-conditioned skill metadata requires goal_dim > 0")
    if role == FINAL_AGENT and goal_dim not in (None, 0):
        raise CheckpointRoleError("final-agent metadata cannot declare a goal dimension")

    payload: dict[str, Any] = {
        "schema_version": CHECKPOINT_METADATA_SCHEMA_VERSION,
        "policy_type": role,
        "goal_conditioning": role == GOAL_CONDITIONED_SKILL,
        "observation_dim": int(observation_dim),
        "action_dim": int(action_dim),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if goal_dim is not None:
        payload["goal_dim"] = int(goal_dim)
    if initialized_from:
        payload["initialized_from"] = str(initialized_from)
    if extra:
        reserved = set(payload)
        overlap = reserved.intersection(extra)
        if overlap:
            raise CheckpointRoleError(
                "extra metadata cannot replace reserved fields: " + ", ".join(sorted(overlap))
            )
        payload.update(dict(extra))
    return payload


def _validate_metadata(payload: Mapping[str, Any], source: Path) -> dict[str, Any]:
    role = str(payload.get("policy_type", "")).strip().lower()
    if role in _OBSOLETE_POLICY_TYPES:
        raise CheckpointRoleError(
            f"{source} declares obsolete policy_type={role!r}; HSP checkpoints are not supported"
        )
    if role not in _VALID_POLICY_TYPES:
        raise CheckpointRoleError(f"{source} has unknown or missing policy_type={role!r}")
    if int(payload.get("schema_version", -1)) != CHECKPOINT_METADATA_SCHEMA_VERSION:
        raise CheckpointRoleError(
            f"{source} uses unsupported checkpoint metadata schema "
            f"{payload.get('schema_version')!r}"
        )
    goal_conditioning = bool(payload.get("goal_conditioning", False))
    if goal_conditioning != (role == GOAL_CONDITIONED_SKILL):
        raise CheckpointRoleError(
            f"{source} has inconsistent goal_conditioning for policy_type={role!r}"
        )
    if int(payload.get("observation_dim", 0)) <= 0 or int(payload.get("action_dim", 0)) <= 0:
        raise CheckpointRoleError(f"{source} is missing a valid observation/action contract")
    if role == GOAL_CONDITIONED_SKILL and int(payload.get("goal_dim", 0)) <= 0:
        raise CheckpointRoleError(f"{source} is missing goal_dim for a skill policy")
    return dict(payload)


def write_checkpoint_metadata(
    checkpoint: str | Path,
    metadata: Mapping[str, Any],
) -> Path:
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    sidecar = metadata_path(checkpoint_path)
    validated = _validate_metadata(metadata, sidecar)
    sidecar.write_text(json.dumps(validated, indent=2) + "\n", encoding="utf-8")
    return sidecar


def read_checkpoint_metadata(checkpoint: str | Path) -> dict[str, Any]:
    sidecar = metadata_path(checkpoint)
    if not sidecar.exists():
        raise CheckpointRoleError(
            f"checkpoint metadata is missing: {sidecar}. "
            "Legacy untyped skill checkpoints require an explicit compatibility flag."
        )
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointRoleError(f"could not read checkpoint metadata {sidecar}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CheckpointRoleError(f"checkpoint metadata must be a JSON object: {sidecar}")
    return _validate_metadata(payload, sidecar)


def require_checkpoint_role(
    checkpoint: str | Path,
    expected_policy_type: str,
) -> dict[str, Any]:
    expected = str(expected_policy_type).strip().lower()
    if expected not in _VALID_POLICY_TYPES:
        raise CheckpointRoleError(f"unsupported expected policy role={expected!r}")
    payload = read_checkpoint_metadata(checkpoint)
    actual = str(payload["policy_type"])
    if actual != expected:
        raise CheckpointRoleError(
            f"checkpoint role mismatch: actual={actual}, expected={expected}"
        )
    return payload


def save_typed_checkpoint(model: Any, checkpoint: str | Path, metadata: Mapping[str, Any]) -> Path:
    """Save the SB3 archive and its human-readable role sidecar together."""
    path = Path(checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Embed the same payload for archive provenance; the sidecar permits role
    # validation without deserialising torch objects.
    model.checkpoint_metadata = dict(metadata)
    model.save(str(path))
    archive = path if path.suffix == ".zip" else path.with_suffix(".zip")
    write_checkpoint_metadata(archive, metadata)
    return archive


def metadata_for_skill_model(
    model: Any,
    *,
    phase: str,
    training_stage: str,
    initialized_from: str | None = None,
) -> dict[str, Any]:
    """Build role metadata from a live goal-conditioned SB3 model contract."""
    extractor = getattr(getattr(model, "policy", None), "features_extractor", None)
    goal_dim = int(getattr(extractor, "G", 0))
    feature_names = list(getattr(extractor, "goal_feature_names", []))
    observation_shape = getattr(getattr(model, "observation_space", None), "shape", None)
    action_dim = int(getattr(getattr(model, "action_space", None), "n", 0))
    if observation_shape is None or len(observation_shape) != 1:
        raise CheckpointRoleError("skill model must expose a 1D observation contract")
    if goal_dim <= 0 or len(feature_names) != goal_dim:
        raise CheckpointRoleError("skill model does not expose a valid goal-conditioned extractor")
    return make_checkpoint_metadata(
        policy_type=GOAL_CONDITIONED_SKILL,
        observation_dim=int(observation_shape[0]),
        action_dim=action_dim,
        goal_dim=goal_dim,
        initialized_from=initialized_from,
        extra={
            "training_stage": str(training_stage),
            "phase": str(phase),
            "goal_feature_names": feature_names,
        },
    )
