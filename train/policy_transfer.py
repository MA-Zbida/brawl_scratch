"""Explicit transfer from a goal-conditioned skill policy to the final policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from feature_extractor.film_extractor import StageGoalFiLMExtractor
from feature_extractor.final_extractor import FinalStateExtractor


@dataclass(frozen=True)
class TransferItem:
    source: str
    target: str | None
    source_shape: tuple[int, ...]
    target_shape: tuple[int, ...] | None
    reason: str


@dataclass(frozen=True)
class TransferReport:
    transferred: tuple[TransferItem, ...]
    skipped: tuple[TransferItem, ...]

    @property
    def transferred_count(self) -> int:
        return len(self.transferred)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)

    def to_dict(self) -> dict[str, Any]:
        return {
            "transferred_count": self.transferred_count,
            "skipped_count": self.skipped_count,
            "transferred": [asdict(item) for item in self.transferred],
            "skipped": [asdict(item) for item in self.skipped],
        }

    def format(self) -> str:
        lines = [
            f"Transferred {self.transferred_count} tensors; skipped {self.skipped_count} tensors."
        ]
        for item in self.transferred:
            lines.append(
                f"  COPY {item.source} {item.source_shape} -> "
                f"{item.target} {item.target_shape}"
            )
        for item in self.skipped:
            lines.append(f"  SKIP {item.source} {item.source_shape}: {item.reason}")
        return "\n".join(lines)


_EXTRACTOR_PREFIXES = (
    "features_extractor.",
    "pi_features_extractor.",
    "vf_features_extractor.",
)


def _extractor_for(policy: Any, name: str) -> Any:
    return getattr(policy, name, None)


def _require_contract(skill_policy: Any, final_policy: Any) -> int:
    skill_extractors = {
        id(extractor): extractor
        for extractor in (
            _extractor_for(skill_policy, "features_extractor"),
            _extractor_for(skill_policy, "pi_features_extractor"),
            _extractor_for(skill_policy, "vf_features_extractor"),
        )
        if extractor is not None
    }
    final_extractors = {
        id(extractor): extractor
        for extractor in (
            _extractor_for(final_policy, "features_extractor"),
            _extractor_for(final_policy, "pi_features_extractor"),
            _extractor_for(final_policy, "vf_features_extractor"),
        )
        if extractor is not None
    }
    if not skill_extractors or any(
        not isinstance(extractor, StageGoalFiLMExtractor)
        for extractor in skill_extractors.values()
    ):
        raise TypeError("source must be a goal-conditioned skill policy with StageGoalFiLMExtractor")
    if not final_extractors or any(
        not isinstance(extractor, FinalStateExtractor)
        for extractor in final_extractors.values()
    ):
        raise TypeError("target must be a goal-independent final policy with FinalStateExtractor")

    base_dims = {int(extractor.base_dim) for extractor in skill_extractors.values()}
    input_dims = {int(extractor.input_dim) for extractor in final_extractors.values()}
    if len(base_dims) != 1 or len(input_dims) != 1 or base_dims != input_dims:
        raise ValueError(
            f"state contract mismatch: skill base dimensions={sorted(base_dims)}, "
            f"final input dimensions={sorted(input_dims)}"
        )
    return next(iter(base_dims))


def _target_key(source_key: str) -> str | None:
    for prefix in _EXTRACTOR_PREFIXES:
        if source_key.startswith(prefix):
            suffix = source_key[len(prefix) :]
            if suffix.startswith("state_encoder."):
                return source_key
            if suffix.startswith("post_film."):
                return prefix + "adapter." + suffix[len("post_film.") :]
            return None
    if source_key.startswith(("mlp_extractor.policy_net.", "action_net.")):
        return source_key
    return None


def _skip_reason(source_key: str) -> str:
    if source_key.startswith(("value_net.", "mlp_extractor.value_net.")):
        return "fresh critic: match returns differ from goal-conditioned returns"
    if any(token in source_key for token in ("goal_encoder", "film_generator", "feat_min", "feat_max")):
        return "goal-specific parameter has no final-policy equivalent"
    return "no semantically compatible final-policy parameter"


def transfer_skill_policy(skill_policy: Any, final_policy: Any) -> TransferReport:
    """Copy the state representation and actor, retaining a fresh critic.

    The first skill state layer consumes ``[state, goal-error, mask,
    alignment]``.  Only its leading state columns initialise the final layer;
    all goal-context columns are deliberately discarded.
    """
    base_dim = _require_contract(skill_policy, final_policy)
    source_state = skill_policy.state_dict()
    target_state = final_policy.state_dict()
    transferred: list[TransferItem] = []
    skipped: list[TransferItem] = []

    with torch.no_grad():
        for source_key, source_tensor in source_state.items():
            target_key = _target_key(source_key)
            if target_key is None or target_key not in target_state:
                skipped.append(
                    TransferItem(
                        source=source_key,
                        target=target_key,
                        source_shape=tuple(source_tensor.shape),
                        target_shape=(
                            tuple(target_state[target_key].shape)
                            if target_key is not None and target_key in target_state
                            else None
                        ),
                        reason=_skip_reason(source_key),
                    )
                )
                continue

            value = source_tensor
            if source_key.endswith("state_encoder.0.weight"):
                value = source_tensor[:, :base_dim]
            target_tensor = target_state[target_key]
            if tuple(value.shape) != tuple(target_tensor.shape):
                skipped.append(
                    TransferItem(
                        source=source_key,
                        target=target_key,
                        source_shape=tuple(source_tensor.shape),
                        target_shape=tuple(target_tensor.shape),
                        reason=f"shape mismatch after state projection: {tuple(value.shape)}",
                    )
                )
                continue
            target_state[target_key] = value.detach().to(
                device=target_tensor.device,
                dtype=target_tensor.dtype,
            ).clone()
            transferred.append(
                TransferItem(
                    source=source_key,
                    target=target_key,
                    source_shape=tuple(source_tensor.shape),
                    target_shape=tuple(target_tensor.shape),
                    reason="compatible state/actor/critic representation",
                )
            )

    final_policy.load_state_dict(target_state, strict=True)
    return TransferReport(tuple(transferred), tuple(skipped))
