"""Detection-schema handling and fighter identity resolution.

Two detector schemas are supported so the pipeline keeps running while the
detector is retrained.

**Current (3-class).** ``['character', 'indicator_self', 'weapon']``. Identity is
*geometric*: the agent is the ``character`` carrying the blue self-indicator
triangle. Adding opponents costs no new classes, and the scheme stays valid when
legends are swapped, because nothing is keyed to legend appearance.

**Legacy (5-class).** ``['agent', 'op', 'op1', 'op2', 'weapons']``. Identity is
categorical, and opponent weapon state is baked into the class label.

The schema is inferred from the class names actually present in a frame, so no
flag has to be flipped when the weights are swapped.

Identity resolution has an explicit failure mode. When the indicator is missing
-- occluded, clipped at a screen edge, or simply not detected -- the agent falls
back to nearest-to-last-known-position, and ``identity_source`` records that the
answer is carried forward rather than observed. Downstream code can then treat a
stale identity as stale instead of being silently misinformed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from feature_extractor.memory.utils import bbox_center, euclidian

# ── Class names ─────────────────────────────────────────────────────────────
CHARACTER_CLASS = "character"
INDICATOR_CLASS = "indicator_self"
WEAPON_CLASS = "weapon"

LEGACY_AGENT_CLASSES = ("agent",)
LEGACY_OPPONENT_CLASSES = ("op", "op1", "op2")
LEGACY_WEAPON_CLASSES = ("weapons",)

CURRENT_CLASS_NAMES: tuple[str, ...] = (CHARACTER_CLASS, INDICATOR_CLASS, WEAPON_CLASS)
LEGACY_CLASS_NAMES: tuple[str, ...] = ("agent", "op", "op1", "op2", "weapons")

# Opponent weapon state is a legacy-only signal: it came from the op1/op2 class
# split. Under the 3-class schema it must come from the crop classifier instead.
OPPONENT_WEAPON_STATE_FROM_CLASS = {"op": 0.0, "op1": 1.0, "op2": 2.0}


# ── Indicator association tuning ────────────────────────────────────────────
# The indicator floats just above the agent's head, so in screen coordinates
# (y increasing downward) the character centre sits BELOW the indicator centre.
#
# MEASURED, not guessed: 1475 unambiguous ground-truth pairs via
# `python -m tools.calibrate_indicator_geometry`. Observed geometry is tight --
# dx spans +/-0.019 around zero, dy sits in [0.021, 0.054] -- so the triangle
# really is directly above its character in essentially every frame.
#
#   true-pair match score : p50 0.0536   p95 0.0750   p99 0.0859
#   wrong-pair score      : p01 0.0689   p05 0.0984   p50 0.2396
#
# The threshold is the true-pair p99 with a 1.15x margin. It is deliberately
# tight. The two distributions overlap slightly (wrong p01 < true p99), so no
# threshold separates them perfectly, and the failure directions are asymmetric:
# too tight merely rejects a match and falls back to carry-forward with
# `identity_observed = 0`, which the policy can see; too loose adopts the wrong
# character as the agent silently, inverting every relational feature. Widening
# to a 2.5x margin buys 0.4 points of true acceptance and admits 43% of wrong
# pairs instead of 5% -- roughly a nine-fold increase in silent identity errors.
INDICATOR_HORIZONTAL_WEIGHT = 2.0
INDICATOR_VERTICAL_WEIGHT = 1.0
INDICATOR_MAX_MATCH_SCORE = 0.0988
INDICATOR_ABOVE_TOLERANCE = 0.01


@dataclass
class FighterResolution:
    """Outcome of resolving one frame's detections into fighters."""

    agent: Optional[dict] = None
    opponents: list[dict] = field(default_factory=list)
    weapons: list[dict] = field(default_factory=list)
    schema: str = "none"            # "current" | "legacy" | "none"
    identity_source: str = "none"   # "indicator" | "carry_forward" | "legacy" | "none"
    indicator_score: float = float("inf")

    @property
    def opponent(self) -> Optional[dict]:
        """Primary opponent, for the current single-opponent observation."""
        return self.opponents[0] if self.opponents else None

    @property
    def identity_is_observed(self) -> bool:
        """True when identity came from the indicator this frame."""
        return self.identity_source in ("indicator", "legacy")


def _by_class(detections: Iterable[dict], names: Sequence[str]) -> list[dict]:
    wanted = set(names)
    return [d for d in detections if str(d.get("class_name", "")) in wanted]


def detect_schema(detections: Sequence[dict]) -> str:
    """Infer which detector schema produced these detections."""
    present = {str(d.get("class_name", "")) for d in detections}
    if present & {CHARACTER_CLASS, INDICATOR_CLASS, WEAPON_CLASS}:
        return "current"
    if present & set(LEGACY_CLASS_NAMES):
        return "legacy"
    return "none"


def match_indicator_to_character(
    characters: Sequence[dict],
    indicators: Sequence[dict],
    *,
    max_score: float = INDICATOR_MAX_MATCH_SCORE,
) -> tuple[Optional[dict], float]:
    """Pick the character carrying the self-indicator.

    Returns ``(character, score)``; ``(None, inf)`` when nothing matches well
    enough. Lower scores are better.
    """
    if not characters or not indicators:
        return None, float("inf")

    indicator = max(indicators, key=lambda d: float(d.get("confidence", 0.0)))
    ix, iy = bbox_center(indicator)

    best: Optional[dict] = None
    best_score = float("inf")

    for character in characters:
        cx, cy = bbox_center(character)
        vertical = cy - iy
        # The indicator sits above the head; a character above it is not the one.
        if vertical < -INDICATOR_ABOVE_TOLERANCE:
            continue
        score = (
            INDICATOR_HORIZONTAL_WEIGHT * abs(cx - ix)
            + INDICATOR_VERTICAL_WEIGHT * max(0.0, vertical)
        )
        if score < best_score:
            best_score = score
            best = character

    if best is None or best_score > max_score:
        return None, float("inf")
    return best, best_score


def _nearest(detections: Sequence[dict], xy: Optional[tuple[float, float]]) -> Optional[dict]:
    if not detections:
        return None
    if xy is None:
        return max(detections, key=lambda d: float(d.get("confidence", 0.0)))
    return min(
        detections,
        key=lambda d: (euclidian(bbox_center(d), xy), -float(d.get("confidence", 0.0))),
    )


def resolve(
    detections: Sequence[dict],
    *,
    last_agent_xy: Optional[tuple[float, float]] = None,
    last_opponent_xy: Optional[tuple[float, float]] = None,
) -> FighterResolution:
    """Resolve one frame of detections into agent, opponents and weapons."""
    detections = list(detections or [])
    schema = detect_schema(detections)

    if schema == "legacy":
        agent = _nearest(_by_class(detections, LEGACY_AGENT_CLASSES), last_agent_xy)
        opponents = _by_class(detections, LEGACY_OPPONENT_CLASSES)
        primary = _nearest(opponents, last_opponent_xy)
        ordered = ([primary] + [o for o in opponents if o is not primary]) if primary else []
        return FighterResolution(
            agent=agent,
            opponents=ordered,
            weapons=_by_class(detections, LEGACY_WEAPON_CLASSES),
            schema="legacy",
            identity_source="legacy" if agent is not None else "none",
        )

    if schema == "none":
        return FighterResolution()

    characters = _by_class(detections, (CHARACTER_CLASS,))
    indicators = _by_class(detections, (INDICATOR_CLASS,))
    weapons = _by_class(detections, (WEAPON_CLASS,))

    agent, score = match_indicator_to_character(characters, indicators)
    identity_source = "indicator"

    if agent is None:
        # Indicator missing or unmatched: carry the previous identity forward and
        # mark the answer as unobserved.
        agent = _nearest(characters, last_agent_xy) if last_agent_xy is not None else None
        identity_source = "carry_forward" if agent is not None else "none"
        score = float("inf")

    remaining = [c for c in characters if c is not agent]
    primary = _nearest(remaining, last_opponent_xy)
    ordered = ([primary] + [o for o in remaining if o is not primary]) if primary else []

    return FighterResolution(
        agent=agent,
        opponents=ordered,
        weapons=weapons,
        schema="current",
        identity_source=identity_source,
        indicator_score=score,
    )
