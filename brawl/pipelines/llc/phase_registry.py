"""Canonical ordering for LLC training, retention, and demo tooling."""

from __future__ import annotations


# Recovery remains a required motor skill, but it is deliberately deferred until
# the four phases that can be collected reliably from a normal spawn have produced
# the first end-to-end gameplay checkpoint.
PHASE_ORDER: tuple[str, ...] = (
    "movement_fluency",
    "weapon_acquisition",
    "spacing_neutral",
    "combat_execution",
    "recovery_mastery",
    "all_skills_llc",
)
