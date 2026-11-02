"""Bounded, confidence-aware weapon evidence shared by reward and collection.

No image/model imports: this contract must remain usable in headless collectors.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

DEFAULT_MIN_CONFIDENCE = .70
EVIDENCE_VERSION = "fresh_confident_weapon_v2"


class WeaponEvidence:
    def __init__(self, *, min_confidence: float = DEFAULT_MIN_CONFIDENCE,
                 max_age_steps: int = 8) -> None:
        if not 0 <= min_confidence <= 1 or max_age_steps < 0:
            raise ValueError("Invalid weapon evidence confidence/age limit")
        self.min_confidence = min_confidence
        # Two default classifier periods; missing crops cannot certify an endless hold.
        self.max_age_steps = max_age_steps
        self.reset()

    def reset(self) -> None:
        self.value: bool | None = None
        self.fresh_value: bool | None = None
        self.age_steps = self.max_age_steps + 1

    def update(self, info: Mapping[str, object]) -> bool | None:
        self.age_steps += 1
        self.fresh_value = None
        confidence = float(info.get("player_weapon_visual_confidence", 0.))
        if (float(info.get("player_weapon_visual_fresh", 0.)) > .5
                and float(info.get("player_weapon_visual_valid", 0.)) > .5
                and math.isfinite(confidence)
                and self.min_confidence <= confidence <= 1.):
            self.fresh_value = float(info.get("player_weapon_visual_armed", 0.)) > .5
            self.value = self.fresh_value
            self.age_steps = 0
        if self.age_steps > self.max_age_steps:
            self.value = None
        return self.value


class WeaponHoldGate:
    def __init__(self, *, required_steps: int = 40, drop_grace: int = 3,
                 require_visual: bool = True, evidence: WeaponEvidence | None = None) -> None:
        if required_steps < 1 or drop_grace < 0:
            raise ValueError("Hold steps must be positive and drop grace nonnegative")
        self.required_steps = required_steps
        self.drop_grace = drop_grace
        self.require_visual = require_visual
        self.evidence = evidence or WeaponEvidence()
        self.pickup_seen = False
        self.positive_refreshes = 0
        self.hold_steps = 0
        self.negative_refreshes = 0
        self.dropped = False
        self.success = False

    def update(self, has_weapon: bool, info: Mapping[str, object]) -> None:
        if self.pickup_seen and float(info.get('weapon_drop_action', 0.)) > .5:
            # A newer executed throw outranks the pre-input crop in this step.
            self.dropped = True
            self.success = False
            self.hold_steps = 0
            return
        visual = self.evidence.update(info) if self.require_visual else has_weapon
        fresh = self.evidence.fresh_value if self.require_visual else has_weapon
        if fresh is True:
            self.positive_refreshes += 1
        elif fresh is False or visual is None:
            self.positive_refreshes = 0
        # The live environment owns action/vision fusion. Legacy callers without
        # that metadata must also establish possession with independent positives.
        confirmed = (not self.require_visual or
                     bool(float(info.get('player_weapon_possession_confirmed',
                                         self.positive_refreshes >= 2)) > .5))
        if has_weapon and fresh is True and confirmed:
            self.pickup_seen = True
        if not self.pickup_seen:
            return
        if fresh is False:
            self.negative_refreshes += 1
        elif fresh is True:
            self.negative_refreshes = 0
        self.dropped |= self.negative_refreshes > self.drop_grace
        if has_weapon and visual is True:
            self.hold_steps += 1
        else:
            self.hold_steps = 0
        # Cached positives may bridge cadence, but never supply the final proof.
        self.success = bool(not self.dropped and fresh is True
                            and self.hold_steps >= self.required_steps)
