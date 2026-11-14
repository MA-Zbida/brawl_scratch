"""Damage and KO events, from the HUD readings of successive frames (research/DESIGN.md section 7).

The reward and the virtual matches are computed from these events only. A frame the HUD reader
could not read gives no events and changes nothing: the next readable frame is compared with the
last readable one, so damage dealt behind an unreadable frame is still counted, once.
"""
from __future__ import annotations

from dataclasses import dataclass

from brawl.sense.hud import Slot

NOISE = 2.0       # a damage reading must move by more than this to count (JPEG frames wobble by 1.7)
ZERO = 3.0        # a reading below this after a fall is a fresh ring: the fighter was KO'd
KO_FRAMES = 240   # one KO per fighter within this many game frames (ring reset, then its badge)
CREDIT_FRAMES = 240  # after a lost stock, the other fighter's cyan "+2" badge shows within this if it scored


@dataclass
class Events:
    damage_dealt: float = 0.0  # to Mordex
    damage_taken: float = 0.0  # by Wu Shang
    ko_scored: int = 0         # Mordex lost a stock
    ko_lost: int = 0           # Wu Shang lost a stock
    misread: int = 0           # a damage reading fell without reaching zero (ignored)
    # Known a few seconds after the stock was lost (CREDIT_FRAMES): nobody was credited with it, the
    # fighter fell or self-destructed (the HUD shows -3 and no "+2" on the other). Not part of the reward.
    fell: int = 0              # Wu Shang lost a stock by himself
    opponent_fell: int = 0     # Mordex lost a stock by himself


class Ledger:
    def __init__(self) -> None:
        self.damage = {"wu_shang": None, "mordex": None}  # last accepted reading
        self.badge = {"wu_shang": None, "mordex": None}
        self.last_ko = {"wu_shang": -KO_FRAMES, "mordex": -KO_FRAMES}
        self.pending: dict[str, int] = {}  # fighter -> frame of a lost stock not yet known to be credited

    def update(self, hud: dict[str, Slot] | None, frame: int) -> Events:
        """The events between the last readable frame and this one; ``frame`` counts game frames."""
        e = Events()
        if hud is None:
            return e
        for who, slot in hud.items():
            ko, last = False, self.damage[who]
            if last is None or slot.damage > last + NOISE:
                gained = 0.0 if last is None else slot.damage - last
                e.damage_taken += gained if who == "wu_shang" else 0.0
                e.damage_dealt += gained if who == "mordex" else 0.0
                self.damage[who] = slot.damage
            elif slot.damage < last - NOISE:
                if slot.damage < ZERO:  # the ring was reset: a lost stock, if it held real damage
                    ko, self.damage[who] = last >= ZERO + NOISE, slot.damage
                else:
                    e.misread += 1
            if slot.badge == "lost" and self.badge[who] != "lost":  # also seen when KO'd at zero damage
                ko = True
            self.badge[who] = slot.badge
            if ko and frame - self.last_ko[who] >= KO_FRAMES:
                self.last_ko[who] = frame
                self.pending[who] = frame
                e.ko_lost += who == "wu_shang"
                e.ko_scored += who == "mordex"
        other = {"wu_shang": "mordex", "mordex": "wu_shang"}
        for who, since in list(self.pending.items()):
            if hud.get(other[who]) is not None and hud[other[who]].badge == "scored":
                del self.pending[who]  # the other fighter was credited: a KO, not a fall
            elif frame - since >= CREDIT_FRAMES:
                del self.pending[who]
                e.fell += who == "wu_shang"
                e.opponent_fell += who == "mordex"
        return e


def reward(e: Events, damage_weight: float = 0.01, ko_weight: float = 1.0) -> float:
    return damage_weight * (e.damage_dealt - e.damage_taken) + ko_weight * (e.ko_scored - e.ko_lost)
