"""Outcome rewards shared by final training and evaluation.

Damage is an early learning signal with a bounded budget per life. There is no
income for staying armed, standing near the opponent, or merely pressing attack.
"""
from __future__ import annotations

import math
from typing import Any, Mapping


class MatchObjective:
    VERSION = "stock_damage_v1"

    def __init__(self, *, match_bonus: bool = True) -> None:
        self.match_bonus = match_bonus
        self.reset()

    def reset(self) -> None:
        self.dealt = self.received = 0.0
        self.finished = False

    @staticmethod
    def _positive(value: Any) -> float:
        value = float(value)
        return max(0.0, value) if math.isfinite(value) else 0.0

    def step(self, info: Mapping[str, Any]) -> dict[str, float]:
        terms = dict(dmg_dealt=0., dmg_taken=0., ko_reward=0.,
                     ko_penalty=0., game_win=0., game_loss=0.)
        if not self.finished:
            op_loss = self._positive(info.get("op_stock_lost_step", 0))
            self_loss = self._positive(info.get("self_stock_lost_step", 0))
            terms["ko_reward"] = 5.0 * op_loss
            terms["ko_penalty"] = -5.0 * self_loss
            # Respawn/KO frames can change HUD colour without an actual hit.
            if info.get("damage_valid", False) and not (op_loss or self_loss):
                dealt = min(100.0 - self.dealt, self._positive(info.get("op_delta_damage", 0)))
                received = min(100.0 - self.received, self._positive(info.get("self_delta_damage", 0)))
                self.dealt += dealt
                self.received += received
                terms["dmg_dealt"] = .01 * dealt
                terms["dmg_taken"] = -.005 * received
            if op_loss:
                self.dealt = 0.0
            if self_loss:
                self.received = 0.0
            won = op_loss > 0 and float(info.get("op_stocks_left", 3)) <= 0
            lost = self_loss > 0 and float(info.get("self_stocks_left", 3)) <= 0
            if self.match_bonus and (won or lost):
                self.finished = True
                if won and not lost:
                    terms["game_win"] = 20.
                elif lost and not won:
                    terms["game_loss"] = -20.
        return terms | {"total_reward": sum(terms.values())}


class NoMatchReward:
    """Base environment still measures outcomes; the outer wrapper scores them."""
    def update_memory(self, frame, memory) -> None:
        pass

    def get_reward(self, state, memory) -> float:
        return 0.0
