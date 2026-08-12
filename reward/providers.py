"""Reward providers for the non-goal-conditioned (full-match) setting.

The curriculum phases compute their own goal-shaped reward in StageGoalEnv; this
is the plain match reward used when training against the bot directly.
"""

from __future__ import annotations

from feature_extractor.memory.structured_memory import Memory


class NullRewardProvider:
    """Reward for training-mode vs easy bot.

    Components (8 total):
        1. dmg_dealt      – positive per opponent HP lost
        2. ko_reward       – bonus per opponent stock taken
        3. ko_penalty      – penalty per self stock lost
        4. game_win        – large terminal bonus
        5. game_loss       – large terminal penalty
        6. weapon_held     – small per-step bonus for holding a weapon
        7. approach        – continuous proximity shaping
        8. proximity_bonus – per-step reward for being in strike range
        9. edge            – mild penalty for being off-platform

    NOTE: dmg_taken removed — it made the agent scared/avoidant.
    KO penalty already punishes dying; per-hit pain caused camping.
    """

    # ── tuneable constants ─────────────────────────────────────────
    DMG_DEALT_COEFF: float = 0.15        # per raw-HP point dealt
    KO_REWARD: float = 15.0              # per opponent stock lost
    KO_PENALTY: float = -5.0             # per self stock lost (softened: less risk-averse)
    GAME_WIN_REWARD: float = 30.0        # terminal: opponent loses last stock
    GAME_LOSS_PENALTY: float = -15.0     # terminal: self loses last stock
    WEAPON_HELD_BONUS: float = 0.005     # per step while holding a weapon
    APPROACH_COEFF: float = 0.05         # continuous proximity: -dist * coeff (stronger pull)
    PROXIMITY_BONUS: float = 0.02        # per step while in strike range
    STRIKE_RANGE: float = 0.18           # normalised distance threshold

    # ── edge penalty (mild, prevents running to blast zone) ────────
    #   Platform: x∈[0.323, 0.674]  y∈[0.570, 0.808]
    EDGE_X_MIN: float = 0.30
    EDGE_X_MAX: float = 0.70
    EDGE_Y_MAX: float = 0.83
    EDGE_COEFF: float = -0.05            # gentle nudge back on stage

    def __init__(self) -> None:
        pass

    def reset(self) -> None:
        """Reset per-episode state (called at env.reset)."""
        pass

    def get_reward_breakdown(self, state, memory: Memory) -> dict[str, float]:
        # ── positions & distances ──────────────────────────────────
        # Use pre-calculated values from memory instead of math.hypot
        curr_dist = float(memory.rel_distance) if (memory.player.exists and memory.opponent.exists) else 0.5
        px, py = memory.player.x, memory.player.y
        ox, oy = memory.opponent.x, memory.opponent.y

        both_exist = memory.player.exists and memory.opponent.exists

        # ── stock changes ─────────────────────────────────────────
        op_stock_lost = max(0.0, memory.prev_op_stocks_left - memory.op_stocks_left)
        self_stock_lost = max(0.0, memory.prev_self_stocks_left - memory.self_stocks_left)

        # ═══ 1. DAMAGE DEALT ══════════════════════════════════════
        dmg_dealt = self.DMG_DEALT_COEFF * float(memory.op_delta_damage)

        # ═══ 2. KO EVENTS ═════════════════════════════════════════
        ko_reward = self.KO_REWARD * op_stock_lost
        ko_penalty = self.KO_PENALTY * self_stock_lost

        # ═══ 3. GAME WIN / LOSS (terminal) ════════════════════════
        game_win = self.GAME_WIN_REWARD if (memory.op_stocks_left <= 0.0 and op_stock_lost > 0.0) else 0.0
        game_loss = self.GAME_LOSS_PENALTY if (memory.self_stocks_left <= 0.0 and self_stock_lost > 0.0) else 0.0

        # ═══ 4. WEAPON HELD ═══════════════════════════════════════
        weapon_held = self.WEAPON_HELD_BONUS if memory.player.weapon_state > 0.0 else 0.0

        # ═══ 5. APPROACH (continuous proximity) ═══════════════════
        approach = 0.0
        if both_exist:
            approach = -self.APPROACH_COEFF * curr_dist

        # ═══ 6. PROXIMITY BONUS (in strike range) ════════════════
        proximity_bonus = 0.0
        if both_exist and curr_dist < self.STRIKE_RANGE:
            proximity_bonus = self.PROXIMITY_BONUS

        # ═══ 7. EDGE PENALTY (mild off-platform nudge) ═══════════
        edge = 0.0
        if memory.player.exists:
            overshoot = 0.0
            if px < self.EDGE_X_MIN:
                overshoot = max(overshoot, (self.EDGE_X_MIN - px) / max(self.EDGE_X_MIN, 1e-6))
            elif px > self.EDGE_X_MAX:
                overshoot = max(overshoot, (px - self.EDGE_X_MAX) / max(1.0 - self.EDGE_X_MAX, 1e-6))
            if py > self.EDGE_Y_MAX:
                overshoot = max(overshoot, (py - self.EDGE_Y_MAX) / max(1.0 - self.EDGE_Y_MAX, 1e-6))
            edge = self.EDGE_COEFF * min(overshoot, 1.0)

        # ═══ TOTAL ════════════════════════════════════════════════
        total = (
            dmg_dealt
            + ko_reward + ko_penalty
            + game_win + game_loss
            + weapon_held
            + approach + proximity_bonus
            + edge
        )

        return {
            "dmg_dealt": float(dmg_dealt),
            "ko_reward": float(ko_reward),
            "ko_penalty": float(ko_penalty),
            "game_win": float(game_win),
            "game_loss": float(game_loss),
            "weapon_held": float(weapon_held),
            "approach": float(approach),
            "proximity_bonus": float(proximity_bonus),
            "edge": float(edge),
            "total_reward": float(total),
        }
    
    def get_reward(self, state, memory: Memory) -> float:
        reward_dict = self.get_reward_breakdown(state, memory)
        return float(reward_dict["total_reward"])

    def update_memory(self, frame, memory: Memory,) -> None:
        return
