from __future__ import annotations

from typing import Sequence
import numpy as np

from action_space import COMPONENT_DIM, COMPONENT_NAMES


class StateSpec:
    """Single-frame observation layout.

    Design principle: **measure honestly, then supply history.** Hidden game state
    (hitstun, endlag, exact jump count) cannot be read without memory access, so it
    is not fabricated. The layout carries measured quantities plus the raw
    ingredients a latent can be inferred from, and the environment stacks a temporal
    window so the policy learns the dynamics rather than trusting a formula.

    Layout
    ------
    Indices ``0 .. DYNAMIC_DIM-1`` are the **dynamic block** -- what the environment
    stacks over the temporal window. Everything after is slow-moving context carried
    once, at the current frame only::

        [ core(t) | dynamic(t-k1) | dynamic(t-k2) | ... ]

    The first ``StateSpec.dim()`` entries are always the complete current frame, so
    ``StateSpec.get(obs, name)`` works regardless of window depth.

    The **previous action lives inside the dynamic block**, decomposed into semantic
    components rather than stored as an action index. Two reasons:

    * ``s_{t+1} ~ P(. | s_t, a_t)``, so a history of states without the actions that
      produced them is not Markov. "Rising because I jumped" and "rising because I
      got hit" are indistinguishable from positions alone.
    * A normalised action index would imply an ordering between actions
      (``|a_18 - a_19| < |a_18 - a_26|``) that carries no meaning.

    Frame convention
    ----------------
    Values are emitted in the **canonical (mirrored) frame** -- see
    ``feature_extractor.memory.canonicalize``. The observation is flipped whenever
    needed so the opponent is always on the same side, which halves the state space
    the policy must cover.

    The mirror flag itself is deliberately **not** an observation feature. Handing it
    back would let the policy distinguish the two physical situations that
    canonicalisation exists to merge, partially undoing the symmetry. It travels in
    ``info`` instead, for the environment and wrappers that genuinely need it.
    """

    _DYNAMIC: list[str] = [
        # ── agent kinematics + silhouette (0-7) ──────────────────────────────
        "player_x",                 # 0   normalised [0, 1]; FOOT position
        "player_y",                 # 1
        "player_vx",                # 2   finite difference over real dt
        "player_vy",                # 3
        "player_w",                 # 4   detector bbox width  -- animation cue
        "player_h",                 # 5   detector bbox height -- animation cue
        "player_dw",                # 6   change in width  -- an attack extends the box
        "player_dh",                # 7   change in height -- a dodge compresses it

        # ── opponent kinematics + silhouette (8-15) ──────────────────────────
        "opponent_x",               # 8
        "opponent_y",               # 9
        "opponent_vx",              # 10
        "opponent_vy",              # 11
        "opponent_w",               # 12
        "opponent_h",               # 13
        "opponent_dw",              # 14
        "opponent_dh",              # 15

        # ── relative geometry (16-20) ────────────────────────────────────────
        "rel_dx",                   # 16  signed (opponent.x - player.x)
        "rel_dy",                   # 17  foot-to-foot
        "rel_distance",             # 18  euclidean
        "rel_vx",                   # 19  opponent_vx - player_vx
        "rel_vy",                   # 20

        # ── executed previous action, decomposed (21-27) ─────────────────────
        *COMPONENT_NAMES,
    ]

    _STATIC: list[str] = [
        # ── stage geometry, from measured position + fixed calibration ───────
        "signed_dx_to_ledge",
        "dy_to_ledge",
        "dist_to_nearest_ledge",
        "player_grounded",
        "player_on_edge",
        "player_is_offstage",
        "signed_dx_to_stage_center",
        "dist_to_blastzone_x",
        "dist_to_blastzone_y",
        "opponent_grounded",
        "opponent_is_offstage",

        # ── resources ────────────────────────────────────────────────────────
        "player_damage_pct",        # UI pixel probe
        "opponent_damage_pct",      # UI pixel probe
        "self_stocks_norm",
        "op_stocks_norm",
        "player_has_weapon",        # binary; the weapon-acquisition goal targets it
        # Weapon identity, not just possession. Identical (state, action) pairs with
        # different weapons produce different startup, reach and displacement, so
        # collapsing them to a single bit aliases transitions the policy has to
        # distinguish. 0 = unarmed, 1 / 2 = the legend's two weapons.
        "player_weapon_type",
        "opponent_weapon_type",
        "weapon_dx",                # nearest ground weapon
        "weapon_dy",
        "weapon_on_ground",

        # ── uncertain estimates, shipped WITH their measured ingredients ─────
        "player_jumps_norm",        # estimate: jumps remaining
        "player_airborne_time",     # ingredient: time since ground contact
        "dodge_cooldown_norm",      # estimate: dodge availability
        "time_since_dodge_input",   # ingredient: time since a dodge was sent
        "opponent_airborne_time",

        # ── perception provenance ────────────────────────────────────────────
        "identity_observed",        # 1 = indicator seen, 0 = carried forward
        "time_since_indicator",
        "player_missing_frames",
        "opponent_missing_frames",
        "player_confidence",
        "opponent_confidence",

        # ── derived combat context ───────────────────────────────────────────
        "in_strike_range",          # can land, or be landed on
        "opponent_exists",
    ]

    FEATURES: list[str] = [*_DYNAMIC, *_STATIC]

    #: Leading features stacked over the temporal window.
    DYNAMIC_DIM: int = len(_DYNAMIC)

    #: Where the decomposed previous action starts inside the dynamic block.
    ACTION_OFFSET: int = len(_DYNAMIC) - COMPONENT_DIM

    _INDEX: dict[str, int] = {name: i for i, name in enumerate(FEATURES)}

    @classmethod
    def dim(cls) -> int:
        """Single-frame width (excludes the history window)."""
        return len(cls.FEATURES)

    @classmethod
    def dynamic_dim(cls) -> int:
        return cls.DYNAMIC_DIM

    @classmethod
    def dynamic_names(cls) -> list[str]:
        return list(cls.FEATURES[: cls.DYNAMIC_DIM])

    @classmethod
    def observation_dim(cls, history_offsets: Sequence[int] = ()) -> int:
        return cls.dim() + (len(tuple(history_offsets)) * cls.DYNAMIC_DIM)

    @classmethod
    def observation_names(cls, history_offsets: Sequence[int] = ()) -> list[str]:
        names = list(cls.FEATURES)
        for offset in history_offsets:
            names.extend(f"t-{int(offset)}_{n}" for n in cls.dynamic_names())
        return names

    @classmethod
    def index(cls, name: str) -> int:
        return cls._INDEX[name]

    @classmethod
    def get(cls, obs: np.ndarray, name: str) -> float:
        return float(obs[cls._INDEX[name]])

    @classmethod
    def get_multi(cls, obs: np.ndarray, *names: str) -> np.ndarray:
        indices = [cls._INDEX[n] for n in names]
        return np.asarray(obs)[indices].astype(np.float32)

    @classmethod
    def names(cls) -> list[str]:
        return list(cls.FEATURES)

    @classmethod
    def validate_vector(cls, vec: np.ndarray) -> None:
        expected = cls.dim()
        actual = vec.shape[-1] if vec.ndim >= 1 else 0
        if actual != expected:
            raise ValueError(
                f"Observation vector has {actual} dim, expected {expected}"
            )
