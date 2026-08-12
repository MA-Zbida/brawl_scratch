# Observation Space

## Design principle

Hidden game state — hitstun, endlag, exact jump count, frame advantage — cannot be
read without memory access. The earlier design **guessed** at it with hand-written
formulas. This one does not.

> **Measure honestly, then supply history.**

Estimating latent state from history is exactly what a network with a temporal
window does, and it does it better than a formula, because it learns the mapping
from data instead of from an assumption about Brawlhalla's physics. So the layout
carries measured quantities plus the raw ingredients a latent can be inferred
from, and the environment stacks a short history so the policy can do the
inferring.

## Shape

| Piece | Width |
|---|---:|
| Single frame (`StateSpec.dim()`) | 59 |
| Dynamic block (`StateSpec.DYNAMIC_DIM`) | 21 |
| History offsets (default `(2, 4, 8)`) | 3 × 21 = 63 |
| **Environment observation** | **122** |
| Goal target + mask (goal-conditioned training) | +22 |
| **Policy observation** | **144** |

Layout:

```
[ core(t) : 59 | dynamic(t-2) : 21 | dynamic(t-4) : 21 | dynamic(t-8) : 21 ]
```

The first 59 entries are always the complete current frame, so
`StateSpec.get(obs, name)` works regardless of how deep the window is. Only the
**dynamic block** is stacked — slow-moving context is carried once, which keeps
the observation from tripling for no information gain.

Frame stacking is used rather than an LSTM deliberately: recurrent PPO is
markedly more sample-hungry, and this setup is bounded by wall-clock time, not
compute. Revisit recurrence when samples become cheap.

## Canonical frame

Observations are emitted **horizontally mirrored** whenever needed so the opponent
is always on the same side. Brawlhalla is left-right symmetric, so without this
the policy learns "opponent on my right" and "opponent on my left" as two separate
problems out of the same real-time budget. Canonicalising roughly halves the state
space it has to cover.

- The decision uses `rel_dx`, or the stage-centre offset when no opponent is
  visible, with a **deadband** so the frame cannot chatter when the key quantity
  hovers near zero.
- Absolute positions reflect about the **stage** centre, not `0.5` — the
  calibrated platform is not centred on the screen, and reflecting about the wrong
  axis leaves a residual asymmetry in every ledge-relative feature.
- The policy's action is flipped back before it reaches the game. That half is
  easy to forget and is covered by tests.
- `canon_mirrored` records whether the flip was applied.

Details in [`feature_extractor/memory/canonicalize.py`](../feature_extractor/memory/canonicalize.py).

## Provenance

| Class | Meaning |
|---|---|
| **Measured** | Read from the screen this frame |
| **Derived** | Deterministic function of measured values plus fixed stage calibration |
| **Estimate + ingredient** | Cannot be observed; the estimate ships beside a measured quantity the policy can use to detect drift |
| **Exact** | The agent's own previous action |

### Dynamic block — stacked (0–20)

| # | Feature | Class | Note |
|---:|---|---|---|
| 0–1 | `player_x`, `player_y` | Measured | detector centre, foot-shifted |
| 2–3 | `player_vx`, `player_vy` | Derived | finite difference over real `dt` |
| 4–5 | `player_w`, `player_h` | Measured | **bbox extent — animation cue** |
| 6–7 | `player_dw`, `player_dh` | Derived | a swing widens the box, a dodge compresses it |
| 8–15 | opponent equivalents | — | same treatment |
| 16–18 | `rel_dx`, `rel_dy`, `rel_distance` | Derived | relative geometry |
| 19–20 | `rel_vx`, `rel_vy` | Derived | closing speed |

Bounding-box extent is the closest thing to frame data available without reading
memory, and the detector already produces it — the previous design computed it and
threw it away by keeping only the centre.

### Stage geometry (21–31)

`signed_dx_to_ledge`, `dy_to_ledge`, `dist_to_nearest_ledge`, `player_grounded`,
`player_on_edge`, `player_is_offstage`, `signed_dx_to_stage_center`,
`dist_to_blastzone_x`, `dist_to_blastzone_y`, `opponent_grounded`,
`opponent_is_offstage` — all **derived** from measured position plus the fixed
calibration in [`config.py`](../config.py).

Blast-zone margins are measured about the stage centre so they stay symmetric
under the mirror.

### Resources (32–40)

`player_damage_pct`, `opponent_damage_pct`, `self_stocks_norm`, `op_stocks_norm`
are **measured** from UI pixel probes. `weapon_dx`, `weapon_dy`,
`weapon_on_ground` are **measured** from the detector.

`player_has_weapon` / `opponent_has_weapon` become measured once the crop
classifier lands; until then possession is action-inferred, which is tracked as a
known issue in the README.

### Estimates with their ingredients (41–45)

| Estimate | Ingredient beside it |
|---|---|
| `player_jumps_norm` | `player_airborne_time` |
| `dodge_cooldown_norm` | `time_since_dodge_input` |

These cannot be observed. The estimate is kept because it is usually right; the
measured ingredient beside it is what lets the policy learn *when it is wrong*
rather than trusting it blindly. `opponent_airborne_time` completes the pair.

### Perception provenance (46–51)

`identity_observed`, `time_since_indicator`, `player_missing_frames`,
`opponent_missing_frames`, `player_confidence`, `opponent_confidence`.

A vision-only agent that knows when its own senses are unreliable can learn to
play conservatively during those frames. One that is silently misinformed just
learns noise. On this pipeline that matters more than any individual game feature.

### Combat context and action (52–58)

`in_strike_range` (derived distance band), `opponent_exists`, the previous action
(**exact**), and `canon_mirrored`.

## Removed

| Feature | Why |
|---|---|
| `facing_opponent` | Derived from the agent's own previous input — circular. Canonicalisation makes it unnecessary: the opponent is always on the same side |
| `player_hitstun`, `opponent_hitstun` | Invented formula `0.15 + 0.25 · damage_pct` |
| `frame_advantage_estimate` | Difference of two invented quantities — and it was a **combat goal dimension** |
| `opponent_dodge_cooldown_norm` | Detector defaults to `None`; constant zero |
| `opponent_jumps_norm` | Never decremented; constant |
| `ledge_is_occupied`, `last_knockback_*`, `*_time_since_hit` | Negligible signal |

The combat goal now targets `in_strike_range` + `opponent_damage_pct` — both
grounded in measurement, so the curriculum no longer optimises toward a target
defined over a guess.

## Conventions

- Positions normalised to `[0, 1]` against capture resolution.
- `y` **increases downward** (screen coordinates). Moving up means decreasing `y`.
- Signed offsets clamped to `[-1, 1]` in the raw observation; mapped to `[0, 1]`
  in goal space with `0.5` as the zero point.
- Stage geometry is a **fixed calibration** for one map at 1920×1080. Changing map
  or resolution invalidates every geometry-derived feature.
