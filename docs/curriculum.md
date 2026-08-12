# Curriculum, Goals and Rewards

## Goal space

All phases share one 11-dimensional goal vector, defined in
[`train/curriculum_goals.py`](../train/curriculum_goals.py):

| # | Feature | Family |
|---:|---|---|
| 0 | `signed_dx_to_ledge` | recovery |
| 1 | `dy_to_ledge` | recovery |
| 2 | `player_x` | movement |
| 3 | `player_y` | movement |
| 4 | `player_has_weapon` | weapon |
| 5 | `weapon_dx` | weapon |
| 6 | `weapon_dy` | weapon |
| 7 | `rel_distance` | spacing |
| 8 | `rel_dy` | spacing |
| 9 | `in_strike_range` | combat |
| 10 | `opponent_damage_pct` | combat |

`extract_curriculum_goal_features` maps raw state into this space, normalising every dimension to
`[0, 1]`. Signed quantities map zero to **0.5**, so "no offset" is `0.5`, not `0`.

Goal targets are sampled in the **world frame**, then mirrored into the observation's canonical frame
by `StageGoalEnv._sync_goal_frame` before any error is computed. Comparing a world-frame target
against a canonicalised observation would silently invert every horizontal goal.

The combat dimension is `opponent_damage_pct`, read from the UI, replacing an earlier
`frame_advantage_estimate` that was derived from invented hitstun values — a curriculum should not
optimise toward a target defined over a quantity that was never measured.

A goal is a `(target, mask)` pair. The mask selects which dimensions are active, so a single network
and a single goal vector cover all five families. Goal error is

```
error = Σ mask · |feature − target|          (or L2 when use_l2_error)
```

The observation carries both target and mask, so the policy knows *which* goal it is pursuing.

## Phase specifications

Each phase is a `StageSpec` ([`train/curriculum_config.py`](../train/curriculum_config.py)) bundling
the goal sampler, mask, reward shaping and action restrictions.

| Phase | Mask | Success threshold | Reward clip | Attacks |
|---|---|---:|---:|---|
| `recovery_mastery` | ledge dx, dy | 0.08 | 4.0 | disabled |
| `movement_fluency` | player x, y | 0.04 | 3.0 | disabled |
| `weapon_acquisition` | has_weapon, weapon dx/dy | 0.10 | 10.0 | NUM5 only |
| `spacing_neutral` | rel_distance, rel_dy | 0.06 | 3.0 | disabled |
| `combat_execution` | in_strike_range, opponent damage | 0.15 | 6.0 | all |
| `all_skills_llc` | sampled per episode | 0.10 | 6.0 | all |

Two reward modes exist:

- `reward_from_goal_progress = True` → `reward = progress_scale · (prev_error − curr_error)`, clipped
- `reward_from_goal_progress = False` → `reward = −curr_error`

Plus optional shaping: step penalty, success bonus, offstage penalty, death penalty, weapon pickup and
hold bonuses, hit/damage bonuses, whiff penalties, combo-chain bonuses.

> **Note.** `reward_clip` varies from 3.0 to 10.0 across phases. The value head must therefore relearn
> its output scale at every phase transition, which works against the anti-forgetting machinery
> described in [architecture.md](architecture.md).

## Episode structure

`StageGoalEnv` truncates the episode when goal error falls below the success threshold. Importantly,
`env.reset()` does **not** reset the match — Brawlhalla keeps running. A reset clears the `Memory`
object, releases held keys, and carries stock/damage/weapon state forward.

Consequences worth being aware of when reading results:

- The initial-state distribution is "wherever the previous episode ended", not a designed distribution.
- Truncating on success means the value function never observes what follows a completed goal.
- Every reset releases all held keys, so frequent short episodes produce input chatter.

## Retention gates

Phase advancement is gated on retention of earlier skills
([`train/retention.py`](../train/retention.py)):

```
retention = current_score / best_score_so_far
amnesia   = max(0, 1 − retention)
```

Advance only when the current phase clears its minimum skill score, all previous phases hold
`retention ≥ 0.85` and `amnesia ≤ 0.15`, damage trade is non-negative on combat phases, and visual
inspection agrees. Thresholds and the operational protocol are in
[`LLC_MASTERY_HANDOFF.md`](../LLC_MASTERY_HANDOFF.md).

## Scripted teachers

[`train/heuristic_teachers.py`](../train/heuristic_teachers.py) provides a deterministic policy per
goal family, used to bootstrap behaviour cloning.

These are **reactive deadband controllers over the same observation the policy receives**. They hold
no privileged information and no lookahead, so behaviour cloning against them distils a hand-written
function rather than transferring knowledge the network could not otherwise obtain. Their value is in
removing the initial random-walk phase, not in solving exploration.

Fixed:

- the weapon teacher no longer emits a pure idle once armed. Because only successful episodes are
  saved, that idle was replayed as a behaviour-cloning anchor and taught a freeze reflex after every
  pickup; it now holds position and corrects its footing without pressing pickup again.
- the movement teacher no longer requests a jump on every step while the target is above. Doing so
  burned all three jumps in three frames, so airborne targets were never reached and — again, because
  only successes are saved — disappeared from the dataset entirely.

Remaining weaknesses:

- demonstrations are accepted only on success, so goals the teacher cannot reach are still
  under-represented — dataset coverage tracks teacher competence rather than the goal distribution
- `--enforce-recovery-sequence` defaults off, so recovery demonstrations need not contain an actual
  offstage → onstage transition
- the collector still forces idle through the weapon hold window, independently of the teacher
