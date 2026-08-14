# Codex task — fix demo collection quality

## Context

Repo: `c:\Users\mazbi\OneDrive\Bureau\brawl_scratch` (Windows, PowerShell).
Interpreter: **`C:\venvs\brawl312\Scripts\python.exe`** — use this, not any other Python
on PATH. The other interpreters lack CUDA torch and will hide device bugs.

Run the suite with:

```
C:\venvs\brawl312\Scripts\python.exe -m pytest tests\ -q
```

243 tests currently pass. They must still pass when you are done.

This is a vision-only RL agent for Brawlhalla. Phase-specific heuristic teachers
generate behaviour-cloning demos; those demos initialise PPO. The action space is
`Discrete(27)` defined in `action_space.py`, with **canonical** directions
(`TOWARD` = toward the opponent, or toward stage centre when no opponent is visible).

**Do not change the action space, the observation space, or `tools/validate_llc_demos.py`'s
thresholds.** The validator is the instrument; the collectors are what is broken.

## Evidence

Five demo archives were collected in `train\models\*_demos.npz` (50 episodes each).
Measured from the archives themselves:

| phase | frames | eps | 1-step eps | idle | distinct actions used |
|---|---|---|---|---|---|
| recovery_mastery | 62 | 50 | **48 (96%)** | 0.048 | 6/27 |
| movement_fluency | 1614 | 50 | 9 (18%) | 0.032 | 9/27 |
| weapon_acquisition | 3049 | 50 | 0 (0%) | **0.341** | 6/27 |
| spacing_neutral | 768 | 50 | **32 (64%)** | 0.009 | 9/27 |
| combat_execution | 1237 | 50 | **20 (40%)** | 0.000 | 8/27 |

Additional measurements:

- `recovery_mastery`: `player_is_offstage` is true in only **14%** of episode *starts*
  and 11% of all frames; `player_grounded` is true in **79%** of frames.
- `recovery_mastery` metadata: `episodes_attempted=50, episodes_collected=50,
  episodes_rejected=0, recovery_sequence_enforced=0`.
- Across **all 5768 frames of all five phases**, the nine dodge actions
  (`DODGE_SPOT`, `DODGE_TOWARD`, `DODGE_AWAY`, `DODGE_UP`, `DODGE_DOWN`,
  `DODGE_UP_TOWARD`, `DODGE_UP_AWAY`, `DODGE_DOWN_TOWARD`, `DODGE_DOWN_AWAY`)
  appear **zero times**. `grep -n DODGE train/heuristic_teachers.py` returns nothing.
- `weapon_acquisition` action mix: `PICKUP` 43.4%, `NOOP` 34.1% — 77.5% of the archive.
- `combat_execution` never emits `MOVE_AWAY` or `NOOP`.

## Defect 1 — recovery episodes succeed at spawn (highest priority)

`_sampler_recovery` in `train/curriculum_config.py` sets the goal to
`signed_dx_to_ledge = 0, dy_to_ledge = 0` ("be at the ledge"), and the recovery spec uses
`success_threshold=0.08` with `use_l2_error=True`.

Measured at spawn: `signed_dx_to_ledge` mean `-0.023` (range `-0.174..+0.183`),
`dy_to_ledge` mean `+0.009` (range `-0.222..+0.091`). The L2 goal error while simply
**standing on the stage** is ~0.013, far inside the 0.08 threshold. So the episode
terminates successfully on step 1, the collector accepts it, and 48 of 50 recovery
"demonstrations" are a single frame of a grounded agent.

A guard already exists at `train/collect_bc_locomotion_demos.py` around line 582:

```python
accept_episode = bool(accept_episode and ep_step1_transition_seen and ep_terminal_success_seen)
```

but it is gated on the recovery-sequence flag, which was `0` for this collection.

**Required changes:**

1. Recovery demos must be recorded **from the moment the agent is actually offstage**,
   not from `reset()`. Add an *arming* condition to the recovery collection loop: after
   reset, step the env with the teacher's action but **discard** the transitions until
   `player_is_offstage > 0.5` becomes true; begin buffering from that frame. If the agent
   never goes offstage within the episode's step budget, reject the episode and count it.
   Read the flag with `StateSpec.get(obs, "player_is_offstage")`.
2. Make the recovery-sequence guard **on by default** for `recovery_mastery`. If a caller
   explicitly disables it, print a clearly worded warning that the resulting archive will
   contain spawn-success episodes.
3. Do **not** solve this by lowering `success_threshold`. The threshold is not the problem;
   recording an episode that starts in the goal state is.

## Defect 2 — vacuous episodes are accepted in every phase

64% of `spacing_neutral` and 40% of `combat_execution` episodes are also exactly one step,
for the same underlying reason: the sampled goal is already satisfied at reset.

**Required changes (apply to all phases, not just recovery):**

1. In the collector, after `reset()` and before recording, evaluate the goal error. If the
   goal is already satisfied at step 0, **resample the goal target** (bounded retries, e.g.
   8) until it is not. If no unsatisfied target can be found, reject the episode.
2. Add a `--min-episode-steps` CLI option (default `4`) to
   `train/collect_bc_locomotion_demos.py`. Episodes shorter than this are rejected.
3. Track rejections in a new metadata key `episodes_rejected_trivial`, saved into the npz
   alongside the existing `episodes_rejected*` keys, and print it in the run summary.
4. Keep `episodes_attempted`/`episodes_collected` semantics unchanged.

## Defect 3 — nine of twenty-seven actions have zero demonstrations

The dodge/dash family is legal in four of five phases (only `weapon_acquisition` restricts
`allowed_actions`, via `WEAPON_PHASE_ACTIONS`) but no teacher ever emits one. BC will drive
their probability to ~0, and PPO starting from that policy will effectively never explore a
third of its own action space. Dodge/dash is a core Brawlhalla mechanic — it is the primary
recovery tool and the primary defensive tool.

**Required changes in `train/heuristic_teachers.py`.** Add these rules, matching the file's
existing style (small helpers, `StateSpec.get`, canonical directions, no new dependencies):

- **Recovery** (`_recovery_action`): when `player_jumps_norm <= cfg.min_jumps_norm` (out of
  jumps) and still offstage, emit a directional air dodge toward the ledge —
  `DODGE_UP_TOWARD` when `dy_to_ledge` indicates the ledge is above, otherwise `DODGE_TOWARD`.
  Respect the canonical sign of `signed_dx_to_ledge` exactly as `_horizontal` already does.
- **Movement** (`_movement_action`): when the horizontal distance to the target exceeds a new
  `cfg.dash_distance` and the agent is grounded, emit `DODGE_TOWARD`/`DODGE_AWAY` (a ground
  dash) instead of a plain walk, at most once every `cfg.dash_cooldown_steps` steps.
- **Spacing** (`_spacing_action`): when `rel_distance` is well below the target distance and
  the agent is grounded, emit `DODGE_AWAY` to disengage rather than walking away.
- **Combat** (`_combat_action`): when `in_strike_range` is true and the opponent is the
  threat (use the existing combat-context features; do not invent new observation entries),
  emit `DODGE_AWAY` or `DODGE_SPOT` some of the time instead of always attacking. Combat
  currently never emits `MOVE_AWAY` or `NOOP` either — a teacher that only ever presses
  forward teaches an agent that cannot disengage.

Add the new tunables to `HeuristicConfig` with conservative defaults and document each with
a one-line comment explaining the value.

## Defect 4 — the weapon teacher spams pickup and idles

`weapon_acquisition` is 43.4% `PICKUP` and 34.1% `NOOP`. In `_weapon_action`, `PICKUP` is
emitted on every step while `distance <= cfg.pickup_distance`, and `_hold_position` returns a
bare `NOOP` whenever the agent is armed and onstage.

**Required changes:**

1. Emit `PICKUP` at most once per contiguous in-range window, with a short retry cooldown,
   rather than every frame.
2. Replace the armed-and-onstage `NOOP` with light positional behaviour (small approach or
   retreat relative to stage centre). The existing comment in `_hold_position` already
   explains why a bare `NOOP` is harmful — it just does not act on it in this branch.

## Tests you must add

In `tests/`, using the existing conftest stubs (there is no game and no GPU in CI):

1. Each phase teacher emits at least one dodge-family action over a synthetic sweep of
   observations that should trigger it (out of jumps + offstage for recovery, far target +
   grounded for movement, too-close for spacing).
2. `_weapon_action` does not emit `PICKUP` on consecutive steps while continuously in range.
3. An episode whose goal is satisfied at step 0 is rejected by the collector's acceptance
   logic, and `episodes_rejected_trivial` counts it.
4. The recovery collector discards pre-offstage frames: given a scripted observation
   sequence that goes onstage → onstage → offstage → offstage, only the last two frames are
   buffered.

## Constraints

- Do not change `action_space.py`, `feature_extractor/memory/state_spec.py`, or the
  validator's thresholds.
- Do not add new observation features; use what `StateSpec` already exposes.
- Do not lower any success threshold to make episodes last longer.
- Do not delete or rewrite the existing archives in `train\models\` — they will be
  recollected by the user after your changes land.
- Every entry point must still run: verify with
  `C:\venvs\brawl312\Scripts\python.exe <script> --help` for each of
  `train\collect_bc_locomotion_demos.py`, `train\collect_heuristic_curriculum_demos.py`,
  `tools\validate_llc_demos.py`, `train\pretrain_bc_locomotion.py`, `train\train_curriculum.py`.

## Report back

State, per defect, what you changed and which test covers it. If you disagree with any part
of this spec, say so explicitly rather than silently doing something else.
