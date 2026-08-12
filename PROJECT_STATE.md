# Brawlhalla LLC Project State

Last updated: 2026-06-10

## Goal

This project trains a low-level controller for Brawlhalla. The controller learns small reusable skills first, then those skills can later be reused by a higher-level policy.

The current training philosophy is pure basics first:

```text
heuristic teacher -> behavior cloning -> PPO fine-tuning -> retention checks
```

The first teacher policies should be simple and predictable, even if they are bad at the full game. The goal is to teach the action grammar for each skill before asking PPO to optimize timing and adaptation.

The current pipeline is intentionally simple:

1. Capture the game frame.
2. Run YOLO for `agent`, `op`, and `weapons`.
3. Store the current state in `Memory`.
4. Convert `Memory` into the observation vector.
5. During demo collection, keep only episodes that satisfy the current phase rule.

## Current Phase Rules

### `recovery_mastery`

The collector records the useful transition only:

- start condition: player is offstage,
- success condition: player returns onstage,
- saved demo segment: cropped from the offstage moment to the onstage moment.

This teaches recovery as `offstage -> onstage`, not normal movement while already safe.

### `movement_fluency`

This phase is unchanged.

The goal is still an absolute target position:

- `player_x`
- `player_y`

The collector saves movement demos when the player reaches the target area.

### `spacing_neutral`

The real goal remains relative spacing:

- `rel_distance`
- `rel_dy`

For manual demo collection, the mouse helper can convert that relative goal into a screen-like `x/y` target so it is easier to move the player with the pointer. The saved goal is still distance and vertical spacing, not mouse coordinates.

### `combat_execution`

Behavior cloning demos are simple:

- phase 1: hit the enemy once,
- the episode can end after the first successful hit.

PPO training can then shape the second part:

- after the first hit, minimize time until the next hit,
- this teaches combo pressure instead of hit-and-run behavior.

### `weapon_acquisition`

Weapon logic is now direct and small.

Memory collects every visible `weapons` detection in the frame, then chooses the closest weapon to the player. That closest weapon is the only source for:

- `weapon_dx`
- `weapon_dy`
- pickup distance

Pickup rule:

```text
player_has_weapon = 1 if NUM5 is pressed and distance_to_closest_weapon <= 0.05
```

Drop rule:

```text
if player_has_weapon = 1 and NUM5 is pressed, player_has_weapon becomes 0
```

For demo collection, a valid `weapon_acquisition` episode is now stricter:

```text
start unarmed -> find weapon -> pick it up -> hold it for N consecutive steps -> success
```

Default hold window:

```text
N = 20 collector steps
```

The hold window must be consecutive observed `player_has_weapon = 1` steps. A short `player_has_weapon = 0` flicker is tolerated for up to 3 steps, but it resets the hold counter. A longer unarmed streak means the weapon was probably dropped, so the episode is rejected and not saved.

This prevents the collector from counting episodes that start while the player is already holding a weapon. During heuristic collection, if a new weapon episode starts armed, the collector runs an unrecorded warmup reset:

```text
tap NUM5 to drop weapon -> wait until player_has_weapon = 0 -> reset collector state -> start recording
```

The collector and stage wrapper still do not force-drop the weapon as part of normal environment reset. The drop is a weapon-demo warmup step only, because the real game does not reset weapon state between collector episodes.

## Cleaned Up

Removed from the active path:

- disappearance-based pickup inference,
- pending pickup candidate counters,
- extra weapon distance debug fields,
- high-FPS debug recorder arguments,
- forced weapon-drop side effects,
- unused YOLO tracker export,
- unused YOLO detection-vector helpers,
- tracker/Yolo blend-alpha configuration,
- stale stage timer settings.

The remaining weapon path is:

```text
all visible weapons -> closest weapon -> NUM5 and distance <= 0.05 -> player_has_weapon
```

## Next Goal

Build pure heuristic teachers for the easiest phases first:

1. `movement_fluency`: move toward target `player_x/player_y`.
2. `weapon_acquisition`: move toward closest weapon, press NUM5 within `0.05`, then avoid pressing NUM5 while armed.
3. `recovery_mastery`: move toward nearest ledge and jump when below/near ledge.

Then use those heuristic rollouts as BC data before PPO fine-tuning.

Combat heuristic rule:

- turn toward opponent before attacking,
- light attack (`NUM4`) only when facing and within about `0.055` normalized horizontal distance,
- heavy attack (`NUM6`) only when facing and within about `0.075` normalized horizontal distance,
- do not attack when vertical offset is larger than about `0.04`.

Heuristic collection is now available through the normal demo collector:

```powershell
python -m train.collect_bc_locomotion_demos --phase movement_fluency --teacher heuristic --episodes 30
python -m train.collect_bc_locomotion_demos --phase weapon_acquisition --teacher heuristic --episodes 30
python -m train.collect_bc_locomotion_demos --phase recovery_mastery --teacher heuristic --episodes 30
```

To collect the full heuristic curriculum in one run, use:

```powershell
python -m train.collect_heuristic_curriculum_demos --episodes-per-phase 50
```

This runs each phase sequentially with `--teacher heuristic` and saves:

```text
train/models/recovery_mastery_demos.npz
train/models/movement_fluency_demos.npz
train/models/weapon_acquisition_demos.npz
train/models/spacing_neutral_demos.npz
train/models/combat_execution_demos.npz
train/models/all_skills_llc_demos.npz
```

Useful variants:

```powershell
python -m train.collect_heuristic_curriculum_demos --episodes-per-phase 50 --dry-run
python -m train.collect_heuristic_curriculum_demos --phases core --episodes-per-phase 50
python -m train.collect_heuristic_curriculum_demos --episodes-per-phase 50 --weapon-hold-steps 30
```

`--dry-run` prints the exact commands without collecting. `--phases core` runs only the five focused phases and skips `all_skills_llc`.

Manual collection is still available with `--teacher manual`, which keeps env key injection disabled and records keyboard labels while you control the game.

For manual `weapon_acquisition` collection, use this exact behavior:

1. Start unarmed.
2. Walk to the closest visible weapon.
3. Press NUM5 only when close.
4. Keep holding the weapon.

If it still fails, the next thing to inspect is not the pickup logic. It would be one of these:

- YOLO weapon center is not where the weapon visually is,
- player center is shifted too far from the true pickup point,
- NUM5 is not reaching the environment as action code `3`,
- the 0.05 threshold is too strict for the detected centers.
