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

Stage success currently happens when the observation reports `player_has_weapon = 1`. The collector and stage wrapper no longer force-drop the weapon between episodes, and reset logic no longer manually overwrites weapon state. If the game still has the player holding a weapon, memory is allowed to preserve that state.

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
