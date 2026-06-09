# Weapon Acquisition Video Feature Audit - 2026-06-09

Source video:
`C:\Users\medre\Downloads\obs_debug_weapon_acquisition_20260609_183318.mp4`

## Verdict

The video confirms a real perception-state bug in `player_has_weapon`.

At `t=15.48s` and `t=16.21s`, the right-side gameplay visibly shows the player holding the sword. At the same time, the overlay reports:

- `player_has_weapon: +0.0000`
- `weapon_on_ground: +0.0000`
- `weapon_dx: +0.0000`
- `weapon_dy: +0.0000`
- active goal feature `player_has_weapon cur=0.000 tgt=1.000`

So weapon localization and ground-weapon disappearance were working. The missing piece was self possession.

## Root Cause

`weapon_on_ground` was direct YOLO state: any visible `weapons` detection set it to `1`, and no visible weapon set it to `0`.

`player_has_weapon` was not direct vision state. It was inferred only when:

1. the action channel said pickup/throw, currently `attack == 3`;
2. the weapon was close enough;
3. the weapon disappeared for enough frames.

In the video, the pickup event is not represented as `attack == 3` on the overlay. This means the possession transition was never armed, even though the visual evidence showed the weapon disappeared into the player's hand.

There was also a stale-distance bug: `BrawlDeepEnv._distance_player_to_weapon()` used cached `memory.weapon_dx/dy`, which are refreshed when observations are built, not immediately after detections update. This could make action-based pickup inference miss tight timing windows.

## Feature Checks From The Video

The following reviewed features looked internally consistent:

- `weapon_on_ground`: correct in the failing pickup window; it turns `0` when the ground weapon disappears.
- `weapon_dx`, `weapon_dy`: correct when a ground weapon exists; both become `0` after no ground weapon remains.
- `player_x`, `player_y`: match the visible player position, using foot-shifted player Y.
- `opponent_x`, `opponent_y`: match the visible bot position.
- `rel_dx`, `rel_dy`, `rel_distance`: match `opponent - player` geometry. Example at `t=15.48s`: `0.6699 - 0.4912 = 0.1787`, matching `rel_dx`.
- `in_strike_range`: matches the current smooth distance band formula. It is `0` when distance is outside the extended band and partial when distance is near.
- `player_grounded`, `player_on_edge`, `player_is_offstage`: looked plausible in the checked grounded/offstage windows.
- `signed_dx_to_ledge`, `dy_to_ledge`: looked plausible during the left-ledge/offstage window.
- `dodge_cooldown_norm`: rose after visible dodge input, as expected.
- stocks and damage: stable and plausible for the shown training-mode state.

The following feature is intentionally action-derived, not pure vision:

- `facing_opponent` / `player_facing_dir`: these depend on previous movement/action proxy logic, so they may disagree with sprite orientation in some frames. That is expected under current code.

## Implemented Fix

Changed self-weapon possession from action-only inference to action-plus-visual-event inference.

New behavior:

1. When a ground weapon is near the player, memory records a short pickup candidate.
2. If that same ground item disappears for two frames while the player is still near the candidate position, `player.weapon_state` becomes `1.0`.
3. This works even if the pickup key was not captured by keyboard polling.
4. It also works when another weapon remains visible elsewhere on the stage.
5. If the player moved away before confirmation, the candidate is discarded to avoid false positives.

Also fixed:

- `BrawlDeepEnv._distance_player_to_weapon()` now computes distance from current `memory.weapon.x/y` and `memory.player.x/y`.
- `tools/debug_observation_overlay.py` now displays weapon diagnostics:
  - `state`
  - `vis`
  - `inf`
  - `act`
  - `drop`
  - `cand`
  - `miss`

## How To Verify

Run:

```powershell
python tools/debug_observation_overlay.py --phase weapon_acquisition --show --max-steps 1000 --yolo-every 1
```

Expected result:

- before pickup: `player_has_weapon = 0`, `weapon_on_ground = 1`
- around pickup: `cand` rises while near the weapon
- after pickup: `inf=1` for the pickup frame or `act=1` if the key path was captured
- after pickup settles: `player_has_weapon = 1`
- weapon phase should reach `SUCCESS: YES` when `player_has_weapon = 1` and `weapon_dx/dy` are at the neutral no-ground-weapon target

If `player_has_weapon` still fails while gameplay visibly shows a held weapon, save another overlay clip and compare the `weapon diag:` line. The next likely issue would be YOLO classifying a held weapon as a ground `weapons` detection.
