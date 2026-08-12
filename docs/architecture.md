# Architecture

## Control loop

One environment step, as implemented in `BrawlDeepEnv.step`
([`env.py`](../env.py)):

1. **Capture** — `DxcamFrameProvider.get_frame()` returns the most recent desktop frame. Capture runs
   non-blocking in video mode, so a step never waits for a fresh frame; if none is ready the previous
   frame is reused.
2. **Act** — the sanitised action is injected via `pydirectinput`. Movement keys are *held*
   (`set_pressed`); jump, dodge and attack are *tapped*.
3. **Detect** — the frame is resized to 640×360 and passed to the YOLO TensorRT engine.
4. **Update state** — `Memory.update_from_detections` associates detections to fighters, integrates
   velocities, and updates stage-relative geometry.
5. **Read UI** — fixed pixel probes decode stocks and damage.
6. **Reward** — the active wrapper computes goal error and shaping terms.
7. **Observe** — `Memory.to_vector()` produces the observation.

An action may be repeated over several inner frames (`action_repeat_*`), with tap-type inputs emitted
only on the first frame of the repeat.

## Perception

The detector is an Ultralytics YOLO model exported to TensorRT
([`feature_extractor/yolo/extract.py`](../feature_extractor/yolo/extract.py)).

**Agent identity is resolved causally.** The agent is the character carrying the blue self-indicator
triangle, not a character identified by legend appearance. This keeps the labelling scheme valid when
legends are swapped and when the opponent pool grows — a legend-identity scheme would need relabelling
for every new character.

Identity resolution lives in
[`feature_extractor/memory/detection_schema.py`](../feature_extractor/memory/detection_schema.py). The
schema is inferred per frame, so the legacy 5-class engine keeps working until the retrained weights
are swapped in.

When the indicator is not detected — occluded, clipped at a screen edge — the agent falls back to
nearest-to-last-position and `identity_observed` reports that the answer was **carried forward rather
than measured**. There is still no Kalman filter or Hungarian assignment, so purely positional
association can swap identities when fighters overlap, which is exactly the situation in
[`assets/screenshots/stage-reference-1080p.png`](../assets/screenshots/stage-reference-1080p.png) —
the indicator is what disambiguates it.

## State estimation

`Memory` is the single source of truth for game state. It maintains:

- `FighterState` for the agent and opponent (position, velocity, bounding-box extent, grounded, edge,
  offstage, damage, stocks, jump count, dodge timers)
- `WeaponState` for the nearest visible ground weapon
- fixed `Platform` geometry and `Physics` constants

Stage geometry (`PLATFORM_BOUNDS` in [`config.py`](../config.py)) is a manual calibration for one map
at one resolution. Grounded, edge, offstage and ledge-distance features are all computed against it.

See [observation-space.md](observation-space.md) for which state variables are measured and which are
simulated — the distinction is load-bearing.

## Reward side-channel

Damage and stocks are not available from the detector, so they are decoded from fixed UI pixel
coordinates (`UI_REGIONS`):

- `reward/extract_rgb.py` samples the pixel
- `reward/rgb_to_dmg.py` maps colour to a damage value
- `reward/stock.py` detects the stock-loss colour flash

`PixelStocksHealthProvider` debounces stock events with a confirmation window, a cooldown, and a
per-side lock, because the flash persists across several frames.

## Policy

`StageGoalEnv` augments the observation to `[state | goal_target | goal_mask]`, then
`StageGoalFiLMExtractor` ([`feature_extractor/film_extractor.py`](../feature_extractor/film_extractor.py)):

1. extracts the goal-relevant state features and normalises them to a common range
2. computes the masked goal error and a velocity-weighted alignment signal
3. encodes `[state, masked_error, mask, alignment]` through an MLP
4. encodes the goal vector separately and produces FiLM `γ`/`β`
5. applies `γ · φ(s) + β + φ(s)` (clamped, with residual)

This is a UVFA: one network conditioned on a goal, rather than one network per skill.

## Action space

`MultiDiscrete([4, 2, 2, 4])`:

| Channel | Values | Semantics |
|---|---|---|
| movement | `A` / `D` / `S` / idle | held until changed |
| jump | 0 / 1 | tap |
| dodge | 0 / 1 | tap |
| attack | none / NUM4 / NUM6 / NUM5 | tap |

Tap-type inputs are **held** across a multi-step latch window rather than pulsed. A press-release
pair inside one step is shorter than the game's ~16.7 ms input poll and is frequently dropped.

Movement is expressed in the canonical frame, so it is flipped back before injection whenever the
observation was mirrored.

Dodge and attack are mutually exclusive (`_sanitize_action`). Stage specs can further restrict the
attack channel via `disable_attack` or `allowed_attack_actions`.

**Limitation.** There is no direction modifier on attacks, so side, down, up and aerial variants —
most of the Brawlhalla moveset — cannot be expressed. The agent can only throw neutral lights and
heavies.

## Training algorithm

`AnchoredReplayPPO` ([`algo/anchored_replay_ppo.py`](../algo/anchored_replay_ppo.py)) extends
Stable-Baselines3 PPO with four mechanisms aimed at catastrophic forgetting:

- **Replay mixing** — each minibatch blends on-policy samples with stored transitions, sampled with
  priority proportional to positive advantage.
- **Snapshot-pool KL anchoring** — a pool of past policy snapshots; each update penalises divergence
  from a randomly chosen snapshot.
- **Behaviour-cloning auxiliary loss** — demonstration archives contribute a log-likelihood term.
- **PCGrad** — when the auxiliary gradient conflicts with the PPO gradient, the PPO gradient is
  projected onto the orthogonal complement before summing.

Neither the BC coefficient nor the KL coefficient is annealed, so these terms retain their full
strength for the whole run.
