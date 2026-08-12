# Next session — run order

Everything below assumes Brawlhalla in **Training Mode, 1920x1080, borderless windowed**,
same legend and map used for the stage calibration, and:

```powershell
C:\venvs\brawl312\Scripts\Activate.ps1
cd C:\Users\mazbi\OneDrive\Bureau\brawl_scratch
```

Steps 1 and 2 need the game running. Steps 3 onward do not.

---

## 1. Profile the step loop — the measurement everything else waits on

**Game open, character idle on the platform.** ~500 steps is enough.

```powershell
python tools/debug_observation_overlay.py --phase movement_fluency --show --max-steps 500
```

The env prints a per-stage average every 500 inner frames:

```
[BrawlDeepEnv] avg inner frame over 500: total=..ms (..hz), frame=..ms, apply=..ms,
               detect=..ms, memory=..ms, logic=..ms, reward=..ms, other=..ms
```

**Copy that line.** It settles several open questions at once:

| if | then |
|---|---|
| `detect` > 20 ms | drop to `yolo26n-p2` before touching `imgsz` — resolution is what makes the indicator detectable |
| `detect` 10–20 ms | async perception is worth wiring in; expect roughly 2x |
| `detect` < 8 ms | async is optional; the bottleneck is elsewhere |
| `total` far above 25 ms | control rate is under 40 Hz and every wall-clock estimate in the docs needs revising |

Nothing about the async integration should be decided before this line exists.

## 2. Confirm the grounding fix on live frames

Same overlay run. While standing on the platform, check:

- `player_grounded` reads **1** (it read 0 before the foot-convention fix)
- `identity_observed` holds at **1**
- boxes do not swap when you and the bot collide
- `rel_dx` stays **positive** on both sides of the opponent

If `player_grounded` still flickers while standing still, the platform calibration in
`config.py` needs revisiting rather than the foot offset — send a screenshot with the
overlay numbers visible.

## 3. Integrate async perception

Codex delivered `perception/async_pipeline.py` with 9 tests; it is deliberately **not**
wired into `env.py` yet. That integration is one change, gated on step 1 telling us it is
worth making.

## 4. Recollect demos

Every existing NPZ is invalid: the observation schema changed (62-dim core, 146 with
history) and the action space changed (`Discrete(27)`).

```powershell
python -m train.collect_heuristic_curriculum_demos --episodes-per-phase 50
```

Watch the accepted/attempted ratio in the output. A low acceptance rate means the teacher
cannot reach the sampled goals, and the dataset will silently under-represent exactly
those goals — that is worth stopping for, not pushing through.

## 5. Validate the demos before training on them

```powershell
python tools/validate_llc_demos.py --phase all --min-samples 100
```

Bad demos become bad anchors: they are replayed as a behaviour-cloning loss for the whole
PPO run, so a freeze reflex or a missing goal family persists long after collection.

## 6. First real training run

```powershell
python -m train.pretrain_bc_locomotion --phase movement_fluency
python -m train.train_curriculum --phase movement_fluency `
    --resume train/models/llc_movement_fluency_bc_init.zip `
    --log-csv --eval-every-steps 20000
python -m analysis.plot_learning_curves --metric success
```

`movement_fluency` first, not recovery: it is the simplest goal, it exercises
canonicalisation end to end, and if the mirror or the history window is wrong it will show
there fastest.

---

## Open decisions

- **Weapon possession is still action-inferred.** `player_weapon_type` exists in the
  schema but reads from the same inferred bit, so treat those two dimensions as
  placeholders until the crop classifier lands. The reward exploits around it are closed,
  but the underlying signal is still not measured.
- **Action masking** is computed and exposed in `info`, but SB3's PPO cannot consume it.
  Switching to `MaskablePPO` would pick it up for free.
- **`in_strike_range`** kept, per your definition (in range to land or be landed on).
- **PPO update stalls**: `n_steps=2048` at ~40 Hz is a 51-second rollout followed by a
  multi-second optimiser pass *while the game keeps running*. Those frames are recorded as
  deliberate actions. Worth addressing before long runs.

## State of the repo

- 206 tests passing
- `env.py` 1273 -> 758 lines; `capture/`, `control/`, `reward/ui_probe.py`,
  `reward/providers.py` split out
- CI runs the suite plus an entry-point launch check on every push
- MIT licensed, `CITATION.cff` present
