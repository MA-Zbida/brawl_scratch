# Codex task — make the full training loop runnable end to end

## The problem

Two days of attempts, no completed run. Every failure was a different one, each
found only by crashing mid-run after hours of gameplay had already been spent.
None were failures of the RL method — all were plumbing:

| # | Failure | Found after |
|---|---|---|
| 1 | Collector accepted 0 episodes: visual-confirmation gate could never be satisfied with the classifier off | a full collection run |
| 2 | BC crashed: `StageGoalEnv` refused to build for a phase requiring visual confirmation, though BC never steps the env | ~2 h |
| 3 | Retention eval crashed: same guard, third entry point | ~2 h |
| 4 | Retention eval crashed: `dxcam` camera borrowed but never returned, so the 2nd phase's env collided with the per-process singleton | ~2 h |
| 5 | Run aborted on an *informational warning*: the driver script used `$ErrorActionPreference = "Stop"` with stream merging, and PowerShell wraps native-command stderr in ErrorRecords | ~2 h |

All five are now fixed (418 tests pass). But the pattern is the point: **the pipeline
has no way to detect its own breakage without spending hours of real gameplay.**
That is what needs fixing, not any individual bug.

## What to build

### 1. A preflight command that exercises every stage without the game

`tools/preflight_pipeline.py`. It must construct and tear down the real objects
each stage uses — not mocks — and fail loudly on any problem, in seconds:

- Build a `BrawlDeepEnv` and a `StageGoalEnv` **for every phase** in `PHASES`,
  with and without a weapon classifier. Failure #2 and #3 were exactly this and
  would have been caught instantly.
- Build **three environments in sequence in one process**, closing each. This
  reproduces failure #4; the retention evaluator builds one env per phase.
- Load every checkpoint referenced by `experiments/easy_bot_v0/config.json` and
  assert its role metadata and observation dimension.
- Validate every demo archive the protocol references.
- Assert each phase's `allowed_actions` is non-empty and within `ACTION_DIM`.

It may capture the desktop rather than the game — the point is object lifecycle,
not gameplay. Exit non-zero with a specific message naming the failing stage.

### 2. A driver that cannot be killed by stderr

`experiments/easy_bot_v0/run_chain.ps1`, replacing the ad-hoc scripts:

- **`$ErrorActionPreference = "Continue"`.** PowerShell wraps a native command's
  stderr in ErrorRecords, so with `Stop` any warning aborts a run that exited 0.
  This caused failure #5. Exit codes are the only failure signal.
- Reset `$global:LASTEXITCODE = 0` before each stage so a stale code from an
  earlier command cannot be misread as the current stage failing.
- Log stdout **and** stderr to a UTF-8 file. Failures #2 and #3 printed tracebacks
  that never reached the log — `Tee-Object` wrote UTF-16 and dropped stderr — so
  each had to be reproduced by hand before it could even be read.
- Run preflight first and abort if it fails.
- **Resumable**: a `--from <stage>` argument, and a stage-completion marker file
  per stage so a re-run skips finished work. Two hours of weapon PPO were nearly
  lost twice because the only recovery was a hand-written resume script.

### 3. A smoke mode

`--smoke` on the driver: run the entire chain with tiny budgets (2 episodes per
collection, 1 BC epoch, 512 PPO timesteps, 1 retention episode). Target under
10 minutes. This is what proves the loop closes before committing 4+ hours.

Every failure above would have been caught by a 10-minute smoke run.

## Constraints

- Interpreter: **`C:\venvs\brawl312\Scripts\python.exe`**. Not any other Python.
- `python -m pytest tests\ -q` currently passes **418**. It must still pass.
- Do not change the RL method: reward shaping, action space, observation spec,
  network architecture, and hyperparameters stay as they are. This task is
  operational reliability only.
- Do not enable the weapon classifier. The trained checkpoint scores mean
  P(armed) = 0.873 on unarmed live crops (train/serve skew: it learned from
  960×960 JPEG exports and is served raw 1920×1080 captures). Possession must
  keep falling back to action inference until it is retrained on live crops.
- Preserve `train/models/easy_bot_v0/llc_weapon_acquisition.zip` — 200k PPO steps
  from clean demos, the first genuine weapon skill in the project.

## Current chain

Movement is the base (`llc_movement_fluency.zip`, already trained, not re-run).
Then, each resuming the previous checkpoint:

```
recovery  : BC only from the existing archive   (no PPO -- episodes start onstage,
                                                 and the only path to the +2.5
                                                 success bonus runs through going
                                                 offstage against a -1.0 death
                                                 penalty, so PPO can learn that
                                                 self-endangerment pays)
weapon    : collect -> BC -> PPO 200k -> retention   [DONE, preserve]
spacing   : collect -> BC -> PPO 200k -> retention
combat    : collect -> BC -> PPO 200k -> retention across all phases
```

`--bc-demos-path` accumulates every prior archive; that anchoring is the
anti-forgetting mechanism and the reason phase order matters.

Measured rate: **30.5 Hz** during PPO (200k steps in 1h49m). Collection runs
~47 Hz because a scripted teacher needs no forward pass and no optimiser pause.
Use 30.5 Hz for any PPO wall-clock estimate.

## Definition of done

1. `python tools\preflight_pipeline.py` passes in under 60 seconds.
2. `run_chain.ps1 --smoke` completes the whole chain in under 10 minutes.
3. `run_chain.ps1 --from spacing` resumes without redoing weapon.
4. A deliberately introduced stderr warning in any stage does **not** abort the run.
5. A deliberately introduced non-zero exit **does** abort it, with the failing
   stage named and the traceback present in the log file.
6. 418 tests still pass.

Items 4 and 5 are the ones that matter. Please verify them by actually injecting
the faults, not by inspection.

## Report back

Per item: what changed, which test or check covers it, and the command to run it.
If you disagree with any of this, say so rather than silently doing something else.
