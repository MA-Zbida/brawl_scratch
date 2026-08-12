# Training Workflow

All training runs against the live game. Brawlhalla must be focused, in Training Mode, at 1920×1080,
with the same legend, map and bot difficulty used for calibration. Changing any of these invalidates
the stage geometry and UI probes in [`config.py`](../config.py).

## 0. Preflight

```powershell
python tools/llc_preflight.py --device cuda
```

Then verify perception. Nothing downstream is meaningful if the overlay disagrees with the screen:

```powershell
python tools/debug_observation_overlay.py --phase movement_fluency --show --max-steps 1000
```

Check boxes, stocks, damage, weapon state, platform position and relative geometry before continuing.

## 1. Collect demonstrations

Scripted teacher, one phase:

```powershell
python -m train.collect_bc_locomotion_demos --phase movement_fluency --teacher heuristic --episodes 30
```

Whole curriculum in one run:

```powershell
python -m train.collect_heuristic_curriculum_demos --episodes-per-phase 50
```

Human demonstrations (environment key injection disabled; your keyboard is recorded as labels):

```powershell
python -m train.collect_bc_locomotion_demos --phase combat_execution --teacher manual --episodes 30
```

Only episodes meeting the phase acceptance rule are saved. Output goes to
`train/models/<phase>_demos.npz`.

Validate before training on them:

```powershell
python tools/validate_llc_demos.py --phase all --min-samples 100
```

Bad demonstrations become bad anchors — they are replayed as a BC loss for the entire PPO run.
Recollect any phase with high idle rate, low action entropy, missing goal masks, or wrong metadata.

## 2. Behaviour cloning

```powershell
python -m train.pretrain_bc_locomotion --phase movement_fluency
```

Hindsight goal relabelling is on by default: for each recorded step, the state reached `h` steps later
is treated as the goal that step was pursuing, over horizons `1,2,4,8`. This multiplies the effective
dataset and teaches the policy to interpret goals it was never explicitly given. Horizons longer than
the median episode length are dropped automatically.

Produces `train/models/llc_<phase>_bc_init.zip`.

## 3. PPO fine-tuning

```powershell
python -m train.train_curriculum --phase movement_fluency --resume train/models/llc_movement_fluency_bc_init.zip
```

Demonstration archives are auto-discovered for the BC auxiliary loss; `all_skills_llc` picks up every
prior phase archive. Relevant knobs:

| Flag | Purpose |
|---|---|
| `--replay-ratio` | fraction of each minibatch drawn from replay |
| `--anchor-kl-coef` | strength of KL anchoring to policy snapshots |
| `--anchor-pool-size` | number of retained snapshots |
| `--bc-loss-coef` | weight of the demonstration log-likelihood term |
| `--pcgrad` / `--no-pcgrad` | gradient surgery between PPO and auxiliary losses |
| `--eval-every-steps` | periodic retention evaluation |
| `--eval-include-previous` | evaluate all earlier phases for amnesia |

> **Resource warning.** Periodic evaluation constructs a full environment per evaluated phase, each
> with its own capture instance and TensorRT engine. On 4 GB VRAM, `--eval-include-previous` on later
> phases can exhaust memory.

> **Real-time warning.** The game does not pause during PPO updates. With the default `n_steps=2048`
> at roughly 40 steps/s, the agent is idle for the duration of each update while the match continues.

## 4. Monitor

In a second terminal:

```powershell
python tools/llc_live_monitor.py --phase movement_fluency
```

Treat reported stop signals as real: high idle rate, collapsing action entropy, high whiff rate,
negative damage trade, or a retention failure.

## 5. Evaluate and gate

```powershell
python -m train.evaluate_retention --phase movement_fluency
python tools/check_llc_phase_gate.py --eval-csv train/models/llc_movement_fluency_retention_eval.csv --phase movement_fluency
python tools/plot_llc_diagnostics.py --phase movement_fluency
python tools/summarize_llc_run.py --phase movement_fluency
```

Record your own visual judgement, which can veto the metrics:

```powershell
python tools/record_llc_observation.py --phase movement_fluency --approved yes --notes "what you saw"
```

## 6. Advance

```powershell
python tools/llc_next_action.py
```

The advisor inspects demonstrations, checkpoints, eval CSVs, plots and manual approvals. If it reports
`STOP`, do not advance.

## Artefacts produced

| Path | Contents |
|---|---|
| `train/models/<phase>_demos.npz` | demonstrations with goals, masks, metadata |
| `train/models/llc_<phase>_bc_init.zip` | BC-initialised policy |
| `train/models/llc_<phase>.zip` | fine-tuned policy |
| `train/models/llc_<phase>_steps.csv` | per-step rewards, goal errors, actions, events |
| `train/models/llc_<phase>_episodes.csv` | per-episode returns, success, entropy, whiff rate |
| `train/models/llc_<phase>_eval.csv` | per-phase skill score, retention, amnesia |
| `train/models/llc_retention_best.json` | best score per phase, for retention math |
| `outputs/llc_<phase>_run_report.md` | generated run report |

Figures for the report are regenerated from these CSVs by [`analysis/`](../analysis/).
