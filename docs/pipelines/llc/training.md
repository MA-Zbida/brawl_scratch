# Training Workflow

All training runs against the live game. Brawlhalla must be focused, in Training Mode, at 1920×1080,
with the same legend, map and bot difficulty used for calibration. Changing any of these invalidates
the stage geometry and UI probes in [`brawl/calibration.py`](../../../brawl/calibration.py).

## 0. Preflight

```powershell
python scripts/llc/llc_preflight.py --device cuda
```

Then verify perception. Nothing downstream is meaningful if the overlay disagrees with the screen:

```powershell
python scripts/llc/debug_observation_overlay.py --phase movement_fluency --show --max-steps 1000
```

Check boxes, stocks, damage, weapon state, platform position and relative geometry before continuing.

## Stage A — goal-conditioned skill pretraining

Goals are dense competency scaffolding used only in this stage. The skill policy
observes `[state/history | target | mask]`; it is evaluated by per-goal competence
and retention, not by final match win rate.

### 1. Collect demonstrations

Scripted teacher, one phase:

```powershell
python -m brawl.pipelines.llc.collect_demos --phase movement_fluency --teacher heuristic --episodes 30
```

Whole curriculum in one run:

```powershell
python -m brawl.pipelines.llc.collect_heuristic_demos --episodes-per-phase 50
```

Human demonstrations (environment key injection disabled; your keyboard is recorded as labels):

```powershell
python -m brawl.pipelines.llc.collect_demos --phase combat_execution --teacher manual --episodes 30
```

Only episodes meeting the phase acceptance rule are saved. Output goes to
`runs/legacy/llc/<phase>_demos.npz`.

Validate before training on them:

```powershell
python scripts/llc/validate_llc_demos.py --phase all --min-samples 100
```

Bad demonstrations become bad anchors — they are replayed as a BC loss for the entire PPO run.
Recollect any phase with high idle rate, low action entropy, missing goal masks, or wrong metadata.

### 2. Behaviour cloning

```powershell
python -m brawl.pipelines.llc.pretrain_bc --phase movement_fluency
```

Hindsight goal relabelling is on by default: for each recorded step, the state reached `h` steps later
is treated as the goal that step was pursuing, over horizons `1,2,4,8`. This multiplies the effective
dataset and teaches the policy to interpret goals it was never explicitly given. Horizons longer than
the median episode length are dropped automatically.

Produces `runs/legacy/llc/llc_<phase>_bc_init.zip`.

### 3. PPO fine-tuning

```powershell
python -m brawl.pipelines.llc.train_curriculum --phase movement_fluency --resume runs/legacy/llc/llc_movement_fluency_bc_init.zip
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

### 4. Monitor

In a second terminal:

```powershell
python scripts/llc/llc_live_monitor.py --phase movement_fluency
```

Treat reported stop signals as real: high idle rate, collapsing action entropy, high whiff rate,
negative damage trade, or a retention failure.

### 5. Evaluate and gate

```powershell
python -m brawl.pipelines.llc.evaluate_retention --phase movement_fluency
python scripts/llc/check_llc_phase_gate.py --eval-csv runs/legacy/llc/llc_movement_fluency_retention_eval.csv --phase movement_fluency
python scripts/llc/plot_llc_diagnostics.py --phase movement_fluency
python scripts/llc/summarize_llc_run.py --phase movement_fluency
```

Record your own visual judgement, which can veto the metrics:

```powershell
python scripts/llc/record_llc_observation.py --phase movement_fluency --approved yes --notes "what you saw"
```

### 6. Advance

```powershell
python scripts/llc/llc_next_action.py
```

The advisor inspects demonstrations, checkpoints, eval CSVs, plots and manual approvals. If it reports
`STOP`, do not advance.

## Stage B — goal-independent final agent

After all five families and consolidation pass retention, transfer the compatible
representation and policy weights into a model whose observation is the raw
146-feature state/history vector:

```powershell
python -m brawl.pipelines.final_agent.transfer_skill_checkpoint --skill-checkpoint runs/legacy/llc/llc_all_skills_llc.zip --output runs/legacy/final_agent/final_agent_init.zip --device cpu
```

The command writes a typed checkpoint sidecar and a tensor-by-tensor transfer
report. Goal encoders and FiLM parameters are deliberately skipped. Legacy skill
archives without metadata require the explicit
`--allow-untyped-skill-checkpoint` flag.

Fine-tune one policy directly on match reward:

```powershell
python -m brawl.pipelines.final_agent.run --resume runs/legacy/final_agent/final_agent_init.zip --save-dir runs/legacy/final_agent --model-name final_agent --device cpu
```

Evaluate the same checkpoint with no goal scheduler:

```powershell
python -m brawl.pipelines.final_agent.run_match --model runs/legacy/final_agent/final_agent.zip --matches 30 --device cpu --csv runs/legacy/final_agent/matches.csv
```

Final evaluation reports wins, stock differential, damage dealt/received,
offstage rate, environment steps, and real interaction seconds.

## Artefacts produced

| Path | Contents |
|---|---|
| `runs/legacy/llc/<phase>_demos.npz` | demonstrations with goals, masks, metadata |
| `runs/legacy/llc/llc_<phase>_bc_init.zip` | BC-initialised policy |
| `runs/legacy/llc/llc_<phase>.zip` | fine-tuned policy |
| `runs/legacy/llc/llc_<phase>.zip.metadata.json` | typed skill-checkpoint role and observation contract |
| `runs/legacy/llc/llc_<phase>_steps.csv` | per-step rewards, goal errors, actions, events |
| `runs/legacy/llc/llc_<phase>_episodes.csv` | per-episode returns, success, entropy, whiff rate |
| `runs/legacy/llc/llc_<phase>_eval.csv` | per-phase skill score, retention, amnesia |
| `runs/legacy/llc/llc_retention_best.json` | best score per phase, for retention math |
| `runs/legacy/llc/llc_<phase>_run_report.md` | generated run report |
| `runs/legacy/final_agent/final_agent*.zip` | goal-independent final initializer/checkpoint |
| `runs/legacy/final_agent/final_agent*.zip.metadata.json` | typed final-agent role and provenance |
| `runs/legacy/final_agent/final_agent*.zip.transfer.json` | copied/skipped tensor report |
| `runs/legacy/final_agent/matches.csv` | full-match metrics and wall-clock interaction time |

Figures for the report are regenerated from these CSVs by [`brawl/analysis/`](../../../brawl/analysis/).
