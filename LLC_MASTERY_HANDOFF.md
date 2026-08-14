# Brawlhalla LLC Mastery Handoff

> **Superseded run order.** This document describes the earlier recovery-first
> ladder. Use `experiments/easy_bot_v0/protocol.md` for the current bounded pilot;
> the tested canonical order now defers recovery until after combat.

This branch turns the LLC into a measured skill ladder with anti-collapse tooling. It does not claim the agent is already a master; live training still has to be run by you in Brawlhalla, then judged by metrics and by your own visual approval.

## What Changed

- Added the current LLC phase ladder:
  - `recovery_mastery`
  - `movement_fluency`
  - `weapon_acquisition`
  - `spacing_neutral`
  - `combat_execution`
  - `all_skills_llc`
- Added `all_skills_llc`, a consolidation phase that samples all five goal families.
- Added retention/amnesia math:
  - `retention_i(t) = current_score_i(t) / best_score_i_so_far`
  - `amnesia_i(t) = max(0, 1 - retention_i(t))`
  - previous phases must stay at `retention >= 0.85` and `amnesia <= 0.15`.
- Extended LLC logging:
  - step CSV: actions, active goal errors, raw goal features, goal target/mask, reward components, death/weapon/combat flags.
  - episode CSV: success, mean error, time-to-success, damage trade, action entropy, idle rate, whiff rate, attack precision.
  - eval CSV: per-phase skill score, best score, retention, amnesia, win/damage/combat signals.
- Upgraded anti-collapse PPO support:
  - multi-demo BC archives through `--bc-demos-path`
  - balanced BC sampling across demo archives
  - snapshot-pool KL anchoring
  - replay/anchor/PCGrad CLI knobs
- Added tools:
  - `tools/llc_preflight.py`
  - `tools/llc_next_action.py`
  - `tools/llc_live_monitor.py`
  - `tools/validate_llc_demos.py`
  - `tools/check_llc_phase_gate.py`
  - `tools/evaluate_retention.py` as `python -m train.evaluate_retention`
  - `tools/plot_llc_diagnostics.py`
  - `tools/summarize_llc_run.py`
  - `tools/record_llc_observation.py`
  - `tools/print_llc_phase_commands.py`
- Deferred HSP until the LLC passes `all_skills_llc` retention and manual approval. The HSP trainer now refuses to run unless the all-skills retention CSV passes, unless you explicitly use `--allow-legacy-hsp`.

## Setup

Use a fixed setup until the LLC is stable:

- one legend
- one map/platform calibration
- one bot difficulty
- 1920x1080 UI layout
- Brawlhalla focused during live collection/training

Install the live stack:

```powershell
python -m pip install -r requirements-llc.txt
python tools/llc_preflight.py --device cuda
```

If CUDA fails after installing requirements, reinstall `torch` from the official PyTorch selector for your CUDA version, or run with `--device cpu`.

## Main Run Loop

The safest way to operate this project is to ask the advisor what to do next:

```powershell
python tools/llc_next_action.py
```

It inspects local demos, checkpoints, eval CSVs, plots, reports, and manual observation approvals. If it says `STOP`, do not advance.

To print every command in the ladder:

```powershell
python tools/print_llc_phase_commands.py --phase all
```

## Perception Check

Before demos or training:

```powershell
python tools/debug_observation_overlay.py --phase movement_fluency --show --max-steps 1000 --yolo-every 1
```

Do not continue if boxes, stocks, damage, weapon state, platform position, or relative positions are wrong.

## Collect Demos

```powershell
python -m train.collect_bc_locomotion_demos --phase recovery_mastery --episodes 20 --max-episode-steps 120
python -m train.collect_bc_locomotion_demos --phase movement_fluency --episodes 40 --max-episode-steps 90 --move-mouse-to-goal
python -m train.collect_bc_locomotion_demos --phase weapon_acquisition --episodes 30 --max-episode-steps 140
python -m train.collect_bc_locomotion_demos --phase spacing_neutral --episodes 30 --max-episode-steps 180
python -m train.collect_bc_locomotion_demos --phase combat_execution --episodes 50 --max-episode-steps 240
```

Validate before BC:

```powershell
python tools/validate_llc_demos.py --phase all --min-samples 100
```

Bad demos become bad anchors. Recollect phases with high idle, low entropy, missing goal masks, wrong phase metadata, or no combat attacks.

## Train Each Phase

For each phase:

1. BC pretrain.
2. PPO fine-tune with all demos learned so far.
3. Run retention evaluation.
4. Run gate checker.
5. Generate plots.
6. Generate a phase report.
7. Record your manual visual approval.

Use:

```powershell
python tools/llc_next_action.py --phase recovery_mastery
python tools/llc_next_action.py --phase movement_fluency
python tools/llc_next_action.py --phase weapon_acquisition
python tools/llc_next_action.py --phase spacing_neutral
python tools/llc_next_action.py --phase combat_execution
python tools/llc_next_action.py --phase all_skills_llc
```

While PPO is running, monitor collapse signals in another terminal:

```powershell
python tools/llc_live_monitor.py --phase movement_fluency
python tools/llc_live_monitor.py --phase combat_execution
```

Treat `STOP SIGNALS` as real. Pause before advancing if the monitor reports high idle, low entropy, high whiff rate, negative combat trade, or retention/amnesia failure.

## Subgoal Metrics

| Goal | Features | Pass Evidence |
|---|---|---|
| Recovery | `signed_dx_to_ledge`, `dy_to_ledge` | recovery success, low death rate, falling ledge error, good time-to-stage |
| Movement | `player_x`, `player_y` | target success, low target error, low idle, healthy action entropy |
| Weapon | `player_has_weapon`, `weapon_dx`, `weapon_dy` | pickup success, low time-to-pickup, low drop rate |
| Spacing | `rel_distance`, `rel_dy` | desired-band occupancy, vertical alignment, low self-damage |
| Combat | `in_strike_range`, `frame_advantage_estimate` | hit rate, attack precision, low whiff, positive damage trade, win rate |
| Consolidation | all families | all previous retention gates pass, no visible skill collapse |

More detail lives in `outputs/llc_goal_evaluation_matrix.md`.

## Gates

Default current-phase minimum skill scores:

| Phase | Minimum |
|---|---:|
| `recovery_mastery` | 0.65 |
| `movement_fluency` | 0.65 |
| `weapon_acquisition` | 0.60 |
| `spacing_neutral` | 0.55 |
| `combat_execution` | 0.50 |
| `all_skills_llc` | 0.60 |

Run gate manually when needed:

```powershell
python tools/check_llc_phase_gate.py --eval-csv train/models/llc_PHASE_retention_eval.csv --phase PHASE
```

Advance only when:

- the current phase reaches its skill score threshold;
- all previous phases have `amnesia <= 0.15`;
- all previous phases have `retention >= 0.85`;
- combat/all-skills damage trade is non-negative;
- visual observation approves advancement.

## Manual Approval

After plots and report:

```powershell
python tools/record_llc_observation.py --phase PHASE --approved yes --notes "what you saw"
```

Use `--approved no` if your eyes disagree with the metrics. The advisor will block advancement.

## HSP

Do not train HSP until the final LLC passes all gates and manual approval:

```powershell
python -m train.train_phase3_hsp --llc train/models/llc_all_skills_llc.zip --llc-retention-csv train/models/llc_all_skills_llc_retention_eval.csv
```

The override `--allow-legacy-hsp` exists, but it is intentionally not the recommended path because HSP still uses the older hierarchical goal space.

## What To Bring Back

After a training run, bring back:

- `train/models/llc_<phase>_retention_eval.csv`
- `train/models/llc_<phase>_eval.csv`
- `train/models/llc_<phase>_episodes.csv`
- `train/models/llc_<phase>_steps.csv` if not too large
- generated plots: `train/models/llc_<phase>_*.png`
- `outputs/llc_<phase>_run_report.md`
- `outputs/llc_<phase>_manual_observation.json`
- your plain-language observation: what looked cracked, what collapsed, and what the agent did that surprised you.

Then I can inspect the evidence and tune the next phase or anti-collapse parameters.
