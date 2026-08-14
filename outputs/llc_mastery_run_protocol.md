# LLC Mastery Run Protocol

> **Legacy full-budget protocol.** Do not use these recovery-first commands for
> the Easy-bot v0 pilot. The current, artifact-pinned sequence is
> `experiments/easy_bot_v0/protocol.md`.

Use one fixed setup until LLC retention is stable: same legend, map/platform, bot difficulty, 1920x1080 layout, and calibrated UI/perception. Do not start HSP until all LLC phases pass the retention gate.

For the goal-by-goal metric and plot checklist, use `outputs/llc_goal_evaluation_matrix.md`.

## Gate

After every phase, evaluate current plus previous phases. Advance only when:

- current phase skill score meets its default threshold;
- every previous phase has `amnesia <= 0.15`;
- final `combat_execution` and `all_skills_llc` show positive `mean_damage_trade`;
- visual inspection shows no collapsed movement, no persistent idling, and no attack spam.

Default current-score advance thresholds:

| Phase | Minimum Skill Score |
|---|---:|
| `recovery_mastery` | 0.65 |
| `movement_fluency` | 0.65 |
| `weapon_acquisition` | 0.60 |
| `spacing_neutral` | 0.55 |
| `combat_execution` | 0.50 |
| `all_skills_llc` | 0.60 |

Shared best-score file:

```powershell
$device = "cuda"  # use "cpu" only if CUDA is unavailable
$best = "train/models/llc_retention_best.json"
```

Before any live run, verify the machine/runtime:

```powershell
python -m pip install -r requirements-llc.txt
python tools/llc_preflight.py --device cuda
```

Fix `FAIL` rows before perception or demo collection. If CUDA still fails after installing requirements, reinstall `torch` from the PyTorch selector for your CUDA version or use `$device = "cpu"`. `WARN` rows are acceptable only when you understand the missing capability, such as skipping plots temporarily.

To print the exact command block for any phase instead of copying from this file manually:

```powershell
python tools/print_llc_phase_commands.py --phase movement_fluency
python tools/print_llc_phase_commands.py --phase all_skills_llc
python tools/print_llc_phase_commands.py --phase all
```

The generator prints validation, BC pretraining when needed, PPO training, retention evaluation, gate checking, plotting, and the final phase report command.
By default it defines `$best = "train/models/llc_retention_best.json"` in the printed block; pass `--best-scores $yourVar` only if you already manage a different PowerShell variable.

For day-to-day manual training, use the advisor. It inspects local demo/model/eval/report artifacts and prints the next safe command:

```powershell
python tools/llc_next_action.py
python tools/llc_next_action.py --phase weapon_acquisition
```

If it says `STOP`, do not advance. Fix the demo, perception, score, retention, idle, whiff, or damage-trade issue it reports.

While a PPO phase is running, open a second PowerShell window and watch the live collapse signals:

```powershell
python tools/llc_live_monitor.py --phase movement_fluency
python tools/llc_live_monitor.py --phase combat_execution
```

Use `--once` for a single snapshot. Treat `STOP SIGNALS` as a reason to pause before the next phase; do not ignore high idle, low action entropy, high whiff rate, negative combat trade, or retention/amnesia failures.

Use this checker after each evaluation CSV is written:

```powershell
python tools/check_llc_phase_gate.py --eval-csv train/models/llc_PHASE_retention_eval.csv --phase PHASE
```

It exits with `0` when you may advance and `2` when you should stop, rehearse old demos, or fix perception/reward shaping.

If it fails:

- Missing eval rows: rerun `train.evaluate_retention` with all previous phases included.
- Prior phase amnesia/retention failure: stop the ladder and run a 100k-step rehearsal of the current phase with all demos so far, `--anchor-kl-coef 0.06`, and `--bc-loss-coef 0.12`.
- High idle rate: inspect perception first, then refresh movement demos and keep entropy high for the next rehearsal.
- High combat whiff rate: rehearse spacing before combat, add punish-timing demos, and do not reward random attack spam.
- Negative combat damage trade: do not start HSP; verify damage extraction, collect better combat demos, then rehearse spacing/combat together.

## 1. Perception Sanity Check

```powershell
python tools/debug_observation_overlay.py --phase movement_fluency --show --max-steps 1000 --yolo-every 1
```

Stop here if boxes, stocks, damage, weapon state, or relative positions are wrong.

## 2. Collect Demos

```powershell
python -m train.collect_bc_locomotion_demos --phase recovery_mastery --episodes 20 --max-episode-steps 120
python -m train.collect_bc_locomotion_demos --phase movement_fluency --episodes 40 --max-episode-steps 90 --move-mouse-to-goal
python -m train.collect_bc_locomotion_demos --phase weapon_acquisition --episodes 30 --max-episode-steps 140
python -m train.collect_bc_locomotion_demos --phase spacing_neutral --episodes 30 --max-episode-steps 180
python -m train.collect_bc_locomotion_demos --phase combat_execution --episodes 50 --max-episode-steps 240
```

Expected archives:

- `train/models/recovery_mastery_demos.npz`
- `train/models/movement_fluency_demos.npz`
- `train/models/weapon_acquisition_demos.npz`
- `train/models/spacing_neutral_demos.npz`
- `train/models/combat_execution_demos.npz`

Validate demos before BC pretraining:

```powershell
python tools/validate_llc_demos.py --phase all --min-samples 100
```

If a phase warns about low entropy, high idle, missing goals, wrong phase metadata, or shape mismatch, recollect that phase before training. Bad demos become bad anchors.

## 3. Train Phase Ladder

### Recovery

```powershell
python -m train.pretrain_bc_locomotion --phase recovery_mastery --epochs 20 --output train/models/llc_recovery_mastery_bc_init.zip --device $device

python -m train.train_curriculum --phase recovery_mastery --resume train/models/llc_recovery_mastery_bc_init.zip --timesteps 300000 --model-name llc_recovery_mastery --bc-demos-path "train/models/recovery_mastery_demos.npz" --log-csv --plot-every 10 --eval-every-steps 25000 --eval-episodes 5 --eval-include-previous --retention-scores-path $best --device $device

python -m train.evaluate_retention --model train/models/llc_recovery_mastery.zip --phase recovery_mastery --phases recovery_mastery --best-scores $best --episodes 5 --device $device --csv train/models/llc_recovery_mastery_retention_eval.csv

python tools/check_llc_phase_gate.py --eval-csv train/models/llc_recovery_mastery_retention_eval.csv --phase recovery_mastery --phases recovery_mastery
```

### Movement

```powershell
python -m train.pretrain_bc_locomotion --phase movement_fluency --resume train/models/llc_recovery_mastery.zip --demos train/models/movement_fluency_demos.npz --epochs 20 --output train/models/llc_movement_fluency_bc_init.zip --device $device

python -m train.train_curriculum --phase movement_fluency --resume train/models/llc_movement_fluency_bc_init.zip --timesteps 300000 --model-name llc_movement_fluency --bc-demos-path "train/models/recovery_mastery_demos.npz;train/models/movement_fluency_demos.npz" --log-csv --plot-every 10 --eval-every-steps 25000 --eval-episodes 5 --eval-include-previous --retention-scores-path $best --device $device

python -m train.evaluate_retention --model train/models/llc_movement_fluency.zip --phase movement_fluency --phases recovery_mastery,movement_fluency --best-scores $best --episodes 5 --device $device --csv train/models/llc_movement_fluency_retention_eval.csv

python tools/check_llc_phase_gate.py --eval-csv train/models/llc_movement_fluency_retention_eval.csv --phase movement_fluency
```

### Weapon

```powershell
python -m train.pretrain_bc_locomotion --phase weapon_acquisition --resume train/models/llc_movement_fluency.zip --demos train/models/weapon_acquisition_demos.npz --epochs 20 --output train/models/llc_weapon_acquisition_bc_init.zip --device $device

python -m train.train_curriculum --phase weapon_acquisition --resume train/models/llc_weapon_acquisition_bc_init.zip --timesteps 350000 --model-name llc_weapon_acquisition --bc-demos-path "train/models/recovery_mastery_demos.npz;train/models/movement_fluency_demos.npz;train/models/weapon_acquisition_demos.npz" --log-csv --plot-every 10 --eval-every-steps 25000 --eval-episodes 5 --eval-include-previous --retention-scores-path $best --device $device

python -m train.evaluate_retention --model train/models/llc_weapon_acquisition.zip --phase weapon_acquisition --phases recovery_mastery,movement_fluency,weapon_acquisition --best-scores $best --episodes 5 --device $device --csv train/models/llc_weapon_acquisition_retention_eval.csv

python tools/check_llc_phase_gate.py --eval-csv train/models/llc_weapon_acquisition_retention_eval.csv --phase weapon_acquisition
```

### Spacing

```powershell
python -m train.pretrain_bc_locomotion --phase spacing_neutral --resume train/models/llc_weapon_acquisition.zip --demos train/models/spacing_neutral_demos.npz --epochs 20 --output train/models/llc_spacing_neutral_bc_init.zip --device $device

python -m train.train_curriculum --phase spacing_neutral --resume train/models/llc_spacing_neutral_bc_init.zip --timesteps 350000 --model-name llc_spacing_neutral --bc-demos-path "train/models/recovery_mastery_demos.npz;train/models/movement_fluency_demos.npz;train/models/weapon_acquisition_demos.npz;train/models/spacing_neutral_demos.npz" --log-csv --plot-every 10 --eval-every-steps 25000 --eval-episodes 5 --eval-include-previous --retention-scores-path $best --device $device

python -m train.evaluate_retention --model train/models/llc_spacing_neutral.zip --phase spacing_neutral --phases recovery_mastery,movement_fluency,weapon_acquisition,spacing_neutral --best-scores $best --episodes 5 --device $device --csv train/models/llc_spacing_neutral_retention_eval.csv

python tools/check_llc_phase_gate.py --eval-csv train/models/llc_spacing_neutral_retention_eval.csv --phase spacing_neutral
```

### Combat

```powershell
python -m train.pretrain_bc_locomotion --phase combat_execution --resume train/models/llc_spacing_neutral.zip --demos train/models/combat_execution_demos.npz --epochs 20 --output train/models/llc_combat_execution_bc_init.zip --device $device

python -m train.train_curriculum --phase combat_execution --resume train/models/llc_combat_execution_bc_init.zip --timesteps 500000 --model-name llc_combat_execution --bc-demos-path "train/models/recovery_mastery_demos.npz;train/models/movement_fluency_demos.npz;train/models/weapon_acquisition_demos.npz;train/models/spacing_neutral_demos.npz;train/models/combat_execution_demos.npz" --log-csv --plot-every 10 --eval-every-steps 25000 --eval-episodes 5 --eval-include-previous --retention-scores-path $best --device $device

python -m train.evaluate_retention --model train/models/llc_combat_execution.zip --phase combat_execution --phases recovery_mastery,movement_fluency,weapon_acquisition,spacing_neutral,combat_execution --best-scores $best --episodes 5 --device $device --csv train/models/llc_combat_execution_retention_eval.csv

python tools/check_llc_phase_gate.py --eval-csv train/models/llc_combat_execution_retention_eval.csv --phase combat_execution
```

### Consolidation

```powershell
python -m train.train_curriculum --phase all_skills_llc --resume train/models/llc_combat_execution.zip --timesteps 500000 --model-name llc_all_skills_llc --bc-demos-path "train/models/recovery_mastery_demos.npz;train/models/movement_fluency_demos.npz;train/models/weapon_acquisition_demos.npz;train/models/spacing_neutral_demos.npz;train/models/combat_execution_demos.npz" --log-csv --plot-every 10 --eval-every-steps 25000 --eval-episodes 5 --eval-phases all --retention-scores-path $best --device $device

python -m train.evaluate_retention --model train/models/llc_all_skills_llc.zip --phase all_skills_llc --phases all --best-scores $best --episodes 10 --device $device --csv train/models/llc_all_skills_llc_retention_eval.csv

python tools/check_llc_phase_gate.py --eval-csv train/models/llc_all_skills_llc_retention_eval.csv --phase all_skills_llc --phases all
```

## 4. Plot Evidence After Each Phase

Replace the prefix/model name after each phase:

```powershell
python tools/plot_llc_diagnostics.py --steps-csv train/models/llc_all_skills_llc_steps.csv --episodes-csv train/models/llc_all_skills_llc_episodes.csv --eval-csv train/models/llc_all_skills_llc_eval.csv --prefix llc_all_skills_llc
```

Inspect:

- `*_retention_amnesia.png`: old skills must stay above 0.85 retention and below 0.15 amnesia.
- `*_goal_family_errors.png`: recovery, movement, weapon, spacing, and combat errors should trend down or remain stable during consolidation.
- `*_goal_feature_traces.png`: per-feature error traces for ledge dx/dy, player x/y, weapon dx/dy, spacing band, strike range, and frame advantage.
- `*_goal_phase_spaces.png`: recovery ledge-offset phase space, movement trajectory-to-target, weapon offset convergence, spacing band phase plot, and combat frame-advantage histogram.
- `*_episode_health.png`: success, action entropy, idle rate, whiff rate, and damage trade.
- `*_combat_precision.png`: attack precision should rise, whiff rate should fall, and damage trade should become positive.

## 5. Write Phase Report

After eval/gate/plot, generate one Markdown report for the phase:

```powershell
python tools/summarize_llc_run.py --phase all_skills_llc --phases all --eval-csv train/models/llc_all_skills_llc_retention_eval.csv --prefix llc_all_skills_llc --out outputs/llc_all_skills_llc_run_report.md
```

Then record your visual approval. The advisor will not advance while this is missing or negative:

```powershell
python tools/record_llc_observation.py --phase all_skills_llc --approved yes --notes "movement, recovery, weapon pickup, spacing, and combat look retained"
```

Use `--approved no` when your eyes disagree with the metrics. That blocks advancement and preserves the reason.

## 6. HSP Is Deferred

Do not train HSP until `all_skills_llc` passes `tools/check_llc_phase_gate.py --phase all_skills_llc --phases all` and your visual notes agree.

The HSP trainer now enforces that:

```powershell
python -m train.train_phase3_hsp --llc train/models/llc_all_skills_llc.zip --llc-retention-csv train/models/llc_all_skills_llc_retention_eval.csv
```

`--allow-legacy-hsp` exists only as an explicit override, because current HSP still uses the legacy `hierarchical/goals.py` goal space rather than the current 11-feature curriculum space.
