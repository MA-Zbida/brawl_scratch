# Easy Bot v0 pilot protocol

This is an integration pilot, not a mastery run. Its purpose is to answer one
question quickly: can a single sequential LLC checkpoint pass through movement,
weapon, spacing, and combat training without a broken data, checkpoint, or
retention path?

Recovery and all-skills consolidation are deliberately deferred. No phase is
promoted as "mastered" from this 4,096-step pilot.

## Frozen contract

- Interpreter: `C:\venvs\brawl312\Scripts\python.exe`
- Output directory: `train/models/easy_bot_v0`
- Phase order: movement -> weapon -> spacing -> combat -> recovery -> consolidation
- Perception: YOLO every step; weapon classifier integration remains off
- PPO: 4,096 steps per active phase, `n_steps=512`, `batch_size=256`
- Evaluation: two sequential skill episodes after each PPO phase
- Source data: only the four archives named in `manifest.json`

The existing weapon, spacing, and combat BC initializers are independent
specialists. They are useful baselines, but they must not initialize the
sequential LLC. Each later BC pass resumes the previous phase's final PPO
checkpoint.

## Gate 0 - record the live setup and checkpoint the source

Before the first live PPO step:

1. Replace the three `RECORD_...` / `CURRENT_...` values in `config.json` with
   the exact player legend, Easy-bot legend, and calibrated map.
2. Make a deliberate source-control checkpoint. `manifest.json` records that
   its current commit is dirty, so the commit hash alone cannot reproduce this
   code state.
3. Put Brawlhalla in Training Mode at 1920x1080, on the calibrated map, against
   the same Easy bot for every phase and evaluation.

Do not change opponent, legend, map, resolution, detector, or cadence within
this experiment.

## Gate 1 - verify the frozen inputs

```powershell
C:\venvs\brawl312\Scripts\python.exe -m tools.verify_experiment_manifest
C:\venvs\brawl312\Scripts\python.exe -m tools.llc_preflight --device cuda --strict-warnings
C:\venvs\brawl312\Scripts\python.exe -m tools.validate_llc_demos `
  train\models\recollected\movement_fluency_demos.npz `
  train\models\recollected\weapon_acquisition_demos.npz `
  train\models\recollected\spacing_neutral_demos_v2.npz `
  train\models\recollected\combat_execution_demos.npz `
  --strict-warnings --min-samples 1
```

Stop if any command fails. The manifest is fail-closed: a changed archive,
checkpoint, or detector is a different experiment.

## Gate 2 - movement smoke PPO

Use the already-verified movement BC initializer. This is the only independent
BC initializer allowed into the sequential chain.

```powershell
C:\venvs\brawl312\Scripts\python.exe -m train.train_curriculum `
  --phase movement_fluency `
  --resume train\models\llc_movement_fluency_bc_init.zip `
  --timesteps 4096 `
  --n-steps 512 `
  --batch-size 256 `
  --max-episode-steps 200 `
  --bc-demos-path "train\models\recollected\movement_fluency_demos.npz" `
  --save-dir train\models\easy_bot_v0 `
  --model-name llc_movement_fluency `
  --seed 42 `
  --device cuda `
  --log-csv `
  --plot-every 0 `
  --diag-report-every 512 `
  --eval-every-steps 0
```

Then run the two-episode skill check:

```powershell
C:\venvs\brawl312\Scripts\python.exe -m train.evaluate_retention `
  --model train\models\easy_bot_v0\llc_movement_fluency.zip `
  --phase movement_fluency `
  --phases movement_fluency `
  --episodes 2 `
  --max-episode-steps 200 `
  --best-scores train\models\easy_bot_v0\retention_best.json `
  --csv train\models\easy_bot_v0\llc_movement_fluency_retention_eval.csv `
  --seed 42 `
  --device cuda
```

Continue only if the checkpoint and both CSV logs exist, training/evaluation
contain no NaNs or exceptions, and the trainer reports one enabled BC dataset.
Low skill score alone does not fail this short integration smoke.

## Gate 3 - weapon handoff

BC the weapon demos into the movement PPO checkpoint, then PPO with both demo
archives anchoring the same policy.

```powershell
C:\venvs\brawl312\Scripts\python.exe -m train.pretrain_bc_locomotion `
  --phase weapon_acquisition `
  --demos train\models\recollected\weapon_acquisition_demos.npz `
  --resume train\models\easy_bot_v0\llc_movement_fluency.zip `
  --epochs 8 `
  --batch-size 512 `
  --learning-rate 0.0002 `
  --entropy-coef 0.001 `
  --goal-relabel `
  --output train\models\easy_bot_v0\llc_weapon_acquisition_bc_init.zip `
  --device cuda

C:\venvs\brawl312\Scripts\python.exe -m train.train_curriculum `
  --phase weapon_acquisition `
  --resume train\models\easy_bot_v0\llc_weapon_acquisition_bc_init.zip `
  --timesteps 4096 `
  --n-steps 512 `
  --batch-size 256 `
  --max-episode-steps 200 `
  --bc-demos-path "train\models\recollected\movement_fluency_demos.npz;train\models\recollected\weapon_acquisition_demos.npz" `
  --save-dir train\models\easy_bot_v0 `
  --model-name llc_weapon_acquisition `
  --seed 42 `
  --device cuda `
  --log-csv `
  --plot-every 0 `
  --diag-report-every 512 `
  --eval-every-steps 0

C:\venvs\brawl312\Scripts\python.exe -m train.evaluate_retention `
  --model train\models\easy_bot_v0\llc_weapon_acquisition.zip `
  --phase weapon_acquisition `
  --phases movement_fluency,weapon_acquisition `
  --episodes 2 `
  --max-episode-steps 200 `
  --best-scores train\models\easy_bot_v0\retention_best.json `
  --csv train\models\easy_bot_v0\llc_weapon_acquisition_retention_eval.csv `
  --seed 42 `
  --device cuda
```

Apply the same integration gate as movement. Evaluation is sequential and
explicit so policy updates never run while another live evaluation environment
is interacting with the same game.

## Gate 4 - spacing handoff

```powershell
C:\venvs\brawl312\Scripts\python.exe -m train.pretrain_bc_locomotion `
  --phase spacing_neutral `
  --demos train\models\recollected\spacing_neutral_demos_v2.npz `
  --resume train\models\easy_bot_v0\llc_weapon_acquisition.zip `
  --epochs 8 `
  --batch-size 512 `
  --learning-rate 0.0002 `
  --entropy-coef 0.001 `
  --goal-relabel `
  --output train\models\easy_bot_v0\llc_spacing_neutral_bc_init.zip `
  --device cuda

C:\venvs\brawl312\Scripts\python.exe -m train.train_curriculum `
  --phase spacing_neutral `
  --resume train\models\easy_bot_v0\llc_spacing_neutral_bc_init.zip `
  --timesteps 4096 `
  --n-steps 512 `
  --batch-size 256 `
  --max-episode-steps 200 `
  --bc-demos-path "train\models\recollected\movement_fluency_demos.npz;train\models\recollected\weapon_acquisition_demos.npz;train\models\recollected\spacing_neutral_demos_v2.npz" `
  --save-dir train\models\easy_bot_v0 `
  --model-name llc_spacing_neutral `
  --seed 42 `
  --device cuda `
  --log-csv `
  --plot-every 0 `
  --diag-report-every 512 `
  --eval-every-steps 0

C:\venvs\brawl312\Scripts\python.exe -m train.evaluate_retention `
  --model train\models\easy_bot_v0\llc_spacing_neutral.zip `
  --phase spacing_neutral `
  --phases movement_fluency,weapon_acquisition,spacing_neutral `
  --episodes 2 `
  --max-episode-steps 200 `
  --best-scores train\models\easy_bot_v0\retention_best.json `
  --csv train\models\easy_bot_v0\llc_spacing_neutral_retention_eval.csv `
  --seed 42 `
  --device cuda
```

## Gate 5 - combat handoff

```powershell
C:\venvs\brawl312\Scripts\python.exe -m train.pretrain_bc_locomotion `
  --phase combat_execution `
  --demos train\models\recollected\combat_execution_demos.npz `
  --resume train\models\easy_bot_v0\llc_spacing_neutral.zip `
  --epochs 8 `
  --batch-size 512 `
  --learning-rate 0.0002 `
  --entropy-coef 0.001 `
  --goal-relabel `
  --output train\models\easy_bot_v0\llc_combat_execution_bc_init.zip `
  --device cuda

C:\venvs\brawl312\Scripts\python.exe -m train.train_curriculum `
  --phase combat_execution `
  --resume train\models\easy_bot_v0\llc_combat_execution_bc_init.zip `
  --timesteps 4096 `
  --n-steps 512 `
  --batch-size 256 `
  --max-episode-steps 200 `
  --bc-demos-path "train\models\recollected\movement_fluency_demos.npz;train\models\recollected\weapon_acquisition_demos.npz;train\models\recollected\spacing_neutral_demos_v2.npz;train\models\recollected\combat_execution_demos.npz" `
  --save-dir train\models\easy_bot_v0 `
  --model-name llc_combat_execution `
  --seed 42 `
  --device cuda `
  --log-csv `
  --plot-every 0 `
  --diag-report-every 512 `
  --eval-every-steps 0

C:\venvs\brawl312\Scripts\python.exe -m train.evaluate_retention `
  --model train\models\easy_bot_v0\llc_combat_execution.zip `
  --phase combat_execution `
  --phases movement_fluency,weapon_acquisition,spacing_neutral,combat_execution `
  --episodes 2 `
  --max-episode-steps 200 `
  --best-scores train\models\easy_bot_v0\retention_best.json `
  --csv train\models\easy_bot_v0\llc_combat_execution_retention_eval.csv `
  --seed 42 `
  --device cuda
```

## Gate 6 - first Easy-bot match

Blocked for now. `train/run_match.py` currently switches model files while the
environment remains conditioned on the combat goal. Its weapon and spacing
decisions therefore do not test the intended skill policies, and missing recovery
can be silently labelled as recovery while actually running combat.

Before any win-rate claim, the match harness must either:

1. use one final joint checkpoint and explicitly switch the target and mask passed
   to that checkpoint, or
2. switch both the specialist checkpoint and its corresponding goal target/mask,
   while reporting unavailable skills honestly.

After that contract is tested, run 3 one-stock smoke matches, then 10 diagnostic
matches. Only a stable harness may run the 30-match promotion gate in
`config.json`. Recovery is revisited only after this first gameplay evidence shows
it is the limiting failure mode.
