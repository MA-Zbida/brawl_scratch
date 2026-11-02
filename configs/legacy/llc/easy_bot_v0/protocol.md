# Easy Bot v0 pilot protocol

This is an integration pilot, not a mastery run. Its purpose is to answer one
question quickly: can a single sequential LLC checkpoint pass through movement,
weapon, spacing, and combat training without a broken data, checkpoint, or
retention path?

Recovery PPO and all-skills consolidation are deliberately deferred. Recovery
still contributes its existing validated archive through one BC-only handoff;
there is no recovery collection stage because its teacher only becomes useful
after the setup has already put the fighter offstage.

## Canonical runner

The phase commands below remain the expanded protocol, but the supported entry
point is now the resumable driver:

```powershell
# Prove that the whole control flow closes with tiny budgets first.
scripts\llc\easy_bot_v0\run_chain.ps1 --smoke

# Continue the production chain after the preserved weapon PPO checkpoint.
scripts\llc\easy_bot_v0\run_chain.ps1 --from spacing

# Run spacing only, then stop after its retention evaluation.
scripts\llc\easy_bot_v0\run_chain.ps1 --from spacing_bc --to spacing --timesteps 200000

# After changing the opponent setup, run combat only.
scripts\llc\easy_bot_v0\run_chain.ps1 --from combat --to combat --timesteps 200000
```

The driver runs preflight first, records stdout and stderr in one UTF-8 log, and
writes a completion marker only after the stage exits zero and its artifacts
exist. Recovery is always `recovery_bc`; it is never collected or PPO-trained.

## Frozen contract

- Interpreter: `C:\venvs\brawl312\Scripts\python.exe`
- Output directory: `runs/legacy/llc/easy_bot_v0`
- Chain order: movement base -> recovery BC -> weapon -> spacing -> combat
- Compute split: MLP policy and optimizer on CPU; TensorRT detection remains on CUDA
- Perception: YOLO every step; weapon classifier disabled. The existing classifier
  reports mean P(armed)=0.873 on unarmed live crops because its export-time crop
  domain differs from live capture, so possession falls back to action inference.
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
C:\venvs\brawl312\Scripts\python.exe -m scripts.llc.verify_experiment_manifest
C:\venvs\brawl312\Scripts\python.exe -m scripts.llc.llc_preflight --device cuda --strict-warnings
C:\venvs\brawl312\Scripts\python.exe -m scripts.llc.validate_llc_demos `
  runs\legacy\llc\recollected\movement_fluency_demos.npz `
  runs\legacy\llc\recollected\weapon_acquisition_demos.npz `
  runs\legacy\llc\recollected\spacing_neutral_demos_v2.npz `
  runs\legacy\llc\recollected\combat_execution_demos.npz `
  --strict-warnings --min-samples 1
```

Stop if any command fails. The manifest is fail-closed: a changed archive,
checkpoint, or detector is a different experiment.

## Gate 2 - movement smoke PPO

Use the already-verified movement BC initializer. This is the only independent
BC initializer allowed into the sequential chain.

```powershell
C:\venvs\brawl312\Scripts\python.exe -m brawl.pipelines.llc.train_curriculum `
  --phase movement_fluency `
  --resume runs\legacy\llc\llc_movement_fluency_bc_init.zip `
  --timesteps 4096 `
  --n-steps 512 `
  --batch-size 256 `
  --max-episode-steps 200 `
  --bc-demos-path "runs\legacy\llc\recollected\movement_fluency_demos.npz" `
  --save-dir runs\legacy\llc\easy_bot_v0 `
  --model-name llc_movement_fluency `
  --seed 42 `
  --device cpu `
  --log-csv `
  --plot-every 0 `
  --diag-report-every 512 `
  --eval-every-steps 0
```

Then run the two-episode skill check:

```powershell
C:\venvs\brawl312\Scripts\python.exe -m brawl.pipelines.llc.evaluate_retention `
  --model runs\legacy\llc\easy_bot_v0\llc_movement_fluency.zip `
  --phase movement_fluency `
  --phases movement_fluency `
  --episodes 2 `
  --max-episode-steps 200 `
  --best-scores runs\legacy\llc\easy_bot_v0\retention_best.json `
  --csv runs\legacy\llc\easy_bot_v0\llc_movement_fluency_retention_eval.csv `
  --seed 42 `
  --device cpu
```

Continue only if the checkpoint and both CSV logs exist, training/evaluation
contain no NaNs or exceptions, and the trainer reports one enabled BC dataset.
Low skill score alone does not fail this short integration smoke.

## Gate 3 - weapon handoff

BC the weapon demos into the movement PPO checkpoint, then PPO with both demo
archives anchoring the same policy.

```powershell
C:\venvs\brawl312\Scripts\python.exe -m brawl.pipelines.llc.pretrain_bc `
  --phase weapon_acquisition `
  --demos runs\legacy\llc\recollected\weapon_acquisition_demos.npz `
  --resume runs\legacy\llc\easy_bot_v0\llc_movement_fluency.zip `
  --epochs 8 `
  --batch-size 512 `
  --learning-rate 0.0002 `
  --entropy-coef 0.001 `
  --goal-relabel `
  --output runs\legacy\llc\easy_bot_v0\llc_weapon_acquisition_bc_init.zip `
  --device cpu

C:\venvs\brawl312\Scripts\python.exe -m brawl.pipelines.llc.train_curriculum `
  --phase weapon_acquisition `
  --resume runs\legacy\llc\easy_bot_v0\llc_weapon_acquisition_bc_init.zip `
  --timesteps 4096 `
  --n-steps 512 `
  --batch-size 256 `
  --max-episode-steps 200 `
  --bc-demos-path "runs\legacy\llc\recollected\movement_fluency_demos.npz;runs\legacy\llc\recollected\weapon_acquisition_demos.npz" `
  --save-dir runs\legacy\llc\easy_bot_v0 `
  --model-name llc_weapon_acquisition `
  --seed 42 `
  --device cpu `
  --log-csv `
  --plot-every 0 `
  --diag-report-every 512 `
  --eval-every-steps 0

C:\venvs\brawl312\Scripts\python.exe -m brawl.pipelines.llc.evaluate_retention `
  --model runs\legacy\llc\easy_bot_v0\llc_weapon_acquisition.zip `
  --phase weapon_acquisition `
  --phases movement_fluency,weapon_acquisition `
  --episodes 2 `
  --max-episode-steps 200 `
  --best-scores runs\legacy\llc\easy_bot_v0\retention_best.json `
  --csv runs\legacy\llc\easy_bot_v0\llc_weapon_acquisition_retention_eval.csv `
  --seed 42 `
  --device cpu
```

Apply the same integration gate as movement. Evaluation is sequential and
explicit so policy updates never run while another live evaluation environment
is interacting with the same game.

## Gate 4 - spacing handoff

```powershell
C:\venvs\brawl312\Scripts\python.exe -m brawl.pipelines.llc.pretrain_bc `
  --phase spacing_neutral `
  --demos runs\legacy\llc\recollected\spacing_neutral_demos_v2.npz `
  --resume runs\legacy\llc\easy_bot_v0\llc_weapon_acquisition.zip `
  --epochs 8 `
  --batch-size 512 `
  --learning-rate 0.0002 `
  --entropy-coef 0.001 `
  --goal-relabel `
  --output runs\legacy\llc\easy_bot_v0\llc_spacing_neutral_bc_init.zip `
  --device cpu

C:\venvs\brawl312\Scripts\python.exe -m brawl.pipelines.llc.train_curriculum `
  --phase spacing_neutral `
  --resume runs\legacy\llc\easy_bot_v0\llc_spacing_neutral_bc_init.zip `
  --timesteps 4096 `
  --n-steps 512 `
  --batch-size 256 `
  --max-episode-steps 200 `
  --bc-demos-path "runs\legacy\llc\recollected\movement_fluency_demos.npz;runs\legacy\llc\recollected\weapon_acquisition_demos.npz;runs\legacy\llc\recollected\spacing_neutral_demos_v2.npz" `
  --save-dir runs\legacy\llc\easy_bot_v0 `
  --model-name llc_spacing_neutral `
  --seed 42 `
  --device cpu `
  --log-csv `
  --plot-every 0 `
  --diag-report-every 512 `
  --eval-every-steps 0

C:\venvs\brawl312\Scripts\python.exe -m brawl.pipelines.llc.evaluate_retention `
  --model runs\legacy\llc\easy_bot_v0\llc_spacing_neutral.zip `
  --phase spacing_neutral `
  --phases movement_fluency,weapon_acquisition,spacing_neutral `
  --episodes 2 `
  --max-episode-steps 200 `
  --best-scores runs\legacy\llc\easy_bot_v0\retention_best.json `
  --csv runs\legacy\llc\easy_bot_v0\llc_spacing_neutral_retention_eval.csv `
  --seed 42 `
  --device cpu
```

## Gate 5 - combat handoff

```powershell
C:\venvs\brawl312\Scripts\python.exe -m brawl.pipelines.llc.pretrain_bc `
  --phase combat_execution `
  --demos runs\legacy\llc\recollected\combat_execution_demos.npz `
  --resume runs\legacy\llc\easy_bot_v0\llc_spacing_neutral.zip `
  --epochs 8 `
  --batch-size 512 `
  --learning-rate 0.0002 `
  --entropy-coef 0.001 `
  --goal-relabel `
  --output runs\legacy\llc\easy_bot_v0\llc_combat_execution_bc_init.zip `
  --device cpu

C:\venvs\brawl312\Scripts\python.exe -m brawl.pipelines.llc.train_curriculum `
  --phase combat_execution `
  --resume runs\legacy\llc\easy_bot_v0\llc_combat_execution_bc_init.zip `
  --timesteps 4096 `
  --n-steps 512 `
  --batch-size 256 `
  --max-episode-steps 200 `
  --bc-demos-path "runs\legacy\llc\recollected\movement_fluency_demos.npz;runs\legacy\llc\recollected\weapon_acquisition_demos.npz;runs\legacy\llc\recollected\spacing_neutral_demos_v2.npz;runs\legacy\llc\recollected\combat_execution_demos.npz" `
  --save-dir runs\legacy\llc\easy_bot_v0 `
  --model-name llc_combat_execution `
  --seed 42 `
  --device cpu `
  --log-csv `
  --plot-every 0 `
  --diag-report-every 512 `
  --eval-every-steps 0

C:\venvs\brawl312\Scripts\python.exe -m brawl.pipelines.llc.evaluate_retention `
  --model runs\legacy\llc\easy_bot_v0\llc_combat_execution.zip `
  --phase combat_execution `
  --phases movement_fluency,weapon_acquisition,spacing_neutral,combat_execution `
  --episodes 2 `
  --max-episode-steps 200 `
  --best-scores runs\legacy\llc\easy_bot_v0\retention_best.json `
  --csv runs\legacy\llc\easy_bot_v0\llc_combat_execution_retention_eval.csv `
  --seed 42 `
  --device cpu
```

## Gate 6 - first Easy-bot match

Blocked for now. `brawl/pipelines/final_agent/run_match.py` currently switches model files while the
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

## Attempt ledger

- Movement attempt 1 stopped before its first PPO update at the 512-step
  diagnostic boundary. A legacy multidiscrete diagnostic indexed a scalar
  `Discrete(27)` action. No model checkpoint was produced; the two CSVs are
  preserved with `failed_step512` in their names. The callback, categorical
  entropy counter, and dormant reset perturbation were migrated together, and
  the rerun starts again from the frozen movement BC initializer on CPU.

- Movement attempt 2 reached 1,024 environment steps after one complete
  512-step rollout optimization cycle. The next optimization cycle stopped
  before its first optimizer minibatch when anchored replay tried to concatenate
  live `Discrete(27)` actions shaped `[N, 1]` with replay actions shaped `[N]`.
  No model checkpoint was produced; the 1,024-row step CSV and 8-row episode CSV
  are preserved with `failed_replay_step1024` in their names. The rerun starts
  again from the frozen movement BC initializer on CPU after the replay
  action-shape contract is covered by a two-update regression test.

- Movement attempt 3 completed all 4,096 steps on CPU and produced a loadable
  checkpoint plus 4,096 step rows and 55 completed-episode rows. The two-episode
  deterministic evaluation then scored `0.060` with `0/2` successes and both
  episodes reaching the 200-step cap. This passes the integration smoke but
  fails the movement promotion threshold of `0.650`; no later phase may resume
  from this checkpoint until the diagnostic below identifies the failure mode.

## Movement BC/PPO diagnostic gate

The first completed PPO run used replay rows inside the core PPO/value loss,
despite the algorithm contract specifying auxiliary-only replay. That behavior
is now blocked by regression tests. Reset-time targets already inside the goal
radius are also resampled; an exhausted target remains inactive instead of
creating a fake one-step success.

Run the five-target grounded/airborne route once for each checkpoint and policy
mode. The route is target-paired, but Training Mode cannot restore an exact
physical starting frame, so each row records start error and excludes trivial
starts from its score.

```powershell
C:\venvs\brawl312\Scripts\python.exe -m brawl.pipelines.llc.compare_checkpoints `
  --bc-model runs\legacy\llc\llc_movement_fluency_bc_init.zip `
  --ppo-model runs\legacy\llc\easy_bot_v0\llc_movement_fluency.zip `
  --csv runs\legacy\llc\easy_bot_v0\movement_bc_ppo_diagnostic.csv `
  --repeats 1 `
  --max-episode-steps 200 `
  --seed 42 `
  --device cpu
```

Decision rules:

- PPO deterministic passes `0.650`: movement may advance after a clean rerun
  with the corrected replay/reset contracts.
- BC deterministic beats PPO deterministic materially: PPO damaged a useful BC
  policy; discard the PPO checkpoint and rerun from the frozen BC initializer.
- Stochastic materially beats deterministic for the same checkpoint: the action
  distribution contains useful behavior but its argmax policy is collapsed;
  inspect the exact 27-action traces before choosing inference mode.
- BC and PPO both fail, especially on airborne targets: improve the movement
  demonstrations/target curriculum before spending a longer PPO budget.
