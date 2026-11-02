# Round A deployment benchmark (declared before the 2M runs)

**Current next gate:** [uncapped logging + BC/PPO diagnosis](round-a-profiling.md).
The 30 Hz protocol below is historical. The new uncapped protocol is versioned
separately; do not change or pool old results. The working `deployment_v1.json`
was reduced to 100k for screening, so the historical 2M schedule below is not a
description of that current file.

Question: which complete policy system gives the best stock-exchange performance
per environment interaction and per wall-clock hour on this laptop?

This is **not** a fixed-cadence architecture ablation. Same interaction counts
do not imply the same gameplay exposure when decisions run at different speeds.
Multi-input also receives additional visual information, not just more capacity.

## Frozen conditions

- Wu Shang vs ATTACKING Medium Mordex, Small Brawlhaven, continuous-respawn
  training room. These are stock exchanges, **not match wins**. Configure manually.
- Same aligned dataset: `train/models/round_a/collection_pilot_v1/demos`.
- Existing architecture-specific 40-epoch BC initializers, seed 42; no further
  epoch selection after seeing the production outcomes. Their filenames say
  pilot; their fixed dataset and checkpoint hashes are recorded at initialization.
- `experiments/round_a/deployment_v1.json`: 2,000,000 requested PPO interactions,
  rounded identically to 2,000,384 (complete 512-step rollouts).
- MLP and recurrent on CPU; multi-input on CUDA. Same learning hyperparameters.
- 30 Hz maximum target. Late decisions are logged, not discarded or fatal.
- Initial deterministic evaluation: 15 minutes. Interim: 5 minutes every
  250,000 steps (at the next completed rollout). Final: 1 hour, frozen policy.
- Training uses stochastic PPO actions; evaluation uses deterministic actions.
- Structured world replay is enabled in the commands below. No videos by default.

All arms still use raw DXCam frames for YOLO, HUD, identity and weapon sensing.
Only the CNN prepares its extra RGB image history during deployment. Collection
always prepares images for every arm's shared BC dataset. Detector resolution,
classifier cadence, policy inputs, action semantics and rewards are unchanged.

## Start one candidate

Begin with moving NON-ATTACKING Medium Mordex for qualification, approve only
visibly correct behavior, then switch to ATTACKING Medium Mordex when prompted.
Qualification still blocks on skill failure or missing/invalid timing. A cadence
warning is deliberately allowed; there is no other bypass.

```powershell
$Candidate = 'mlp'  # Next runs: 'recurrent', then 'multi'
$PolicyDevice = if ($Candidate -eq 'multi') { 'cuda' } else { 'cpu' }
C:\venvs\brawl312\Scripts\python.exe -m brawl.pipelines.round_a.run `
  --stage train --candidate $Candidate `
  --protocol experiments\round_a\deployment_v1.json `
  --init-bc "train\models\round_a\pilot_bc_${Candidate}_40ep_s42\bc_init.zip" `
  --demos train\models\round_a\collection_pilot_v1\demos `
  --run-dir "train\models\round_a\deployment_v1_${Candidate}_s42" `
  --device $PolicyDevice `
  --world-replay outputs\world_replay\round_a_deployment_v1
```

Run directories must be new. `--init-bc` permits only zero-step BC artifacts;
it records source/destination contracts in `initialization.json` without editing
the source checkpoint. Only the experiment schedule and execution mode may
change. Optimizer settings, observation resolution, dataset, seed and BC settings
remain checked. `--resume` continues PPO under its exact protocol and remaining
budget, never relabeling trained weights as fresh BC.

After interruption, use `--resume <run>/latest.zip` or a completed-rollout
`step_*.zip`, the same protocol/candidate/seed/device/dataset, and a NEW output
directory. Retain both segments and account for any repeated evaluation windows.
An `interrupted_or_failed.zip` may be mid-rollout and is not always resumable.

At 20-30 actual training decisions/s, 2M interactions alone take roughly
18.5-27.8 hours **per candidate**, before optimizer updates and evaluations.
Plug in the laptop and disable sleep; do not interpret a few hours as completion.

## Evidence retention and reporting

Keep manifests, hashes, initialization, qualifications, every checkpoint,
optimizer CSVs, evaluations, retention logs, training summaries and replay.
Do not delete failed runs or pick the best interim score for the main table.
Use the final-budget checkpoint; baseline/interim plots are supporting evidence.
The shared report now writes `run_inventory.json` including stopped runs with no
evaluation points, and labels curves by execution mode. List every declared run
when generating the comparison; it does not discover omitted directories for you.

After the runs, produce a new comparison directory (also include any failed or
resumed segments in the explicit list):

```powershell
C:\venvs\brawl312\Scripts\python.exe -m brawl.analysis.plot_round_a `
  train\models\round_a\deployment_v1_mlp_s42 `
  train\models\round_a\deployment_v1_recurrent_s42 `
  train\models\round_a\deployment_v1_multi_s42 `
  --output outputs\round_a_deployment_v1_report
```

Primary metric: final net stocks/minute. Also report damage dealt/received,
damage trade/minute, live recovery and pickup opportunity counts, censor/identity
uncertainty counts, imitation retention, actual cadence, gameplay interaction
time, optimizer time and total wall time. Imitation retention is not live mastery.
Read gameplay videos manually when available, but do not select favorable clips
as a substitute for the hour-long quantitative evaluation.

Do not pool the earlier fixed-cadence pilots with deployment results. One seed
is screening evidence; repeat promising comparisons with paired training seeds
before claiming robust superiority. Neither contiguous evaluation blocks nor
millions of transitions are independent training replicates.

## Bounded optimization decision

One candidate change: omit unused pixel preparation for structured deployment.
Three alternating-order offline 1080p tests measured 4.37-4.45 ms median for shared
image preparation versus about 0.003 ms for structured-only, with identical
structured observations (`work/deployment_image_benchmark.json`). This is not a
claim that live 30 Hz now passes. No TensorRT rebuild or asynchronous sensing was
introduced. The legacy `fixed_cadence` protocol retains shared pixel preparation.
The standalone confirmation tool measured 4.73-5.23 ms versus 0.003 ms over three
more repeats (`work/deployment_overhead_confirm.json`); pixel/structured equality
checks passed. Stop optimization here and collect the declared deployment results.

Reproduce with the offline tool (new output filename):

```powershell
C:\venvs\brawl312\Scripts\python.exe -m scripts.round_a.benchmark_round_a_overhead `
  --output work\deployment_overhead_repeat.json
```
