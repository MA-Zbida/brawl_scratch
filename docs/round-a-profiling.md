# Uncapped logging and policy diagnosis

Goal: distinguish logging overhead, detector latency and policy improvement before
another long PPO run. This is a **frozen-policy deployment screening**, not training,
not a fixed-cadence architecture ablation, and not a claim that PPO must win.

## Run

Configure **Wu Shang vs ATTACKING Medium Mordex, Small Brawlhaven, continuous respawns**.
Keep resolution, bot behavior, game FPS setting, laptop power mode and OneDrive state
unchanged for the entire run. Connect power. Do not run another collector/trainer.
The prompt gives five seconds to focus the game. No menu keys are automated.

```powershell
C:\venvs\brawl312\Scripts\python.exe -m tools.profile_round_a `
  --bc train\models\round_a\pilot_bc_mlp_40ep_s42\bc_init.zip `
  --ppo train\models\round_a\deployment_v1_mlp_s42_run02\step_000050176.zip `
  --output outputs\round_a_profile_uncapped_v1 `
  --device cpu --block-seconds 300 --rounds 2
```

About **40 minutes plus warm-up/loading/saving**. Output must be a NEW directory.
For a plumbing check only, use a different directory and `--block-seconds 10 --rounds 1`.
That short check cannot establish gameplay improvement or replay shard overhead.
Ctrl+C releases controls and keeps completed blocks plus the partial timing/result.
A process kill/power loss can lose the current in-memory block; earlier blocks remain.

The four conditions are BC/normal, PPO/minimal, PPO/normal, BC/minimal; the second
round reverses their order. Both policies get equal scored wall time within both
logging modes. This reduces order bias; it cannot reset the physical starting state
or produce independent training seeds. Recurrent state restarts at block boundaries
and reporting truncations. Warm-up play is excluded from scores.

## What is measured

- **Normal:** the existing buffered transition/window CSV writer and structured
  world-replay writer (4096-row compressed shards). The replay is tagged evaluation;
  do not train the compared policies on it.
- **Minimal:** no CSV/replay persistence during play. Both modes accumulate equal
  timing instrumentation and outcome accounting in RAM, then write evidence at the
  block boundary. This is not a completely uninstrumented baseline.
- Policy inference, full decisions, adapter/base steps, capture, input execution,
  detector preprocessing/inference/postprocessing/output transfer, state estimation,
  weapon classification/fusion, HUD/game logic, observation assembly, reward,
  CSV write/flush, replay append/shard flush/manifest, and evaluation accounting.
- Calls, totals, mean, p50/p95/p99/max, exclusive host time, requested/effective
  action histograms, decision cadence, damage dealt/received, stocks, recovery and
  pickup opportunities, and identity-uncertainty/censor counters.
- Tail flushes/closing are separately labelled outside scored playing time. Periodic
  shard writes remain inside measured decisions. Saving the profiler trace is outside
  scored play. There is **no video** to confound the logging comparison.

`summary.json` keeps all blocks and aggregates only completed blocks, separately by
policy AND logging condition. It reports logging throughput differences and PPO-minus-BC
outcomes without automatically declaring a winner. Each block also has `result.json`,
`timing.json`, and `timing_trace.npz` with stage ID, decision index, start, duration and
exclusive time. `timing.json` maps stage names to IDs. Missing probes are explicit,
not reported as zero-cost stages. Up to one million events per block are retained;
overflow is explicit, with all-call totals/max preserved and prefix-only quantiles.
`in_loop_replay_shards` distinguishes real periodic flushes from the closing tail;
if it is zero in normal blocks, those blocks did not test periodic shard stalls.

## Interpretation limits

These are **host wall times**, including waits/scheduling. Nested inclusive times
must not be added. Exclusive time subtracts instrumented children, not every possible
operation. No extra CUDA synchronization is inserted: sub-stage GPU work can be
charged at a later synchronization point. Capture duration is not frame age; this
does not prove every DXCam result is a new rendered frame. Physical disk writes,
OneDrive background I/O, GPU kernel time and thermal throttling are not directly
measured. A larger normal/minimal gap motivates a focused follow-up, not an automatic
attribution to one of those causes. Instrumentation itself has overhead in both modes.

Only these checkpoints are compared. Positive stock/damage deltas are screening
evidence, not proof across training seeds. Compare within logging conditions; do not
pool a faster minimal-logging PPO block against a slower normal-logging BC block.
Training-room stock exchanges are not match wins. Inspect sensing uncertainty before
interpreting differences. No sensor, reward, action, or checkpoint weights are changed.

## Cadence and future training

This tool overrides live pacing to **uncapped**, and records the original checkpoint
cadence separately. Histories remain step-indexed, so their temporal span changes
with measured Hz; this is a deployment distribution shift, not a free inference-only
optimization. A faster policy is allowed to make more decisions per second.

The original `deployment_v1.json` and all checkpoint contracts remain untouched.
`deployment_uncapped_v2.json` is a NEW 100k-step screening protocol with no cap and
evaluations every 25k steps (300-second baseline/interims, 600-second final).
Only proven zero-step BC may enter that new training protocol using `--init-bc`;
resuming old PPO with altered cadence is still rejected. The dataset's original
capture cadence is retained in its manifest. Do not present the 100k screening as
the later 2M benchmark. Freeze the long-run budget only after this diagnosis.
