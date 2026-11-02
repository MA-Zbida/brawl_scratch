# Round A: information and memory

**Current production route:** [Deployment benchmark](deployment.md).
It uses the existing BC initializers, a 2M-step budget, reported rather than
blocking cadence, and candidate-specific preprocessing. The fixed-cadence
workflow below is retained as the original protocol, not a prerequisite to run
the deployment comparison. Do not combine their results as one condition.

This is an executable benchmark scaffold, not a claim of measured superiority.
No live game has been run by the implementation agent. The three candidates are:

| CLI candidate | Policy | Inputs |
| --- | --- | --- |
| `mlp` | CPU PPO, 128x128 actor/critic | Existing structured146 + fixed weapon geometry |
| `recurrent` | RecurrentPPO, separate 128-unit actor/critic LSTMs | Same structured observation and history |
| `multi` | PPO, shared compact CNN per temporal slot + 128x128 actor/critic | Same structured input + RGB frames at offsets 0,2,4,8 |

Multi-input is **additional information plus an encoder**, not a pure network-size
ablation. Parameter counts are reported. No FiLM, goal conditioning, PCGrad,
anchored replay or goal selector is added in this round. Existing primitive27 and
semantic57 entry points/checkpoints remain unchanged.

## Before committing to production

`configs/legacy/round_a/protocol_v1.json` is a **proposed** 256x144, 30 Hz protocol.
Resolution and cadence are NOT proven optimal. Inspect the preview and measure
live qualification on all three candidates before freezing production. CPU is
the default; `--device cuda` is explicit and recorded, principally for the CNN.
Keep the device assignment fixed for repeated runs of the same candidate.

The installed brawl312 environment now includes sb3-contrib 2.9.0. On another
machine, after installing the correct CUDA torch independently:

```powershell
C:\venvs\brawl312\Scripts\python.exe -m pip install -r requirements.txt
```

Never silently reuse structured-only NPZ demos for a visual model. Video that
cannot be aligned exactly to observations and requested/executed inputs is not a
substitute. The aligned dataset below is shared by all candidates.

## 1. Collect a small feasibility dataset once

Configure **Wu Shang vs moving NON-ATTACKING Medium Mordex, Small Brawlhaven,
continuous respawns**. Settings are manual; the program cannot verify the menu.

```powershell
C:\venvs\brawl312\Scripts\python.exe -m brawl.pipelines.round_a.run `
  --stage collect --episodes-per-drill 5 `
  --run-dir runs\legacy\round_a\collection_pilot_v1
```

The existing primitive teacher drives weapon, recovery and combat drills. As
before, recovery setup walks offstage and is NOT an imitation label. Weapon
pickup/hold and recovery acceptance rules are preserved. Combat needs confirmed
attack presses and measured damage. A failed quota saves partial evidence and
stops; partial datasets cannot enter pretraining.

Each accepted episode stores all intervening observations, frames, action
requests and a valid-label mask. Recurrent training sees unlabeled setup/blocked
context, but loss is computed only on accepted labels. No fictional adjacency is
created by dropping rows. Frames are stored once as RGB, with explicit history
references and the current canonical mirror flag. NPY files are memory-mapped;
at 256x144 each stored frame is about 108 KiB. Reserve several GB for a full
dataset, depending on setup/episode lengths. All arms use the same resized frame
preparation in live play, even when the policy does not consume pixels.

```powershell
C:\venvs\brawl312\Scripts\python.exe -m brawl.pipelines.round_a.run `
  --stage preflight `
  --demos runs\legacy\round_a\collection_pilot_v1\demos `
  --run-dir runs\legacy\round_a\preflight_v1
```

Inspect `resolution_preview.png` and `preflight.json`. These are offline model
latencies, **excluding capture/detection**. Live qualification additionally checks
that at least 95% of measured decision intervals are within 110% of the proposed
period. This is a declared feasibility tolerance, not a performance score. A
slower candidate must use a feasible COMMON cadence with fresh comparable data;
do not quietly lower only its control rate. Do not change the protocol after
seeing scored outcomes.

## 2. Collect the common production dataset

After choosing the common resolution/cadence, use a fresh directory:

```powershell
C:\venvs\brawl312\Scripts\python.exe -m brawl.pipelines.round_a.run `
  --stage collect --episodes-per-drill 30 `
  --run-dir runs\legacy\round_a\collection_v1
```

Add `--world-replay data\world_replay\round_a` if desired. This saves the
existing structured dynamics replay, including rejected/setup interactions,
separately from accepted multimodal BC episodes. World replay has no continuous
video. Evaluation/update-gap rows are tagged and never fed back to PPO.

## 3. BC each architecture on those same episodes

```powershell
foreach ($candidate in @('mlp', 'recurrent', 'multi')) {
  C:\venvs\brawl312\Scripts\python.exe -m brawl.pipelines.round_a.run `
    --stage pretrain --candidate $candidate `
    --demos runs\legacy\round_a\collection_v1\demos `
    --run-dir "runs\legacy\round_a\bc_${candidate}_s42"
  if ($LASTEXITCODE -ne 0) { throw "BC failed for $candidate" }
}
```

This is offline: no game or capture required. A fixed split seed produces
disjoint 60/20/20 train/validation/test episodes within each drill. Identical
per-label weights and epochs are used across candidates. Episodes are shuffled,
time inside each episode is not. Recurrent BC carries hidden state through
context and detaches it every 32 rows (truncated backpropagation). It resets at
episode boundaries. Validation excludes unlabeled context; the test partition
is not used during training or checkpoint selection. BC saves the fixed final
epoch. The value predictor is not supervised by BC.

This fair common BC protocol differs from the historical primitive pipeline's
replacement sampler and 80/20 split. Therefore A0 is a new baseline trained on
the SAME new data/protocol as A1/A2, not a claim of identity with an old BC run.

## 4. Engineering pilot before the 2M campaigns

Use `--protocol configs\legacy\round_a\pilot_v1.json` for **both pretraining and
training**, in separate directories, to test 4096 steps and short evaluations.
Pilot BC uses eight epochs; a failed live qualification is evidence to inspect,
not permission to bypass the check. Production BC must be retrained using the
production protocol. Pilot checkpoints are never silently promoted into the
production comparison.

The synthetic path is safe without a game:

```powershell
foreach ($candidate in @('mlp', 'recurrent', 'multi')) {
  C:\venvs\brawl312\Scripts\python.exe -m brawl.pipelines.round_a.run `
    --stage smoke --candidate $candidate
  if ($LASTEXITCODE -ne 0) { throw "Smoke failed for $candidate" }
}
```

## 5. Fine-tune one candidate

Example for the MLP; replace `mlp` in BOTH candidate and checkpoint path for the
other arms. Each starts from its own BC initializer, not another candidate's PPO.

```powershell
C:\venvs\brawl312\Scripts\python.exe -m brawl.pipelines.round_a.run `
  --stage train --candidate mlp `
  --demos runs\legacy\round_a\collection_v1\demos `
  --resume runs\legacy\round_a\bc_mlp_s42\bc_init.zip `
  --run-dir runs\legacy\round_a\ppo_mlp_s42 `
  --world-replay data\world_replay\round_a
```

It first qualifies deterministic pickup/hold, recovery and combat against the
NON-ATTACKING bot. You must confirm visible execution with `YES`. Then it asks
you to configure ATTACKING Medium Mordex. There is no automatic menu control.

Production schedule:

- 15-minute frozen baseline, before updates.
- 2,000,000 requested training interactions, rounded equally to **2,000,384**
  (complete 512-step rollouts) for all arms.
- 5-minute frozen evaluations at the first rollout boundary after each 250k.
- One-hour final deterministic evaluation in six 10-minute blocks.
- Periodic checkpoints at rollout boundaries after each 50k, plus `latest.zip`.

Evaluation and neutral update-gap interactions are counted separately. Recurrent
state is reset at reporting boundaries, evaluation blocks and unobserved update
gaps; this limitation is part of v1, not a claim of unlimited memory. PPO's
time-limit bootstrapping remains enabled. Updates release all keys; one neutral
bridge refreshes state after downtime without attributing it to the next action.
The game is NOT paused or physically reset during these gaps. Gap time and bridge
outcomes are recorded. This is still a real-time, non-resettable benchmark.

The Round A imitation monitor is **report-only**, unlike the legacy primitive
retention stop. It does not stop successful PPO merely for disagreeing with the
teacher. Invalid sensing/model operations still stop and release inputs. There
is no claim that the monitor automatically detects every live sensor failure.

To resume, pass the periodic `latest.zip` or a completed-rollout `step_*.zip` to
`--resume` and use a NEW output directory, the same seed, dataset and protocol.
The remaining budget is used, not another 2M. A partial-rollout failure checkpoint
can be evaluated but cannot safely resume an exactly matched rollout budget;
use the last complete checkpoint and report discarded/extra interactions. Keep
all interrupted run manifests for total-cost accounting.

## 6. Frozen evaluation and bounded manual-review clips

```powershell
C:\venvs\brawl312\Scripts\python.exe -m brawl.pipelines.round_a.run `
  --stage evaluate --candidate mlp `
  --resume runs\legacy\round_a\ppo_mlp_s42\latest.zip `
  --demos runs\legacy\round_a\collection_v1\demos `
  --eval-video-seconds 60
```

This runs another frozen one-hour evaluation and reports test-set imitation.
Video is OFF by default. `--eval-video-seconds 60` records the first minute of
each block, at most six minutes for the hour, with frame timestamps. It also
works on `--stage train`. Use the SAME video/replay settings across scored arms;
recording overhead can affect cadence. Movie FPS is only playback; JSONL elapsed
timestamps are authoritative. Do not tune on this held-out test report.

Results are stock exchanges, NOT match wins. Damage and recovery/pickup metrics
are estimates from the current sensors. Recovery requires two grounded onstage
observations; death is checked first so respawn is not recovery. Missing identity
and unfinished events are censored and counted. Raw opportunity/censor counts
are included so short blocks cannot silently inflate success rates.

Review the same clip intervals for each candidate, and separately annotate
perception, execution, tactics, timing, memory, and reward-exploitation failures.
Do not call every attack that misses a bad tactical decision.

## 7. Reports and the actual decision

Each completed BC/training/evaluation run produces `report/learning_curves.png`,
`report/wall_clock_curves.png` and `report/summary.json`. Wall-clock curves include
updates and evaluation, start anew for each resumed segment, and do not include
earlier BC/collection: account for those separately from their manifests. Runtime
cadence counts are recorded by usage; compare missed deadlines as well as the
initial qualification. To compare runs without touching the game:

```powershell
C:\venvs\brawl312\Scripts\python.exe -m brawl.analysis.plot_round_a `
  runs\legacy\round_a\ppo_mlp_s42 `
  runs\legacy\round_a\ppo_recurrent_s42 `
  runs\legacy\round_a\ppo_multi_s42 `
  --output runs\legacy\round_a_comparison_s42
```

Primary comparison: final-budget net stocks/minute, with damage trade and live
skill outcomes alongside it. Report baseline quality, imitation retention and
training cost. There is no test-selected `best.zip` in this benchmark. One seed
is screening, not a publication-grade superiority claim. Repeat baseline and
finalists with at least three paired declared seeds; BC itself is rerun per seed.
Game trajectories are not reproducible just because a model seed is fixed.

All directories are new-only. Source hashes (including dirty files), dataset and
checkpoint hashes, dependencies, protocol and operator settings are recorded.
Old data/models are neither moved nor rewritten. The only shared collector
change adds source-step indices; it does not alter its acceptance rules.

## Weapon evidence correction (2026-09-23)

Live weapon fusion now gives executed input intent priority: nearby pickup is
provisionally armed for at most 12 subsequent perception steps, while throw sets
unarmed with an eight-step visual guard. Two fresh confident positive refreshes
establish possession; three fresh negative refreshes correct it. Cached results
are not extra votes, and evidence gaps longer than eight steps clear votes.
Death/respawn clears fusion state. These are conservative debounce defaults, not
measured optimal durations or a guarantee that the game accepted an input.

The collector requires confirmed possession before calling a later loss a drop.
A lone positive followed by negatives stays unconfirmed. The hold duration and
cadence tolerance are unchanged. Qualification reports pickup requests, peak hold
and negative refresh counts; replay schema 5 adds confirmation/pending flags.
Existing checkpoints still load (no action/observation dimension changes), but
this changes live sensing semantics. Keep old runs for provenance; do not pool
pre-fix and post-fix results as the same benchmark condition. Requalify all arms.

## Implementation verification (2026-09-23)

- Full regression suite: 774 passed, one existing tqdm warning.
- Final focused suite: 30 passed, including the subsequently added nonfinite
  observation safety guard.
- All three architectures: standalone synthetic learn/save/reload smoke passed.
- Nine new/existing CLI `--help` checks passed.
- Focused executable-line coverage was 89% before the final two-line guard;
  BC was 100%, collection 96%, data/model/runtime 90% or higher.
- Tests cover aligned labels/history, masked sequence BC, LSTM state lifecycle,
  complete training/evaluation callbacks, budget-preserving resume, typed
  checkpoint rejection, bounded clips and read-only plotting.
- No live gameplay, real-data collection or production training was performed.
- sb3-contrib 2.9.0 and coverage were installed into brawl312 without changing
  its existing dependencies. Git metadata was read-only; no commits were made.

## Decision-loop timing (2026-09-24)

The fusion-v2 MLP qualification passed all three drills but missed the 30 Hz
cadence gate (p95 decision interval 53.70 ms). Do not treat this as a BC failure,
or bypass the deadline. The earlier 47 Hz loop is not an equivalent workload.

Add `--profile-timing` to a `--stage train` command with a **new run directory**.
This saves `qualification_timing.json`, including on qualification exceptions or
gate rejection. It instruments only qualification, restores original methods
before PPO, and does not change rewards, model inputs, input timing, detector or
classifier cadence. Keep world replay enabled if it is intended for training.

The report separates policy, capture, detection, input, state estimation, weapon
fusion/classification, HUD/game logic, structured observations, image history,
CSV writes, replay appends, and pacing sleep. Times are **inclusive**: base-step
and match-step include nested work, so do not add all rows. Host wall timing adds
no GPU synchronizations; GPU waits can be charged to a downstream synchronization.
Percentiles retain at most 8192 recent calls; means/totals include initialization.

### Offline optimization evidence

Bounded search: existing CPU MLP prediction, OpenCV resize threading, and history
copying. No live key injection, capture, model updates or archive changes.

| Variant | Repeated evidence | Decision |
| --- | --- | --- |
| OpenCV one thread vs default eight | Resize ordering reversed between repeats; no consistent improvement | Keep default |
| Copy four selected frames directly to CHW | Byte-identical at both mirror orientations; faster in every paired repeat | Adopt |
| MLP CPU prediction | Median 0.315-0.477 ms; not the tens-of-ms bottleneck | Keep CPU |

Confirmation run (`work/round_a_overhead_v1.json`): history median
0.278-0.462 ms before vs 0.161-0.173 ms after; mirrored median
0.277-0.325 ms before vs 0.233-0.248 ms after. Initial runs showed larger deltas;
the smaller repeated measurements are the conservative evidence. Full-frame
INTER_AREA resize alone measured 4.261-5.202 ms median. Its interpolation and
resolution are unchanged to preserve the visual benchmark contract. These are
microbenchmarks, **not evidence that live 30 Hz now passes**.

Reproduce without launching the game (new output filename each time):

```powershell
C:\venvs\brawl312\Scripts\python.exe -m scripts.round_a.benchmark_round_a_overhead `
  --checkpoint runs\legacy\round_a\pilot_bc_mlp_40ep_s42\bc_init.zip `
  --output work\round_a_overhead_repeat.json
```

The old copy implementation remains only inside the benchmark for A/B checks;
rollback of the optimization is confined to `stack_frames` and its live caller.
