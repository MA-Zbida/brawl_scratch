# Experiment workflow

The baseline is `structured27_bc_ppo_v1`: structured observations plus fixed
weapon geometry, a feedforward 128x128 actor/critic, heuristic BC, then plain CPU
PPO. This is not a goal-conditioned, recurrent, anchored-replay or CNN policy.
The separate [Round A benchmark](pipelines/round_a/benchmark.md) now supplies recurrent and
visual-plus-structured adapters with aligned data and timed evaluation. They are
not switches in this legacy baseline configuration, and are not yet live-qualified.

## Boundaries

| Responsibility | Current location |
| --- | --- |
| Live capture, controls, state | `brawl/env/live.py`, `brawl/capture/`, `brawl/control/`, `brawl/state/` |
| Match objective and reporting windows | `brawl/reward/match_objective.py`, `brawl/env/match.py` |
| Structured representation | `brawl/pipelines/primitive27/features.py` |
| Model construction, BC and demo contract | `brawl/pipelines/primitive27/training.py` |
| Shared PPO/evaluation lifecycle | `brawl/training/runner.py` |
| Backend-specific monitoring hooks | `TrainingHooks` in the shared runner |
| Shared CLI validation | `brawl/training/cli.py` |
| Configuration and provenance | `brawl/training/experiment_config.py` |
| Operator prompts and artifact utilities | `brawl/training/run_utils.py` |
| Stable user entry point | `brawl/pipelines/primitive27/run.py` |
| Experimental semantic57 entry point | `brawl/pipelines/semantic57/run.py` |

The primitive entry point no longer imports the semantic entry point. Both use
the same runner; the runner receives metadata, agreement, split and retention
hooks instead of importing either action-space implementation. Semantic57 keeps
its original monitoring cadence. Compatibility wrappers preserve old helper
imports. Serialized policy classes have not moved: old checkpoint module paths
still resolve. Existing commands still work without an experiment file.

Collection's existing duck-typed controls interface is reused unchanged. Moving
all data/teacher code or migrating to `src/` is deliberately outside this change.

## Run a named experiment

From the repository root in PowerShell:

```powershell
C:\venvs\brawl312\Scripts\python.exe -m brawl.pipelines.primitive27.run `
  --experiment configs\legacy\primitive27\structured27_bc_ppo_v1.json `
  --stage train `
  --resume runs\legacy\primitive27\pickup_geometry_v1\bc_init.zip `
  --demos runs\legacy\primitive27\20260922-204156_198297\demos.npz `
  --world-replay data\world_replay\primitive27
```

The baseline file specifies the completed pilot's 4096-step budget, 512-step
rollouts, batch 256 and two deterministic evaluation windows. The legacy CLI
default for `eval_episodes` remains three; explicit CLI values still take priority.
Keep Mordex non-attacking for qualification, then switch to attacking at the
prompt. The configuration documents these settings but cannot set or verify them.

For an offline smoke test, use the same experiment file with `--stage smoke`
and omit `--resume`/`--demos`. Smoke uses its original tiny synthetic budget;
`model_config.json` records those effective overrides. It does not test game skill.

Precedence is **defaults < experiment parameters < explicit CLI < `--set`**:

```powershell
# Append these to the training command when intentionally changing the budget:
--set timesteps=8192 --set learning_rate=0.00005
```

`--set` accepts existing CLI parameter names only (underscores or hyphens).
Values use JSON types where applicable: `false`, not the string `"false"`.
Paths have the same meaning as CLI paths: relative to the working directory,
not the JSON file. Typos, duplicate JSON keys, invalid types, unknown schema
versions and unsupported component combinations fail before capture starts.
CLI/config cannot bypass live qualification or the retention guard.

## What is recorded

- Existing `config.json`, gates, checkpoints, metrics and replay remain intact.
- `resolved_experiment.json`: merged parameters, selected components, software
  versions, source-content hashes, input/perception hashes, loaded policy class,
  bot protocol and retention settings. This is an execution record, not itself
  an input configuration; it is intentionally not accepted by `--experiment`.
- `model_config.json`: effective model/optimizer hyperparameters and representation
  after construction or resume overrides, including non-CLI defaults.
- Existing run directories are never reused automatically.

The historical pilot is identified in
`results/legacy/primitive27/structured27_bc_ppo_v1.artifacts.json`. It references files
in place with hashes; it is not a backup. Perception hashes were observed when
freezing the baseline, not recorded retroactively as historical certainty.
No demos, checkpoints or videos were moved or deleted.

## Adding a method

1. State the hypothesis and which axis changes: representation, memory,
   optimization, initialization or objective. Keep unrelated axes fixed.
2. Add the implementation behind a model/backend factory. New algorithms need
   their own tested loading/compatibility path; a config label is not an adapter.
3. Supply `TrainingHooks` for bookkeeping. The shared runner currently expects
   the SB3 on-policy lifecycle. Do not claim it supports arbitrary RL libraries.
4. Add observation/action/checkpoint compatibility tests. Recurrent policies also
   need hidden-state lifecycle support in collection and evaluation; these are
   not yet implemented by this runner.
5. Register the supported combination in configuration validation and add a
   versioned experiment file. An explicit geometry experiment refuses an old
   checkpoint with a different encoder; legacy CLI loading is preserved.
6. Reuse the evaluation protocol, budgets and reporting. Track wall-clock and
   environment interactions, including setup/evaluation costs separately. Keep
   retention stopping rules consistent and report early stops.

Do not change rewards, sensors or action timing inside a structural refactor.
Training-room stock exchanges are not match-win measurements. The pilot is a
working baseline, not a statistically established performance improvement.

## Refactor verification (2026-09-23)

- 745 headless tests passed with the brawl312 interpreter (one existing tqdm warning).
- Seven CLI entry-point help checks passed.
- The pre-refactor BC and PPO checkpoints produced bit-identical probability
  vectors on 256 recorded observations after extraction.
- The original and extracted runners produced bit-identical policy/value tensors
  after two synthetic PPO updates with deterministic evaluation enabled.
- Focused executable-line coverage: shared CLI 92.2%, runner 97.5%, utilities
  93.8%, experiment configuration 90.1%. These are line, not branch, figures.
- No live game input, archive rewriting or checkpoint conversion was performed.
