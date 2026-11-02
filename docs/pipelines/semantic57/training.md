# Semantic controls: runnable BC → live PPO pipeline

This is a **new, opt-in policy contract**, not an in-place conversion of existing
checkpoints. Old Discrete(27) demos and 146/168-dimensional policies are rejected.
The previous entry points, checkpoints and archives remain available unchanged.

The new path builds one goal-free MLP policy on CPU, warms it from integrated
teacher demonstrations, and fine-tunes it with plain masked PPO. It does **not**
use FiLM, an HSP or anchored-replay PPO. That gives the new input contract a small,
testable training baseline before adding algorithmic changes. No competitive or
optimal-performance claim follows from completing the software pipeline.

## Before a live run

- Wu Shang player, Mordex opponent, Small Brawlhaven, training-room respawns.
- Keep the existing configured controls and HUD positions. The script does not
  select characters, move HUD slots, configure the bot, or navigate menus.
- Confirm recovery, directional dodge, pickup/throw and explicit gravity cancels
  with the [control qualification tool](controls-handoff.md).
  `completed` means input injection, not independently verified game execution.
- The weapon classifier defaults to `data/models/weapon_classifier_r18_live_v1.pt`,
  matching the latest validated live diagnostic manifests rather than the older model.
  Override `--weapon-classifier-checkpoint` if your validated model is elsewhere.
- Use the specified CUDA-capable interpreter even though the policy runs on CPU:
  perception still uses the installed GPU stack. Do not disable CUDA globally.

## One command: fresh collection, BC, then live PPO

From the repository root, use a **new** output directory:

```powershell
C:\venvs\brawl312\Scripts\python.exe -m brawl.pipelines.semantic57.run `
  --stage all --experimental-controls `
  --run-dir runs\legacy\semantic57\pilot_01 `
  --episodes-per-drill 5 --bc-epochs 8 `
  --timesteps 4096 --n-steps 512 --batch-size 256 `
  --eval-every 4096 --eval-episodes 2 --window-steps 1200 `
  --world-replay data\world_replay\semantic57
```

The command prompts twice:

1. Configure moving/non-attacking Mordex for collection. Three drills create
   weapon acquisition, recovery and general combat interactions. The same teacher
   prioritizes recovery, visible weapons, safe positioning, spacing and combat
   in every drill; it does not use a hidden goal unavailable to the final policy.
2. After collection and BC, configure attacking **Medium Mordex** for PPO.

Each prompt is followed by a five-second switch-to-game countdown. `--no-prompt`
keeps the bot configuration unchanged across stages; it does not change difficulty
for you. Do not use it if you intend to switch bot behavior between stages.

Setup actions are discarded from BC: recovery walks/jumps offstage before arming;
weapon setup drops an existing weapon and waits for confident unarmed evidence.
Pickup acceptance then requires fresh visual confirmation and a 40-control-step
hold. Attempts and step budgets are bounded. Missing quotas stop the pipeline
before BC/PPO and save `partial_demos.npz` when any episodes were accepted.
Ctrl+C during collection also preserves completed accepted episodes.

For a larger **fresh** campaign, change `--episodes-per-drill` to 20, `--timesteps`
to 1000000, `--eval-every` to 51200, and optionally add `--hours 5`. Use another
new run directory. The time cap applies to the PPO stage, not collection or BC;
it is checked during rollout steps, so an ongoing evaluation/update can overshoot.

## Continue PPO without recollection or repeating BC

After reviewing the pilot, resume into a **new** directory:

```powershell
C:\venvs\brawl312\Scripts\python.exe -m brawl.pipelines.semantic57.run `
  --stage train --experimental-controls `
  --resume runs\legacy\semantic57\pilot_01\latest.zip `
  --demos runs\legacy\semantic57\pilot_01\demos.npz `
  --run-dir runs\legacy\semantic57\campaign_01 `
  --timesteps 1000000 --hours 5 --n-steps 512 --batch-size 256 `
  --world-replay data\world_replay\semantic57
```

`--timesteps` is an additional interaction budget (SB3 rounds to full rollouts).
The first limit reached—timesteps or hours—ends training. `--learning-rate` and
`--ent-coef` can be adjusted explicitly on resume. `--demos` is optional for PPO;
supplying it enables held-out teacher-agreement diagnostics, **not a BC loss or
replay anchor**. Resume from `interrupted.zip` after Ctrl+C if present. A
`failed_run.zip` is a diagnostic snapshot, not a promoted checkpoint.

Separate stages are also supported: `collect` writes the new archive; `pretrain
--demos PATH` runs offline BC; `train --resume PATH` runs live PPO. Each stage uses
its own new directory. A pretrain-stage `bc_init.zip` is a valid train input.

## Deterministic evaluation

```powershell
C:\venvs\brawl312\Scripts\python.exe -m brawl.pipelines.semantic57.run `
  --stage evaluate --experimental-controls `
  --resume runs\legacy\semantic57\campaign_01\latest.zip `
  --run-dir runs\legacy\semantic57_eval_01 `
  --eval-episodes 2 --window-steps 3000
```

These are training-room reporting windows, not independently reset episodes or
real match wins. The policy still senses live state and applies action masks.

## What is optimized and recorded

Reward is unchanged `MatchObjective.stock_damage_v1`: +5 per opponent stock lost,
−5 per own stock lost; +0.01 per measured damage dealt and −0.005 per damage
received, each damage budget capped at 100 damage per corresponding life. Damage
is gated on valid observations and excluded on stock-change frames. No income
comes merely from pressing pickup, holding a weapon, attacking or standing close.
Unlimited-respawn training has no match-win bonus.

PPO uses rollouts of 512, minibatches of 256, four epochs, learning rate 1e-4,
gamma 0.999, GAE 0.95, clip 0.1, entropy coefficient 0.01, target KL 0.01. Policy
and optimizer are CPU-only; inputs are released during updates. A fresh neutral
observation bridges each optimizer pause outside PPO's rollout data.

The observation is 267 values: 146 existing state/history, 64 executor state, 57
action-mask flags. The mask is excluded from neural features but used directly
on logits, including during PPO's reevaluation of stored observations. Executor
state makes pending sequences and timing guards observable. A masked request
raises instead of silently substituting another action.

Artifacts include:

- `config.json`, `status.json`, collection counts and action/family histograms.
- `collection_events.jsonl`: pickup keypresses, fresh visual weapon evidence,
  interrupted inputs, exact execution failure reasons, and per-attempt summaries.
  This lightweight log does not record video.
- `demos.npz`, `bc_split.json`, `bc_metrics.csv`, `bc_validation.json`.
- `bc_init.zip`, periodic checkpoints, `latest.zip`, and deterministic-evaluation
  `best.zip`, each with `.metadata.json` and embedded contract metadata.
- `windows.csv`, `train_transitions.csv`, `evaluations.csv`, `ppo/progress.csv`,
  and `training_progress.png` when reporting windows complete.
- `bc_agreement.jsonl` when demos are available. This is episode-held-out teacher
  agreement, not an amnesia/retention benchmark on live skills.

`--world-replay` saves separate versioned collection/training sessions, including
pending input states, sampled masks, timing, requested actions and reward evidence.
Collection replay includes deliberately discarded setup and failed attempts;
only `demos.npz` contains accepted imitation labels. No video is recorded. Do not
combine these 267-dimensional/57-action sessions with legacy replay as one schema.

## Limits and promotion gate

BC validation splits whole collection episodes, but adjacent episodes from the
same live session remain correlated. This is not a multi-session generalization
claim. Some actions, particularly explicit gravity cancels, may have zero teacher
labels; the action histogram exposes that. Those actions remain available for
PPO exploration. The teacher is a warm start, not a combo expert.

Software tests cannot validate HUD correctness, actual move execution, combo
timing or eventual strength. Inspect the pilot and compare BC vs PPO using the
same bot settings before increasing the budget. Do not infer competitive play
from loss, imitation accuracy, or a `completed` run status.

### Collection repair after the first live pilot

Executor `semantic_sequence_candidate_v2` distinguishes completed input injection
from a context change before the minimum hold. Release-only cleanup no longer
requires the original identity/ground context. Early identity loss still releases
all keys immediately; the collector removes the interrupted request's BC label
and retains earlier valid decisions. Missed gravity-cancel deadlines and rejected
requests still reject the attempt. A completed injection is not proof that the
game accepted the intended move.

The weapon drill now requires an actual PICKUP keypress before evaluating armed
evidence. The capture from the keypress step is excluded because capture precedes
input injection. Confidence, hold duration, and sustained-drop rejection remain
unchanged; this does not make classifier predictions infallible.

Use a fresh run directory (for example `runs/legacy/semantic57/pilot_02`). Older
executor-v1 demos and checkpoints are intentionally rejected by contract checks;
do not reuse the first pilot's partial archive for BC. Its evidence is preserved.

Hardware-free artifact smoke test (synthetic data, never a game fallback):

```powershell
C:\venvs\brawl312\Scripts\python.exe -m brawl.pipelines.semantic57.run --stage smoke
```

This exercises serialization, BC, PPO, reload, evaluation and plotting without
capture or keyboard input. It says nothing about game performance.

Implementation verification (2026-09-20): 667 headless tests passed with the
specified interpreter; one existing tqdm experimental warning. Eight entry-point
`--help` checks passed. A separate CLI smoke run is retained under
`runs/legacy/semantic57/smoke_20260920`. Targeted stdlib trace measurement
covered 91.8% of executable source lines in the eight new semantic training
modules (not branch coverage). No live game run was launched by this implementation.

After the first-pilot collection repair: 680 headless tests passed (the same
tqdm warning), and the semantic pipeline `--help` check passed. Targeted stdlib
trace measured 95.5% executable-line coverage for the executor and 94.1% for the
collector; these are not branch-coverage figures. Live recollection remains the
next validation gate.
