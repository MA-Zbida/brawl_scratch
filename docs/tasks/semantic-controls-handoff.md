# Weapon gate and semantic controls: implementation handoff

Status: **active collector/teacher fixes plus an opt-in semantic57 training pipeline.**
Existing entry points and policies still use Discrete(27). The separate
`train.semantic_pipeline` entry point collects new demos, pretrains BC and runs
masked CPU PPO. See [the runnable guide](../semantic-training.md).

## Why this gate exists

The user owns promotion. The objective is a trainable Wu Shang fighter against
Mordex, not a larger action count. Wrong BC acceptance and wrong physical moves
are unacceptable training labels. Brief abstention and rejected control requests
are preferable to silently claiming a pickup, recovery, or gravity cancel happened.

Baseline: the existing 27-action checkpoints and demos remain intact. None have
been relabelled, deleted, overwritten, or retrained by this change.

## Active fixes

`perception/weapon_evidence.py` is shared by the stage reward wrapper and collector.

- Fresh, valid predictions must meet the classifier's confidence threshold (0.70
  by default). A valid crop alone is not sufficient evidence.
- Repeated cached negatives count as one negative refresh, not multiple drops.
- Confirmation expires after eight control steps without trusted new evidence.
- A hold can bridge short classifier gaps, but acceptance requires a fresh armed
  observation at completion. Unknown evidence never counts as a disarm.
- `--weapon-drop-grace-steps` is retained for CLI compatibility, but its unit is now
  **fresh confident negative refreshes** when vision is enabled. With value 3, the
  fourth such negative rejects the episode. Action-only mode still counts steps.
- Archive metadata records `weapon_evidence_version` and `weapon_drop_grace_unit`;
  attempt logs expose negative refresh count and evidence age.

This fixes a gate bug, not every possible classifier error. Confident overlap
misclassification can still produce false evidence. A real lost weapon still
resets hold progress and sustained fresh negative evidence rejects the episode.

The legacy recovery teacher now avoids neutral spot dodges in its horizontal
deadband and waits 150 ms after its own directional-dodge request before requesting
heavy. It drifts during that interval. This is a conservative timing guard, not
proof of game acceptance. Its existing heavy cooldown and action IDs are unchanged.
The guard tracks teacher requests, not an independently observed successful dodge.

Tests: `test_weapon_evidence.py`, the weapon collection regression in
`test_demo_collection_quality.py`, visual-confidence regression in
`test_stage_goal_env.py`, and recovery regressions in `test_heuristic_teachers.py`.

## Candidate semantic controls

`control/semantic_actions.py` defines a **57-action candidate**, separate from
`action_space.py`. `control/semantic_executor.py` implements non-blocking input
sequencing. This is a starting control vocabulary, not a claim of optimality or
complete coverage of every game mechanic.

| Family | Candidate actions |
| --- | ---: |
| Locomotion, jumping, fast falling | 9 |
| Ground spot dodge and directional dashes | 3 |
| Air dodges | 9 |
| Ground light attacks | 4 |
| Ground signatures | 4 |
| Air light attacks | 4 |
| Recovery with neutral/toward/away steering | 3 |
| Ground pound with neutral/toward/away steering | 3 |
| Explicit gravity-cancel light/signature sequences | 8 |
| Pickup | 1 |
| Directional throws | 9 |

Grounded state gates recovery; **offstage is not required**. Unknown ground/air
state does not authorize a context-specific move. Unknown resource availability
does not automatically ban a dodge. Pickup and throw share a physical key but
have different possession conditions.

GC starts with a neutral dodge and then injects the selected ground attack.
Ordinary air attacks are rejected during the post-dodge guard. Physical direction
is fixed at sequence start, so a canonical mirror flip cannot reverse an ongoing
sequence. Late GC input is cancelled; obsolete requests are never queued.
Every result distinguishes started/pending/released/completed/rejected/cancelled.
**Completed means the input sequence completed, not that the game accepted it.**

The game has historically changed directional gravity-cancel timing and explicitly
discussed unintended GC instead of recovery in its [developer patch notes](https://store.steampowered.com/news/posts/?appids=291550&enddate=1643814009&feed=steam_community_announcements).
Those historical numbers are not treated as current installed-build guarantees.
Candidate timings require direct verification on this setup.

Not implemented in this candidate: independently controlled facing versus drift,
charged attack durations, chase-dodge qualification, dash-jump sequencing, or any
multi-second combo macro. Neutral/down attacks retain the game's existing facing.
The context is supplied by the caller. The new policy observes executor state and
sampling masks; the legacy StateSpec and its existing observation remain unchanged.

The executor advances on caller ticks; it has no background watchdog. Delayed
ticks can prolong a held key. Live checks must therefore include real perception
load before training promotion. `close()` and context-manager exit release keys.

## Commands

All commands below are from the repository root. Listing and dry runs do not
import the detector, start training, or inject input.

```powershell
C:\venvs\brawl312\Scripts\python.exe -m tools.check_semantic_actions --list

C:\venvs\brawl312\Scripts\python.exe -m tools.check_semantic_actions `
  --action GC_SIGNATURE_TOWARD --context air --mirrored
```

For a **single manual input test only**, add `--execute --delay 5`. Be in the
declared state when the countdown ends. Test recovery while airborne above safe
stage first, not near a blast zone. The tool trusts the manually asserted state;
it does not sense or verify it. For throws/pickup, also declare `--weapon-state`.
An optional `--output <new-file.json>` writes the trace and refuses overwrites.
Record or inspect the actual animation separately. A dry-run pass is not live proof.

Suggested live matrix: both canonical mirror directions; grounded/airborne;
unarmed/spear/gauntlets; recovery alone; directional dodge then recovery; explicit
GC light and signature; pickup versus throw. Check missed inputs and wrong moves,
not merely key trace completion.

## Training implementation and remaining live gates

The new environment supplies 146 state/history values + 64 executor values +
57 sampling-mask values. Only the new semantic policy consumes this 267-value
contract. BC, PPO sampling, PPO reevaluation and deterministic evaluation use the
same masked distribution. Metadata rejects legacy/reordered action registries.
Busy sequences admit only NOOP (advance the executor); those forced ticks remain
in PPO/replay but are omitted from BC labels.

`train.semantic_pipeline` provides `all`, `collect`, `pretrain`, `train`, `evaluate`
and hardware-free `smoke` stages. It does not migrate the old anchored-replay PPO
or old final-agent runner. It uses plain masked PPO, a fresh goal-free MLP warmed
from integrated-teacher demonstrations, and unchanged stock/damage rewards.
It is not the old goal-conditioned curriculum checkpoint with a larger head.

Remaining live gates: qualify input timings under actual perception load, inspect
new accepted acquisition/recovery demonstrations, then run a short Medium Mordex
pilot. No headless test proves game acceptance, competitive performance, or combo
mastery. The CLI requires `--experimental-controls` to make this distinction
explicit. Returning to the old entry points is the rollback; old artifacts remain
untouched.

## Offline verification

The new evidence and semantic control/check modules have 100% function-line
coverage under a targeted stdlib tracing run (generated dataclass/library code
excluded). This is not branch coverage and does not replace live input tests.
The targeted suite also exercises the live CLI path with a fake clock and fake
controller; no real keyboard injection is needed for the test suite.

```powershell
$env:BRAWL_NO_EARLY_CAPTURE = '1'
C:\venvs\brawl312\Scripts\python.exe -m pytest tests/ -q
Remove-Item Env:BRAWL_NO_EARLY_CAPTURE
```

The collection, curriculum collection, validator, BC pretraining, curriculum
training, and semantic-check `--help` entry points were checked with the specified
interpreter. No live collection or training run was launched during implementation.
