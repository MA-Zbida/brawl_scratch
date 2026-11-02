# The discrete27 fighting baseline

Use `python -m brawl.pipelines.primitive27.run`. This is the main fighting-baseline
experiment; semantic57 remains intact for comparison. No old archive or checkpoint
is overwritten, converted, or silently reused.

For named configurations, provenance records and the shared-runner architecture,
see [Experiment workflow](../../experiment-workflow.md). The existing CLI remains valid;
`--experiment` and `--set` are optional.

The policy uses the unchanged 27 canonical actions and 146 state/history values
at offsets (2, 4, 8). Ground/air outcomes remain context-dependent. Execution uses
the existing poll-visible tap sequencer, not semantic macros. Identity loss
releases inputs. Cadence-deferred tap requests are excluded from BC labels unless
their physical key edge was emitted. This is not proof that the game accepted a
move; the live gate and operator inspection are still required.

## First run

From the repository root in PowerShell:

```powershell
C:\venvs\brawl312\Scripts\python.exe -m brawl.pipelines.primitive27.run `
  --stage all `
  --episodes-per-drill 30 --bc-epochs 40 `
  --timesteps 4096 --n-steps 512 --batch-size 256 `
  --eval-every 4096 --eval-episodes 2 --window-steps 1200 `
  --world-replay data\world_replay\primitive27
```

Output defaults to a fresh timestamped `runs/legacy/primitive27/` directory.
You may supply a NEW `--run-dir`. The runner prints its absolute path.

1. Configure Wu Shang, moving **non-attacking Medium Mordex**, Small Brawlhaven,
   unlimited-respawn training room. Confirm the collection prompt and switch back
   to the game during the five-second countdown.
2. The teacher collects weapon, recovery, and combat drills. These are sampling
   contexts, not three separate policies. Movement and defensive decisions are
   included. Recovery setup intentionally goes offstage; those setup frames are
   excluded from BC but retained in optional dynamics replay. Weapon setup drops
   the weapon using the existing interaction key, then requires a new pickup and
   visually confirmed hold. Combat needs an actual attack press and measured damage.
3. Inputs are released and BC runs offline. Its bounded square-root inverse-action
   sampler lifts rare actions (weights capped at 8), using training episodes only.
   New policies use `PrimitiveGeometry`: internally scale weapon dx/dy by 20
   and append their Euclidean distance at the same scale. This exposes the small
   interaction neighbourhood without changing the 146-value observation or the
   27 actions, adding a sensor, masking an action, or forcing pickup. The fixed
   transform is serialized with the policy and used identically in BC/PPO/live
   inference. Existing checkpoints keep their original preprocessing.
4. Offline gate: validation holds out whole episodes from **each drill**. Required
   exact action recall is at least 60% on each of attack, ground attack, pickup, and
   recovery slices, with at least 20, 10, 5, and 20 held-out rows respectively.
   Insufficient evidence fails; aggregate accuracy cannot override a failed slice.
   These are initial engineering gates, not statistically established mastery.
   `pickup_diagnostics` also reports exact-label precision and predictions while
   armed, without a ground weapon, or beyond the teacher's pickup radius. These
   diagnostics are not new gates: exact timing can differ because the teacher
   has an internal retry timer. Inspect the independent live pickup/hold check.
5. Keep the bot non-attacking for a separate deterministic live check. The policy
   must complete each drill in at most three attempts. The heuristic only performs
   discarded setup, never replaces the evaluated policy's decisions. Observe the
   game, then type `YES` only if attacks, pickup/hold, and recovery actually worked.
   Passing sensor checks alone is not trusted as control qualification.
6. At the next prompt, switch Mordex to **attacking Medium**, confirm, and return
   to the game. CPU PPO begins with measured stock/damage reward.

No menu automation or bot-setting detection exists. Do not use `--no-prompt` for
`all` or `train`; it cannot bypass qualification. There is no reward for issuing an
attack or pickup request. Reward remains `MatchObjective.stock_damage_v1`.

This run is pooled goal-free BC followed by **plain PPO**. It does not run GCRL,
PCGrad, anchored-replay PPO, a BC auxiliary PPO loss, an LSTM, or a world model.
The explicit alternative to an unvalidated retention algorithm in this baseline
is a monitored stop. Initial admission still requires the unchanged offline gate.
After each PPO update, compare each slice with its **fixed run-start agreement**:

- Below 90% of starting agreement: warn; stop after three consecutive distinct
  updates below that ratio on the same slice. Recovery resets that slice's streak.
- Below 50% of starting agreement: stop immediately.
- Missing/nonfinite evidence or changed slice counts: stop immediately.

Repeated logs at the same training step do not count as additional updates.
`bc_agreement.jsonl` records both the unmodified BC report (which may say `passed:
false`) and a separate `retention` decision. Stops save `retention_stop.zip` and
the reason in `retention_stop.json`. The initial anchor never ratchets down.
These are engineering tolerances, not a statistical test or measured live skill
retention. A policy may find a different effective action; inspect live results
rather than treating disagreement as proof of catastrophic forgetting.

## Failure and resumption

- Incomplete collection saves `partial_demos.npz` and stops before BC.
- Failed BC saves `bc_candidate.zip`, `bc_gate.json`, and its split/metrics. Do not
  solve a failed gate by removing it. Inspect scarce/mislabeled actions first.
- A complete `demos.npz` can be retrained offline with `--stage pretrain --demos
  <path> --bc-epochs <count>` in a new output directory. No recollection is needed
  solely to adjust BC optimization.
- `--stage train --resume <typed primitive27 checkpoint> --demos <matching demos>`
  rechecks offline agreement and deterministic live qualification before PPO.
  A new run establishes a new baseline only after those admission checks; this
  is not a way to bypass a failed initializer. For the 20260922 one-label pickup
  stop, restart from `pickup_geometry_v1/bc_init.zip`, not `retention_stop.zip`.
- `--stage evaluate --resume <typed checkpoint>` runs deterministic reporting
  windows without optimization. It is diagnostic, not a promotion bypass.
- Ctrl+C releases inputs and saves the last available training checkpoint. A
  partially collected PPO rollout is not itself an optimizer update.

Evidence includes contracts/config, collection events and histograms, split IDs,
BC metrics and slice gates, live-gate attempts, operator confirmation with checkpoint
hash, `bc_agreement.jsonl`, PPO metrics, windows, deterministic evaluations, typed
checkpoints, and `training_progress.png` after a completed training run with windows.
`--world-replay` records separate collection/qualification/training sessions without
video. Keep their schemas separate from semantic57 and goal-conditioned archives.

## Boundaries

The live check does not qualify frame-perfect gravity cancels or long combos.
Opponent resources remain unobserved/estimated, not invented exact counters.
These unlimited-respawn windows are not match wins. A successful software smoke
test (`--stage smoke`) validates model save/load and synthetic PPO, not game skill.
Do not start a multi-hour run until the short pilot's live evaluation is credible.

## Verification (2026-09-22)

The specified brawl312 interpreter passed all 710 headless tests (one existing
tqdm experimental warning), seven entry-point help checks, and a standalone
synthetic CPU PPO/save/reload smoke run. Focused stdlib tracing measured 92.0%,
87.9%, and 100% executable-line coverage for the new training module, runner,
and environment wrapper respectively; these are not branch-coverage figures.
No live game inputs were issued during implementation. Offline and live gates
remain necessary before treating a checkpoint as a fighting-capable initializer.

Pickup geometry follow-up: all 713 headless tests pass, including new transform
parity, PPO/save/reload, and false-positive diagnostic regressions. The existing
20260922-204156_198297 demos were retrained without recollection; the candidate in
`runs/legacy/primitive27/pickup_geometry_v1` passes the unchanged offline gate
(20/33 pickup labels). Its `repair_notes.md` records the baseline, unsuccessful
normalization trial, tradeoffs and validation-informed nature of the repair.
This narrow offline pass still requires independent live qualification.

Retention follow-up (2026-09-23): the live-qualified initializer's first update
changed pickup agreement from 20/33 to 19/33 while other slices changed by less
than one percentage point. Reusing the admission cutoff made that one-label
change fatal. Baseline-relative monitoring now permits it without altering the
admission gate, labels, rewards or PPO optimizer. All 723 headless tests pass;
69 focused tests cover both runners, checkpoint/BC behavior and retention.
Saved reports from the stopped run replay as 100% then 95% pickup retention,
both permitted. This is a guardrail correction, not evidence of improved play.
