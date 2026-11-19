# Lab notebook

One entry per stage: what was run, what came out, what was decided. Newest entries at the bottom.

Template:

```
## YYYY-MM-DD: Stage N, <name>
Command:  ...
Result:   (paste the key rows of summary.md)
Reading:  what the result means for the hypothesis
Decision: what happens next, and why
```

Entries up to 2026-09-27 were **reconstructed on 2026-11-02** from `docs/`, `research/archive/`
and the run records under `runs/legacy/`. Numbers are as those sources report them; they were not
re-derived. All of this work predates the research hypothesis and is exploratory: one seed, short
evaluations, protocols that changed between runs.

---

## 2026-03-15 to 04-19: Era 0, goal-conditioned RL prototyping (reconstructed)

**What:** UVFA and FiLM goal conditioning, HER, SAC, PPO with replay and a BC loss against
forgetting. Reports: [reports/Brawlhalla_Adv.tex](reports/Brawlhalla_Adv.tex) ("Stabilizing
Goal-Conditioned PPO"), [reports/GCRL.pdf](reports/GCRL.pdf).

**Result:** skill collapse and an "always jump" exploit (commit history).

**Decision:** superseded by a phased skill ladder (Era 1).

---

## 2026-06-08 to 06-10: Era 1, LLC mastery ladder (reconstructed)

**What:** six-phase recovery-first ladder with retention and amnesia gates and anti-collapse
tooling (`scripts/llc/`). Protocol: [archive/llc_mastery_run_protocol.md](archive/llc_mastery_run_protocol.md),
goal matrix: [archive/llc_goal_evaluation_matrix.md](archive/llc_goal_evaluation_matrix.md).

**Decision:** superseded after the perception and observation rebuild (Era 2) invalidated the
demos and observation layout it used.

---

## 2026-08-12: Era 2, perception, observation and action rebuild (reconstructed)

**What:** `Discrete(27)` actions relative to the opponent, the 146-dim state/history observation,
mirroring to a canonical side, CI. See [../docs/observation-space.md](../docs/observation-space.md),
[../docs/architecture.md](../docs/architecture.md), [../docs/performance.md](../docs/performance.md).

**Result:** the foundation for everything after. All earlier demonstrations were invalidated.
Measured control rate 27.5-30.7 Hz, with detection about 93% of each step.

---

## 2026-08-13 to 09-11: Era 3, easy_bot_v0 sequential pilot and the final agent (reconstructed)

**What:** a frozen protocol ([../configs/legacy/llc/easy_bot_v0/protocol.md](../configs/legacy/llc/easy_bot_v0/protocol.md),
manifest frozen 2026-08-13), run by `scripts/llc/easy_bot_v0/run_chain.ps1`; then the final agent
(`brawl.pipelines.final_agent`) initialized from the skill policy. Paper draft:
[paper/main.tex](paper/main.tex).

**Result:**

- 3 h pilot skill-score matrix (later marked superseded): movement 0.726 to 0.415, weapon 0.575 to
  0.000, combat 0.042.
- An audit then found three defects: (0, 0) used to mean "no weapon"; observations aliasing a
  shared buffer; 18-64% of demo episodes only one step long.
- Corrected chain: movement attempt 3 scored 0.060; evaluations of only 2-3 episodes;
  `publication_ready: false`.
- Final agent `mordex_5h_baseline` (420k steps): net stocks per minute negative at all 9
  evaluations (-0.30 to -2.09).

**Decision:** stalled; work moved to sensing (Era 4).

---

## 2026-09-11 to 09-13: Era 4, final-agent diagnostics and sensing (reconstructed)

**What:** recorded diagnostics (`scripts/final_agent/record_final_diagnostic.py`), live weapon-crop
labelling, the `weapon_classifier_r18_live_v1` classifier and visual fusion with input intent.
Versioned contracts: `poll_visible_edges_v2` (input), `validated_hud_v2` (HUD),
`bounded_triangle_v2` (identity). See [../docs/pipelines/final_agent/diagnostics.md](../docs/pipelines/final_agent/diagnostics.md).

**Result:** the sensing fixes carried forward to every later pipeline. The `weapon_visual_v1_pilot`
run reached step 471,040 with no analysis written.

---

## 2026-09-20 to 09-21: Era 5, semantic57 (reconstructed)

**What:** 57 macro actions, a 267-dim observation, masked PPO.
[../docs/pipelines/semantic57/training.md](../docs/pipelines/semantic57/training.md).

**Result:** `pilot_01` failed ("Collection quotas not met"); `pilot_02` was interrupted.

**Decision:** parked as a comparison arm.

---

## 2026-09-22 to 09-23: Era 6, primitive27 baseline (reconstructed)

**What:** one integrated teacher with weapon, recovery and combat drills, pooled BC, then plain PPO.
Spec: [../configs/legacy/primitive27/structured27_bc_ppo_v1.json](../configs/legacy/primitive27/structured27_bc_ppo_v1.json),
artifacts: [../results/legacy/primitive27/structured27_bc_ppo_v1.artifacts.json](../results/legacy/primitive27/structured27_bc_ppo_v1.artifacts.json).

**Result:** pickup passed the offline gate at 20/33. A 4,096-step PPO pilot gave net stocks 0 and a
damage trade of about -20 in both evaluations. Status: `completed_pilot_not_evidence_of_improvement`.

**Decision:** frozen as the baseline.

---

## 2026-09-23 to 09-27: Era 7, Round A, information and memory (reconstructed)

**What:** three candidates (`mlp`, `recurrent`, `multi` = CNN on RGB plus the structured state)
trained on one aligned dataset. Protocols: [../configs/legacy/round_a/](../configs/legacy/round_a/).
Docs: [../docs/pipelines/round_a/](../docs/pipelines/round_a/).

**Result:**

- 40-epoch BC exists for all three candidates.
- Pilot PPO failed the 30 Hz cadence gate (p95 decision interval 53.7 ms).
- `deployment_v1` mlp: run 1 stopped after the baseline evaluation; run 2 reached 78,578
  interactions, then failed, at 25.7 Hz with 47% of decisions late.
  Summary: [../results/legacy/round_a/deployment_v1_partial_20260924/](../results/legacy/round_a/deployment_v1_partial_20260924/).
- Uncapped profile (09-27): PPO at 50k steps minus BC = **-1.2 net stocks per minute** under both
  logging modes; logging cost about 1.5 ms per decision; `proves_ppo_improvement: false`.

**Reading:** across every era so far, PPO fine-tuning has made the BC policy worse. No document
explains why. This is the main open question carried into the hypothesis.

**Decision (as of 09-27):** run `deployment_uncapped_v2` (100k uncapped screening) from each
candidate's zero-step BC checkpoint, in the order mlp, recurrent, multi.

---

## 2026-11-02: Restructure into the PADP layout (no new results)

**What:** the repository was reorganized like PADP-Extension so that experiments can be stated,
run and summarized precisely. Behavior-preserving: files moved, imports and default paths were
rewritten, no logic changed. Branch `restructure/package-layout`; the state before is tagged
`pre-restructure`.

| Before | After |
|---|---|
| `env.py`, `config.py`, `action_space.py`, `capture_first.py` | `brawl/env/live.py`, `brawl/calibration.py`, `brawl/control/action_space.py`, `brawl/capture_first.py` |
| `feature_extractor/memory/`, `feature_extractor/yolo/extract.py` | `brawl/state/`, `brawl/perception/yolo.py` |
| `capture/`, `control/`, `reward/`, `perception/`, `world_model/`, `algo/`, `analysis/` | the same names under `brawl/` |
| `train/` (52 files, era prefixes) | `brawl/training/` (shared) and `brawl/pipelines/<era>/` (prefixes dropped, e.g. `train/round_a_models.py` is `brawl/pipelines/round_a/models.py`) |
| `hierarchical/`, `evaluate.py`, `feature_extractor/film_extractor.py` | `brawl/pipelines/llc/` |
| `tools/` | `scripts/{ops,llc,round_a,final_agent}/` |
| `experiments/round_a/`, `experiments/baselines/`, `experiments/easy_bot_v0/` | `configs/legacy/{round_a,primitive27,llc/easy_bot_v0}/`; drivers in `scripts/llc/easy_bot_v0/` |
| `train/models/<era>/`, `outputs/` | `runs/legacy/<era>/` (git-ignored) |
| `train/models/weapon_classifier*.pt` | `data/models/` |
| `brawlhalla_gcrl_paper/`, `reports/`, `notebooks/` | `research/paper/`, `research/reports/`, `research/notebooks/` |
| `feature_extractor/yolo/best.*` | `assets/yolo/` |

The complete module map is `brawl.compat.LEGACY_MODULES`.

**Verification:**

- Tests: 834 passed before; 935 passed after (834 plus 101 new tests for the checkpoint shim).
- All 46 entry points launch with `--help`.
- One checkpoint of each kind (Round A mlp, recurrent and multi BC; primitive27; semantic57; LLC BC
  initializer; final agent; weapon classifier) loads from its new location with the same classes,
  observation shape and parameter checksum as before. Without `brawl.compat` the pre-move SB3
  checkpoints fail with `No module named 'train.round_a_models'`, so the shim is required.
- Round A `--init-bc` contract checks pass for all three candidates with
  `configs/legacy/round_a/deployment_uncapped_v2.json`.
- Every relative Markdown link resolves.

**Notes for later analysis:**

- Source hashes recorded in run manifests are not comparable across the restructure: every file
  path changed. `brawl.paths.source_files()` now hashes all of `brawl/` and `scripts/`; before, three
  separate lists covered a subset of folders.
- Run records written before the restructure keep their original paths (`train/models/...`). They
  are records, not lookups; Round A's contracts compare content hashes, not paths.
- The easy_bot_v0 manifest (frozen 2026-08-13) does not verify, and did not before the
  restructure either: `config.json` changed after the freeze (2,747 bytes frozen, 5,555 bytes at
  `pre-restructure`), and the two `recollected` demo archives were re-collected (2,826 samples on
  disk against 7,938 recorded). Every other recorded artifact verifies at its new path.

**Decision:** next, a live test that an agent still runs on the new layout. Then decide together:

- **One flow.** Consolidate the per-frame path into one environment and one observation pass, with
  every feature computed once per frame and one agent route whose variants are configurations.
  Already found: `BrawlDeepEnv` computes a shaped reward every frame that `MatchTrainingEnv`
  discards and recomputes; `GoalMouseTrackerWrapper` is defined twice; each pipeline's `run.py`
  defines its own env wrappers.
- **The research hypothesis** (`research/HYPOTHESIS.md`), then the PADP experiment harness with the
  first pipeline designed for it.

---

## 2026-11-05: Stage 0, game layer gates (T0-T5, first rounds)

**Command:** `python scripts/gate.py t0 ... t5` (three rounds; reports in
[../results/gates/](../results/gates/)).

**Result:**

- **Rounds 1-2 were measurement errors, not game failures.** The screen keeps animating while
  paused (smoke, respawn ring), so "a new frame arrived" and whole-screen differences could not see
  a stepped frame or a key; and pause is a toggle, so each gate's pause press resumed the game left
  paused by the previous gate.
- **The paused toolbar shows the game's frame counter** ("F5 Prev | 1544 | F6 Next"): the ground
  truth for steps. The game renders at 144 FPS (simulation rate not yet measured).
- **T0 (human reads the counter), without confirmation:** with no fight keys changing, 200/200 F6
  and 200/200 F5 applied. With random keys changing every step, only **166/200 F6** applied; F5
  199/200. The game drops F6 presses, mostly when fight keys change on the same step.
- **T2, key gap sweep without confirmation (300 F6 each):** 8, 2, 1, 31, 6 drops at 0, 5, 10, 20,
  35 ms. Drops are partly random, so no gap removes them.
- **T0 with confirmation** (`SteppedClock(counter=...)`: wait for the counter to change, re-press if
  not; 10 ms key gap): **200/200 in all four rows, as read by the human**; the clock re-pressed 15
  dropped presses.
- **T4:** about 50 ms per step and 12 ms per rewound frame (tap 10 ms, before confirmation; to be
  re-measured with it).
- **T5:** the difficulty macro is not reliable: Tab toggles the bot menu and D changes whichever
  row is selected (it changed Set Damage and pushed AI Setting to "Attack Heavy"). Withdrawn;
  difficulty is set by hand until the macro navigates the menu.
- **Bot menu options found:** Set Damage (0-750, current frame), Damage Reset and Position Reset
  (on Brawlhalla's own timing; kept Off), Attack/Dodge/Jump Reaction (kept Off for fairness).

**Reading:** stepping is exact only when every press is confirmed against the frame counter. With
that, the clock and the game cannot disagree about the frame, which is what rewind, paired
evaluation and determinism all rely on.

**Decision:** confirmation is the default (`--no-confirm` to disable); key gap 10 ms. Next: T1
(keys) and T3 (determinism) with confirmation, and T4 re-measured.

---

## 2026-11-05: T3 stops, and a clock that reads the counter's number

**Command:** `python scripts/gate.py t3` (twice), plus a human witness test.

**Result:**

- **Run 1** stopped on the first rewind after 120 random frames: F5 was ignored 6 times at frame
  120, and the counter read 120. Suspected cause: a dropped key release blocking stepping. Fix
  (`8734497`): force-release every fight key on repeated drops.
- **Run 2** stopped after 67 good rewinds: F5 was ignored 6 times. The clock said frame 53, but
  the counter read **51**. A thrown gauntlet was in flight, so a throw was suspected.
- **Witness test** (pause, 10 steps, throw, 25 steps, then F5): rewinding through the throw works,
  and replay is identical. **The counter restarts at 0 on every pause and goes negative** when F5
  goes back past the pause, into the earlier session.

**Reading:** the throw is not the cause. The 2-frame offset shows that twice, a press landed after
the clock had stopped waiting for it, so the retry moved the game a second frame. The final stop
fits a stretch of about 1 s in which the game did not respond, longer than all the retries combined
(each waited 0.15 s). A clock that only sees "the counter changed" cannot detect or undo either
case.

**Decision:**

- The clock reads the counter's **number**:
  - Digit templates are fitted once by `gate.py counter`. It re-pauses (counter 0), steps up 110,
    rewinds to -15, labels every crop, and the human confirms the final number.
  - The clock's `frame` is the game's number.
  - Retries hold the key longer and wait twice as long each time, up to 2 s.
  - Before every press, and after every press, a mismatch is repaired exactly: F5 back, or F6
    forward with the keys recorded for that frame.
- FakeGame gained late presses, and tests cover forward and rewind double landings.

---

## 2026-11-05: T3 passes, accepted by witness; the game layer is done

**Command:** `python scripts/gate.py t3` (three more runs). The report is in
[../results/gates/t3.md](../results/gates/t3.md), and the per-frame data is in
`runs/gates/t3/frames.csv`.

**Result:**

- **Stepping is exact.** With the clock reading the counter's number, every run kept the clock and
  the game on the same frame: 1,200 presses per run, about 55 retried, about 14 double landings
  undone.
- **A retry hazard was removed:** retries released and re-pressed the fight keys. For jumps and
  attacks a re-press is a new press, so retries now press only F6 or F5.
- **One run stopped:** the game ignored F6 for about 6 s with the clock in sync (counter 61). The
  later runs did not stop. The gate now records which window has focus and keeps probing for
  30 s, so a recurrence will show whether the stall clears by itself or the game lost focus.
- **Replays differ from frame 0, and the difference fades.** For example, in replay 3 the
  20-frame block means were 1.458, 0.296, 0.155, 0.071, 0.020 and 0.028%. The human saw the
  red only on the KO animation, the stock icons and a weapon spawn, plus the phantoms. These
  animations run on their own clock and are not rewound. A diverged fighter would make the
  difference persist or grow instead.

**Reading:** the fighters are deterministic under rewind and replay with random keys. Every
difference is cosmetic. The "run of 10 differing frames" rule was wrong, because a KO explosion
is cosmetic but lasts many frames.

**Decision:**

- **New pass rule:** in each replay, the last 20 frames differ by less than T1's cosmetic level
  (0.18%). The run above passes it (last-20 means 0.054, 0.052, 0.022, 0.028, 0.104%).
- **T3 is accepted by witness.** Pixels cannot tell a fighter from an effect, so the hard check
  (fighter positions equal to the pixel) comes with perception.
- **The game layer (T0-T4) is done.** Next is Stage 0, step 2: recording and labelling.

## 2026-11-06: The old detector dataset, audited; the detector is retrained

**Question:** can the old YOLO dataset (Roboflow `temporal-yolo` v15) train or stand in for the
new detector?

**Found** (contact sheets of the valid split):

- 822 unique frames, not about 1,500: train holds 3 augmented copies of each of its 519 frames
  (rotation of up to 15 degrees), and every image is stretched to 960 x 960.
- **Fighter boxes include the held weapon.** A spear doubles the box width.
- **`weapon` boxes only spawn markers.** Lying and thrown weapons are unlabelled.
- Identity comes only from a separate `indicator_self` class.

**Decision:** body-only positions need new boxes on nearly every frame, so new drafts from our
own recordings are labelled instead.

| Set | Sessions | Frames |
|---|---|---|
| Test | 1 Easy, 1 Hard | 150 |
| Train | 2 Easy, 2 Hard | about 400 |

The two sets never share a session. The old detector makes the drafts, and its P1 score is the
baseline. Spawn markers are boxed whole; items are not labelled. Rules are in docs/labelling.md.

### Detector classes, final (same day)

**Classes:**

| Kind | Classes |
|---|---|
| Fighters, by identity and held weapon | `wu_shang_{unarmed,spear,gauntlets}`, `mordex_{unarmed,scythe,gauntlets}` |
| Free weapons | `weapon_spawn`, `weapon_thrown` |
| Off the screen | `offscreen` |

**Rules:**

- Boxes are **body only, with no weapon ever** (gauntlets included), so the position never moves
  with a weapon.
- The held weapon is **a class**, not left to the crops. It decides the opponent's moves and reach,
  and as a class it costs one click per box and is measured by P1.
- The old resnet18 armed/unarmed classifier is retired. Its replacement reads the new 256 px
  crop. Revised later the same day: the armed state is in the labels, and by default it is read by
  a crop classifier, not by the detector's class. The human objected that a body-only box gives
  the detector little of the weapon to classify. P1 compares the two.
- Only the thrower can pick up a thrown weapon (the human's game knowledge), so ownership matters.
  It is tracked from the armed-to-unarmed change and is not labelled.

**Measured:**

- The camera is fixed: the stage region is identical across stills of both sessions.
- The old fighter+weapon boxes reach at most 151 x 174 px. The crop is therefore 256 px centred on
  the body (DESIGN 5.3), not 160 px.

**Recording problem:** the Hard demo (20261106_155225) lost about 60 s in minutes 1-6, at 43-48
rows/s with gaps up to 0.56 s. The game repeated frames too, so the machine was overloaded.

## 2026-11-09: Labels exported; P1 scorer built; old YOLO baseline

**Labels:** Roboflow project `brawlhalla-s04p0` v1, 645 images, 8 classes (`offscreen` unused).

| Split | Frames | Source |
|---|---|---|
| Test | 149 | Nov 6 Easy and Hard sessions |
| Train + valid | 496 | Every still of one 16.5-minute Easy session (20261109_184230) |

The test set shares no session with training. `brawl/sense/truth.py` maps each label back to its
lossless still. A bare row name is resolved by pixels when several recordings share it.

**Gate P1, scorer:** `scripts/gate_p1.py`. A fighter is found when the predicted box of that
identity has its centre inside the labelled box; the error is the distance between centres.
Thresholds are provisional.

**Old YOLO (results/gates/p1_old_yolo.md): FAIL.**

| Fighter | Found | Swaps |
|---|---|---|
| Wu Shang | 145/148 | 0 |
| Mordex | 126/142 | 1 |

- Missed Mordex: far away, overlapped, or behind effects.
- Spawn recall 0.915; thrown weapons 0/9.

Position error by what the fighter holds (median / p95 px):

| Held | Wu Shang | Mordex |
|---|---|---|
| Unarmed | 0.2 / 8.4 | 0.2 / 10.9 |
| Gauntlets | 5.5 / 20.9 | 3.9 / 13.4 |
| Spear / scythe | 17.8 / 26.1 | 16.9 / 22.0 |

**Reading:** the old weapon-inclusive boxes move the position by about 17 px whenever a long weapon
is held, as predicted.

**Caveat:** the unarmed agreement is optimistic, because the labels started from this model's
drafts. Where the draft was already right it was kept, so a model trained on these labels is not
disadvantaged by the old convention.

## 2026-11-10: First trained detectors on P1; the labels disagree with each other

**Trained:** YOLO11s at 1280 px on Kaggle, from the pack. Two variants: identity classes, and the
armed (nine-class) classes. Both FAIL P1:

| | Identity variant | Armed variant |
|---|---|---|
| Fighters found (Wu Shang / Mordex) | 0.986 / 0.944 | 0.959 / 0.958 |
| Centre error p95 | 20-24 px | 20-24 px |
| Held-weapon accuracy | n/a | about 0.62 |
| `weapon_spawn` precision | about 0.66 | about 0.66 |
| `weapon_thrown` precision | about 0.14 | about 0.14 |

**Audit:** crops of every disagreement between the armed model and the test labels.

- **Held weapon:** 104 disagreements. The test fighters left at the draft class `*_unarmed` visibly
  hold spears, gauntlets and scythes, so the model is right in the cases examined.
- **Thrown weapons:** of 28 predictions without a label, the ones examined are real lying gauntlets
  and flying spears that are missing from the labels.
- **Spawn markers:** extra predictions on phantoms and on red swirl effects. These are real model
  errors.

**Boxes follow two conventions.**

| Class | Train labels: median width | Test labels, corrected: median width |
|---|---|---|
| Spear | 139 px | 69 px |
| Scythe | 106 px | 56 px |
| Gauntlets | 76-80 px | 41-47 px |
| Unarmed | about 75 px | about 75 px |

- **Train labels include the held weapon.** The AI-assisted training labels did not follow rule 1.
- **The test labels mix two conventions.** Corrected boxes are body-only, while uncorrected drafts
  keep the old detector's weapon-inclusive box and the `unarmed` class.

**Reading:** P1 cannot be measured until both sets follow one box convention, and the test set is
complete (held classes, thrown weapons). The models already beat the labels on held weapons and
thrown weapons.

### Box rule revised: weapon included, position from a torso point (same day)

**What failed:** body-only boxes cannot be drawn consistently. When a weapon covers the body (the
human, relabelling), the visible body shrinks to the head and the box is a guess.

**New rule:** each fighter label has three parts.

| Part | Rule |
|---|---|
| Box | The whole fighter, including the held weapon |
| Class | Identity + weapon |
| Point | `torso`, the centre of the torso |

- The detector becomes a one-keypoint pose model, so box, class and point come from one pass.
- The weapon is inside the box, so the class is read from the box's own pixels.
- The point gives the position and centres the crop.
- The training labels' boxes already follow this rule. The fix is classes, points, and the test
  set's body-only boxes.

`scripts/to_keypoints.py` converts the detection export into a keypoint import, with points
pre-placed at box centres. Rules: docs/labelling.md.

## 2026-11-10: the keypoint export, and what the box centre costs

Exported the Roboflow keypoint project (BrawlhallaRL v1, YOLOv8 keypoint) to
<BRAWL_DATA>/labelling/export_kp: 396 train, 99 valid, 149 test frames, every box with one visible
point, all points inside their boxes. The `offscreen` class has no label yet.

- **Test:** boxes now include the weapon, as in train (spear median width 134 px test, 137 train).
  The torso is far from the box centre only when the weapon sticks out: spear 17.7 px median
  (p95 34), scythe 19.7 (p95 31); gauntlets 0.2-2.8 (p95 9-12); unarmed 0 (p95 2).
- **Train/valid:** 21 of about 1,700 points moved; the rest sit at the box centre, wrong for spear and
  scythe by the amount above. Those 289 points (train 224, valid 65) must be dragged before training.
- **Old detector, rescored on torso points:** p95 26 px (Wu Shang), 22 px (Mordex); fails the 12 px
  threshold. That is what a box centre costs as a position.
- Code: Box carries the point (`position`), the reader reads pose labels, P1 error is torso to
  torso, the trained detector returns its points, train_detector trains a one-keypoint pose model
  and writes every non-fighter point at its box centre (29 test spawn points had drifted when
  their boxes were resized).
- Roboflow refused to train ("skeleton must have at least one keypoint"): the project's skeleton
  definition, not the labels. Not needed: training is on Kaggle.

## 2026-11-10: gate R1, the HUD reader and the ledger

Built brawl/sense/hud.py and brawl/ledger.py and replayed all four recordings (139,242 frames)
through them (scripts/gate_r1.py, results/gates/r1.md): pass.

- **The HUD slots swap sides** when the lead changes (the higher score is on the left). The reader
  identifies each portrait every frame on a patch under the face that the badges never cover, so
  the earlier workaround (dying until the order could not change) is not needed.
- **The number is a score:** -1 for a lost stock, +2 to the fighter credited, -3 for a
  self-destruct (nobody credited). With that rule the red and cyan badges counted over each
  recording reproduce its final score exactly, and the ledger's KOs equal the red badges
  (Mordex / Wu Shang: 17/3, 6/10, 26/5, 6/1).
- **Damage:** the earlier colour curve (white, yellow, orange, red, darker reds) holds: 10 single
  hits read from the damage overlay match within 1.9, up to 185 damage.
- **Found by the gate and fixed:** a badge dropping in tints both rings for 3-4 frames (12-30 from
  the curve in RGB; clean rings are within 10): the tolerance is now 12 and those frames are
  unreadable. A ring reset counts as a KO only from real damage (5 or more); below that the badge
  reports it. Before the fixes the long Easy session showed 50 Mordex KOs for 26.
- **Limit of the recordings:** JPEG wobbles the dark reds by up to 8 levels, 5 damage above 150
  (stills vs JPEG: 90% within 2, 99% within 4.6). Lossless frames have one exact ring colour, so
  live readings do not have it; totals computed from recordings run slightly high there.
- Unreadable frames 0.5-1.0% (badge drop-ins, swaps, another window captured); skipped without
  loss, since damage readings are absolute.

## 2026-11-10: first pose detector on P1 (yolo11s-pose, 960 px, export v3)

Kaggle, 101 epochs (best at 61). The 1280 px job died at start: both jobs built the same dataset
folder (fixed: one folder per run). results/gates/p1_armed_yolo11s-pose_960.md: fail.

- Found: Wu Shang 147/148, Mordex 134/142; no swaps.
- **The torso point works but is not yet tight:** spear 7.3 px median to the labelled torso (18.2
  to the box centre), scythe 7.7 (15.2); p95 16-19 px against 12. Gauntlets: the model predicts the
  box centre, as the train labels say (p95 7-11 px to it), while the test points were moved.
- **Held weapon 0.91 / 0.87.** 31 disagreements, viewed: most are model errors, unarmed against
  gauntlets (17 of them Mordex), a fighter holding an item (bomb, mine), a spear just thrown inside the box;
  two or three are doubtful labels.
- **weapon_thrown precision 0.31:** 23 of the 24 unmatched detections are horns and mines lying on the
  stage. The train labels box them as `weapon_thrown` (most of the 88); the test labels follow
  rule 11 and do not. A label inconsistency, not a model fault.
- weapon_spawn: recall 1.0; 9 unmatched (smoke, phantoms, an explosion, twice the red swirl).

## 2026-11-11: export v5 (943 frames), audited

Train 695 (Easy 396; Hard 100, Extreme 99, Chosen 100 from 2026-11-10), valid 99 (Easy only), test
149 (unchanged). Every label has its point, inside its box; every image resolves to its still.

- All 84 train/valid `weapon_thrown` boxes viewed: 13 are items (5 mines, 4 horn boxes on 3 horns,
  2 spike balls, 1 red blob), all in the Hard and Chosen sessions, plus 3 doubtful boxes on a
  held spear or the fighter. The Easy part and the Extreme session are clean. The same fault gave
  thrown precision 0.31 on 2026-11-10, so they go before training.
- Validation covered one session, and the best epoch is chosen on it: train_detector now gives
  every session without validation frames every fifth of its own (train 636, valid 158).
- Gauntlet torso points: moved on 121 of 497 train boxes (new sessions), still at the centre on
  the Easy session's.

## 2026-11-11: pose detector on export v6 (1280 and 960 px), P1

Five frames with two boxes of one fighter fixed in the local export (3 train, 2 test; listed in
export_kp_final/LOCAL_FIXES.txt, not yet in Roboflow). Kaggle, both runs on the same 636 / 158
frames: 1280 px stopped at epoch 83 (best 43), 960 px ran 150 (best 135). The 2026-11-10 model is
kept as runs/detector/armed_yolo11s-pose_960_v3labels.

| Check (pass at) | 1280 px | 960 px |
|---|---|---|
| Found, Wu Shang / Mordex (0.98) | 1.000 / 0.958 | 0.993 / 0.951 |
| Swaps, false fighter boxes (0.01) | 0 / 0 | 0 / 0 |
| Torso error p95, px (12) | 16.9 / 15.1 (median 4.7 / 4.0) | 16.1 / 13.5 (median 5.0 / 3.5) |
| Held weapon (0.95) | 0.953 / 0.949 | 0.939 / 0.926 |
| weapon_spawn recall / precision (0.95) | 0.986 / 0.890 | 1.000 / 0.925 |
| weapon_thrown recall / precision (0.80) | 0.867 / 0.722 | 0.800 / 0.750 |
| Time per frame, PyTorch, RTX 3050 Ti | 33.4 ms | 20.7 ms |

Both fail. Every failing item of the 1280 px run was viewed:

- **Mordex missed, 6 frames:** 4 are hidden (behind a hit flash or behind Wu Shang, or tinted orange
  while hit); 1 has a box at 0.15, under the 0.25 threshold; 1 is a label on the KO bubble at the
  top of the screen (not a fighter by the 2026-11-10 offscreen decision).
- **Torso error:** the three largest Wu Shang errors (31, 21, 21 px) are spear labels whose point
  was left at the box centre; the model's point is on the torso. Six test spear points are still at
  the centre, and 15 of 249 train spear and 25 of 159 train scythe points. In the other large
  errors the two points are 15-25 px apart and neither is clearly wrong: the labels themselves
  place the torso to within about 15 px in attack poses.
- **Held weapon:** 15 disagreements, unarmed against gauntlets in 10. On the validation frames
  (which include Hard, Extreme and Chosen) the 1280 px model reads 0.928 / 0.870: the test value sits
  on the threshold and does not hold on harder sessions.
- **weapon_spawn, 9 unmatched:** 3 second boxes when Mordex stands on a spawn, a thrown spear
  lying across a spawn, smoke, a background bird (both under 0.4), and the red swirl (0.63). The
  swirl is labelled as a spawn in test frame 17989 and not in 29235: the rule is undecided.
- **weapon_thrown, 5 unmatched:** a spear flying at the right edge that has no label (the model is
  right), a raised gauntlet, a spear leaving the hand, two under 0.35.
- **Thresholds:** fighter boxes at 0.10 instead of 0.25 add no false fighter in 307 frames and
  recover Mordex on validation (0.960 to 0.980) and test (0.958 to 0.965). Weapon boxes at 0.50:
  spawn 0.946 / 0.972 and thrown 0.867 / 0.867 on test, spawn 0.937 / 0.963 and thrown 0.636 / 0.700
  on validation (11 thrown labels there).

## 2026-11-11: test labels corrected; the detector over consecutive frames

**Test labels (5 frames, by hand on gridded crops; listed in export_kp_final/LOCAL_FIXES.txt, not
yet in Roboflow):** the Mordex box on the KO bubble deleted (155225 row 4931); three spear points
moved from the box centre to the torso (155225 rows 28278, 2223, 28755); the thrown spear at the
right edge added (151844 row 20387). Rescored, both still fail: 1280 px found 1.000 / 0.965, torso
p95 15.3 / 15.1, held 0.953 / 0.949, spawn 0.986 / 0.890, thrown 0.875 / 0.778; 960 px found
0.993 / 0.958, torso p95 13.6 / 13.5, held 0.939 / 0.926, spawn 1.000 / 0.925, thrown 0.812 / 0.812.

**More epochs will not help:** the 1280 px run stopped by itself 40 epochs after its best (43), and
the 960 px run's best is at 135 of 150 with validation flat.

**Consecutive frames (960 px model, 3,000 frames, rows 6000-8999 of the Hard session 155225,
recording frames scaled up to 1920 x 1080):**

- The JPEG frames stand in for lossless ones: on 60 rows that have both, the same fighters are
  found (56 / 57 against 57 / 57), the class agrees on all but one each, and the torso points are
  0.5-0.6 px apart at the median (p95 3-3.5).
- Wu Shang is detected in 96.4% of frames, Mordex in 94.9%. The misses are short: Wu Shang 9 gaps
  of 8 frames or fewer (19 frames) and one absence of 90; Mordex 29 gaps of 8 or fewer (64 frames),
  two of 9-30 and one longer (88 frames together).
- Carrying the last position and velocity for up to 8 frames, and accepting a box of 0.05-0.25
  when it lies within 60 px of where the fighter should be, leaves Wu Shang without a position in
  81 frames and Mordex in 61 (2.7% and 2.0%): the long absences, to be checked against KOs.
- Held weapon changes 79 (Wu Shang) and 124 (Mordex) times frame to frame; 10 and 25 after a
  5-of-7 vote. The true number is not labelled.
- Between consecutive detections a fighter moves 4 px at the median and 52-62 px at p99; the
  largest steps are 330 and 709 px, which no fighter makes in one frame: single wrong boxes that a
  distance check against the carried position would reject.

## 2026-11-11: TensorRT engine (960 px) and the tracker on P1

**Engine.** scripts/export_detector.py builds runs/detector/armed_yolo11s-pose_960/weights/best.engine:
half precision, fixed 544 x 960 input, TensorRT 10.13 (build 8 minutes). Installed without
dependencies so that nothing existing changed: tensorrt_cu11 10.13.0.35 with its libs and bindings,
nvidia-cuda-runtime-cu11 11.8.89, onnx 1.18.0 (onnx 1.17 has no build for Python 3.13 and fails to
compile; a plain install would have raised sympy above torch's pin).

- Time per frame on 80 test stills, RTX 3050 Ti: 11.3 ms median (p95 13.4, max 15.1) against 23.9
  (p95 40.9, max 80.9) for the .pt. Of the 11 ms, the network is 4.3; preparing the image on the
  CPU is 4.0 and the box filtering 1.8.
- Same answers: on 151 fighters the class never differs and the torso points are 0.06 px apart at
  the median (4.7 at most); P1 on single frames is unchanged (results/gates/p1_armed_yolo11s-pose_960_trt.md).

**Tracker.** brawl/sense/track.py (constants set once from the 3,000-frame run above, not tuned on
test scores). `gate_p1.py --track 30` lets it follow the 30 recording frames before each test
frame, then scores its fighters on the test labels (results/gates/p1_armed_yolo11s-pose_960_trt_track30.md):

| Check (pass at) | Single frames | Tracked |
|---|---|---|
| Found, Wu Shang / Mordex (0.98) | 0.993 / 0.958 | 1.000 / 0.986 |
| Swaps, false Mordex (0.01) | 0 / 0 | 0.007 / 0.007 (one frame each) |
| Torso error p95, px (12) | 13.6 / 13.6 | 13.6 / 15.3 |
| Held weapon (0.95) | 0.939 / 0.926 | 0.959 / 0.950 |

- Four fighters on test frames were carried without a box. Weapons are untouched by the tracker
  (spawn precision 0.925 still fails).
- The swap (151844 row 27707): Mordex is half transparent in a dodge; a 0.17 box around him was
  taken and its point fell inside Wu Shang's box. The miss (151844 row 34067): the detector puts
  Mordex at 0.93 on the hit effect while he is flying off to the left. The false Mordex is not
  examined yet.
- Mordex held weapon is exactly on the threshold (133 of 140): passing, not robust.

## 2026-11-11: P1 passes with the tracker

Three changes, then `gate_p1.py best.engine --track 30` passes every check
(results/gates/p1_armed_yolo11s-pose_960_trt_track30.md):

- **Torso threshold 20 px** instead of 12 (reviewed with the user): hand-placed points agree to about
  15 px in attack poses. Measured p95 13.6 / 15.3.
- **Spawn markers in the tracker:** one box per marker, reported when seen in 2 of the last 3 frames.
  A first version also carried a marker for 3 frames without a box; 5 of its 10 false markers were
  carried ones (the weapon just picked up, or expired in a puff of smoke), so nothing is carried.
- **The red swirl is `weapon_spawn`:** 60 of the 470 train spawn boxes are the swirl, fireball or
  falling flame before the marker, and the model reads the swirl at 0.94. Two test frames had it
  unlabelled (155225 rows 25761, 29235): added. Spawn 0.974 / 0.974.

Weapon score 0.25, 0.4 or 0.5, compared on the validation split (99 Easy frames) with the tracker:
spawn recall 0.951 / 0.939 / 0.939 at precision 0.963, so 0.25 stays.

**Known weak spots (same run, validation split):** Mordex held weapon 0.764 (Wu Shang 0.959);
thrown weapons 0.40 / 0.67 on 5 labels. On the test frames Mordex held weapon is 0.950, exactly the
threshold. The opponent's weapon is the least reliable feature in the observation.

## 2026-11-11: action list and stage geometry

- **brawl/actions.py:** MultiDiscrete([9, 6]) as a key plan per decision, mirroring, and recorded
  keys read back as actions. On the 57 minutes of human play recorded so far (68,653 decisions at
  k = 3): 53 of the 54 actions are used; 5.3% of the 7,002 button presses do not fit (a second
  button inside one decision); the direction changes inside 14% of decisions (the one held longest
  is kept); 34% of decisions are no keys at all.
- **Charged heavies:** 29% of heavy presses are held 10 frames or more (median 7, p90 22); light
  presses 9%. A button pressed for one frame cannot charge a signature. Open decision for A1.
- **brawl/stage.py:** ledges at x 605 and 1305 (grass edge in the median of 40 stills; grounded
  torsos span 635-1292), feet at y 617 (571 standing fighters), torso 35 px above.
- Decision (user): the policy runs on the CPU, the detector on the GPU, as measured before with SB3.

## 2026-11-11: the chain from a frame to a cloned policy

Written and tested without the game (41 tests in the new modules):

- **brawl/sense/perception.py:** one call per decision frame gives a `Sense` (tracked fighters,
  weapons, damage, events). The environment and the demonstration reader both use it. Half-size
  recording frames go to the detector as they are; boxes are scaled to 1920 x 1080.
- **brawl/observation.py:** 99 entries, each named in `FEATURES`: 15 per fighter (position,
  velocity, present, seen, held weapon, grounded/offstage/below, ledge distances, damage), the
  opponent relative to the agent, the nearest spawn marker and thrown weapon, the last 4 actions.
- **brawl/actions.py:** a button is now held for its whole decision and stays held while chosen
  again (charged signatures); pressing it twice needs one decision in between.
- **brawl/brawlenv.py:** the step function of DESIGN section 4 (named so because the legacy
  `brawl/env/` folder still exists). Counts its calls for gate L1.
- **scripts/prepare_demos.py:** a recording becomes decisions_k3.npz (observation, action, events
  per decision). About 30 ms per decision, so an hour of play takes about 10 minutes.
- **brawl/learn.py, scripts/train_bc.py:** behaviour cloning on winning stretches (a recording is
  cut at each KO; stretches ending in Wu Shang's KO and the unfinished last one are dropped) into
  SB3's actor-critic MLP. Whole stretches are held out. The report scores the decisions where the
  human changes action apart, because the action history in the observation makes "repeat the
  previous action" right most of the time.
- **scripts/play.py:** a policy plays the live game (realtime or stepped) and reports stocks,
  damage and the time per decision against k / 60 s.
- **scripts/gate.py a1:** each of the 54 actions played twice from the same frame and compared
  with doing nothing; plus charged against tapped heavy, and two jumps against one.

Measured offline: the policy takes 0.4 ms per decision on one CPU core (p99 1.6 ms); the detector
11 ms on the GPU. The first converted recording reproduces gate R1's count (17 KOs scored, 3 lost).

**The tracker at one frame in three** (what the environment gives it at k = 3; `gate_p1.py --track
10 --step 3`): found 1.000 / 0.972, 2 Mordex swaps (0.014), held 0.946 / 0.964, spawn 0.961 / 0.948.
Four checks miss their threshold by one or two frames of 149; with every frame they pass. Not
tuned on the test frames. If it matters in play, perceive every frame and decide every third.

## 2026-11-11: dry run of behaviour cloning on the existing recordings

Six recordings prepared (64,594 decisions, 54 minutes; the one with the damage overlay skipped),
14 ms per decision. KOs per recording equal gate R1's. Winning stretches: 37,457 decisions (31
minutes), of which 27,670 are from the two Easy sessions; against Hard, Extreme and Chosen the
human won 15 stocks and lost 31, so little of that play is kept.

`train_bc.py --name dryrun_all_levels` (results/bc/dryrun_all_levels.md), held-out stretches:

| Measure | Held out |
|---|---|
| Loss | 1.037 (flat from about epoch 45) |
| Both right, most likely action | 0.704 |
| Baseline: repeat the previous action | 0.702 |
| Both right when the human changes action | 0.155 |
| New button presses predicted | 0.028 |

The pipeline works end to end, and the result is what 31 minutes can give: the policy's most
likely action is to keep doing what it was doing, and it does not yet predict when the human
starts something new. Its sampled play has the human's rate of presses, not yet their timing.
Whether it wins anything is for `scripts/play.py` to say. What should move these numbers: more
winning play against one level (2-3 hours against Medium), and later what the opponent is doing
(the fighter crops of DESIGN 5.3), which the vector does not carry.

## 2026-11-11: first live runs (brawl312), and cloning recipes compared

**Environment.** Everything runs in the venv C:\venvs\brawl312 (Python 3.12, torch 2.13 cu126,
TensorRT 11.2). The engine built earlier with the system Python (TensorRT 10.13) did not load
there; rebuilt in brawl312 (P1 with the tracker passes; 14.3 ms per frame offline).

**Live play, realtime, 5 minutes each (results/play/):**

| Policy | Bot | Stocks taken - lost | Damage dealt / taken |
|---|---|---|---|
| dryrun_all_levels (31 min of winning play, all levels) | Easy | 4 - 3 | 524 / 408 |
| medium_v1 (9.7 min of winning play against Medium) | Medium | 1 - 4 | 346 / 591 |
| the human, for scale (session 20261111_183313, 11.4 min) | Medium | 17 - 4 | 3,219 / 703 |

- Gate L1 holds in the game: advance, grab, perceive and build each ran once per step (6,004 steps,
  plus one at reset).
- Gate D1 holds: perception + policy 25-27 ms median, 33-34 ms p95, of 50 ms; 0.12-0.13% of
  decisions over; the policy itself 1 ms on the CPU. A third of the frames have their keys set
  late: that is the 26 ms, so an action starts one to two frames after the frame it answers.
- Fighters present 97-99% of decisions.

**Gate A1, first run: FAIL as written, mostly the gate's fault.** Its "does something" threshold
(5.7% of a 560 x 440 window) was above what a move changes (1-3%). By the strips and the torso
path the moves work: walking 16-22 px, jump 112 px up, dash 176 px, lights, the tapped heavy, the
dodge; a heavy held for 6 decisions is still charging where the tapped one has struck. Open: a
second jump one decision after the first added no height; walking differed by 7-10 px between two
plays (the bot stood on Wu Shang at the start). The gate now compares a 300 x 300 window and the
torso path, allows the detector's 12 px, and tests the second jump at two delays over 12 decisions.
To be run again.

**Cloning recipes, held-out stretches, 3 seeds each (all winning play, 49,046 decisions, unless said):**

| Recipe | Loss when the human changes action | Loss | Changed right | New presses |
|---|---|---|---|---|
| Winning stretches, action history (current) | 2.56 +- 0.04 | 1.04 | 0.168 | 0.108 |
| No action history | 2.81 +- 0.05 | 2.15 | 0.222 | 0.110 |
| All stretches (64,594 decisions) | 2.46 +- 0.04 | 1.05 | 0.177 | 0.069 |
| No history, 4 stacked observations | 2.80 +- 0.06 | 2.12 | 0.232 | 0.126 |
| History, 4 stacked observations | 2.53 +- 0.07 | 1.03 | 0.169 | 0.114 |
| Network 512 x 512 x 512 | 2.53 +- 0.04 | 1.03 | 0.176 | 0.126 |

(Choosing among the 54 actions at random gives 3.99.) No recipe is clearly better than the
current one: stacking and a bigger network are within noise, dropping the action history loses,
and the only gain is from more decisions (all stretches), which is the lever more recordings pull
without copying lost stocks. The recipe stays; what is missing is data and what the opponent is
doing.

## 2026-11-11: one hour against Medium; the learning curve; vision probes set up

**Recordings.** Six Medium sessions, 69.7 minutes (83,626 decisions): stocks 106 won, 23 lost;
damage dealt 19,389, taken 4,517. Winning stretches: 72,806 decisions (60.7 minutes). Recordings
take about 18 GB an hour (3.9 GB of half-size JPEG frames and 0.7 GB of lossless stills per 15
minutes); 14 GB are free, so another hour does not fit as recorded today. The stills are kept for
future labelling (decision of the user).

**Learning curve** (winning stretches, 3 held-out splits of whole stretches):

| Minutes of winning play | Held-out loss | Loss when the human changes action | Changes right | New presses |
|---|---|---|---|---|
| 5.9 | 1.163 ± 0.020 | 2.657 ± 0.019 | 0.131 | 0.039 |
| 11.7 | 1.093 ± 0.026 | 2.544 ± 0.036 | 0.155 | 0.063 |
| 21.6 | 1.038 ± 0.027 | 2.441 ± 0.047 | 0.175 | 0.083 |
| 31.7 | 1.008 ± 0.021 | 2.358 ± 0.027 | 0.195 | 0.110 |
| 42.9 | 0.994 ± 0.019 | 2.326 ± 0.015 | 0.195 | 0.123 |

Each doubling lowers the loss when the human changes action by about 0.11; the share of changes
predicted right has stopped rising between 32 and 43 minutes. More of the same data helps slowly;
the user's own observation (2026-11-11): from positions alone it is hard to defend or to see
Mordex's next move coming. That is the question for the vision probes.

**medium_v2** (all 60.7 minutes, results/bc/medium_v2.md): held out, changes right 0.194 (medium_v1
0.170), new presses 0.133 (0.114), loss 0.989. Not yet played in the game.

**Vision probes set up (gate S2).** brawl/vision.py cuts a 256 x 256 screen-pixel window around a
torso point, grey, at 96 px; the same function serves recorded and live frames.
scripts/prepare_crops.py cuts both fighters, at each decision's frame and the frame before
(1.5 GB per hour, memory-mapped: the laptop has 8 GB of RAM). scripts/probe_vision.py trains the
same network on each candidate input and scores, on held-out sessions: hit within 0.5 s, Mordex's
position 0.25 and 0.5 s ahead against constant velocity, and the human's action where it changes.
Candidates: vector alone; Mordex 1 frame; both 1 frame; both 2 frames 1/60 s apart; both 4
decisions (0.2 s); the same at 64 px; crops alone.

## 2026-11-11: vision probes (gate S2, first round)

`scripts/probe_vision.py --epochs 15`, six Medium sessions held out two at a time (3 folds, one
seed each), results/probes/vision.md. Crops on the GPU (1.6 GB) for the decision-frame
candidates; the two-frame candidate (3.1 GB) read from disk, since Windows spills GPU memory
beyond 4 GB into RAM. About 50 minutes in all.

Difference from the vector alone on the same held-out sessions (mean, then folds 0 / 1 / 2;
negative is better except for AUC):

| Input | Hit within 0.5 s: AUC | Hit log loss (x1000) | Mordex in 0.25 s (px) | Mordex in 0.5 s (px) | Action NLL when the human changes |
|---|---|---|---|---|---|
| Mordex, 1 frame | +0.006 (+0.018, +0.003, -0.003) | +0.0 | -6.6 (-5.8, -7.2, -6.8) | -6.9 (-5.1, -7.8, -7.7) | +0.039 (+0.029, +0.031, +0.057) |
| both, 1 frame | +0.030 (+0.041, +0.033, +0.016) | -1.9 | -7.6 (-8.0, -7.7, -7.0) | -8.5 (-8.6, -8.4, -8.4) | +0.013 (+0.019, +0.016, +0.004) |
| both, 2 frames 1/60 s apart | +0.030 (+0.030, +0.036, +0.025) | -2.4 | -9.0 (-8.6, -8.2, -10.1) | -10.3 (-9.8, -9.0, -12.1) | -0.013 (+0.014, -0.018, -0.035) |
| both, 4 decisions (0.2 s) | +0.018 (+0.028, +0.015, +0.011) | -0.3 | -9.0 (-9.3, -7.4, -10.2) | -9.5 (-10.3, -7.5, -10.9) | -0.021 (-0.004, -0.002, -0.058) |
| both, 4 decisions, 64 px | +0.003 (+0.015, +0.005, -0.011) | +0.5 | -6.6 (-7.8, -5.3, -6.7) | -6.7 (-8.2, -4.6, -7.3) | +0.025 (+0.035, +0.007, +0.033) |
| crops only, both, 4 decisions | +0.025 (+0.033, +0.022, +0.021) | -2.4 | +14.1 | +45.2 | +0.216 |

The vector alone: hit AUC 0.671, log loss 0.1962 (always saying the 5.0% rate: 0.1978); Mordex
missed by 81 px at 0.25 s and 140 px at 0.5 s (constant velocity: 117 and 246); action NLL when
the human changes 2.615.

- **Pictures carry what the vector lacks, in every fold:** Mordex's position 0.25-0.5 s ahead is
  7-10 px (6-9%) closer, and whether Wu Shang is about to be hit is better ranked (AUC +0.02 to
  +0.03), with both fighters in view. Seeing Mordex alone helps the position but not the hit:
  Wu Shang's own state matters for defence.
- **They do not yet help imitation:** the action at the human's changes is within the spread
  (-0.06 to +0.06) for every crop input. With an hour of play the network does not learn the
  human's reactions from pictures; the gain shows where the target is the game, not the human.
- **Time:** two frames 1/60 s apart and four decisions are within noise of one frame on the hit,
  and 1-2 px better on the position. 64 px loses most of the gain: 96 stays.
- **Crops alone** rank hits as well as the hybrid but lose 45 px on the position and 0.22 on the
  actions: the vector stays.
- Every candidate used all 15 epochs (none stopped early): all are undertrained, the vector too.
- Absolute level: an AUC of 0.70 means being hit is still mostly unpredictable half a second ahead
  from these inputs.

**Decision (proposed):** the policy's input is the vector plus both fighters' crops, one frame,
96 px (1.1 ms on the CPU). The probe heads (hit within 0.5 s, Mordex's motion) become the image
branch's auxiliary loss (DESIGN 5.3), since that is where pictures help. Recordings keep only the
decision frames from now on (`record_demos.py --frame-every 3`, the new default): about 6 GB an
hour instead of 18. The frame-pair input is then unavailable for new recordings; it was within
noise of one frame.

## 2026-11-11: an outside critique of the observation design, checked; the memory probe

The user had another model review DESIGN section 5. Its claims, checked against the data:

- **Right, and measured here: memory.** Four stacked decisions span 9 frames (0.15 s), not 12 (DESIGN
  corrected), and neither they nor the last 4 actions cover a dodge cooldown (60 frames on the
  ground by patch 2.71; a community guide gives 163 in the air, unconfirmed) or the jumps used.
  `scripts/probe_memory.py` (results/probes/memory.md), vector only, 3 folds x 2 seeds, trained
  until early stopping; difference from the vector in the same fold and seed:

  | Input | Hit within 0.5 s: AUC | Mordex in 0.5 s (px) | Action NLL when the human changes |
  |---|---|---|---|
  | vector | 0.712 | 133.5 | 2.426 |
  | + 13 timers | +0.024 ± 0.014 (all 6 runs up) | -1.4 ± 0.9 | -0.033 ± 0.032 |
  | GRU over the session | -0.005 ± 0.014 | -1.6 ± 0.9 | -0.072 ± 0.046 |
  | timers + GRU | +0.013 ± 0.020 | -2.8 ± 0.9 | -0.113 ± 0.040 |

  Timers (decisions since each own button press, since ground, hit and respawn for both fighters,
  jumps since the ground) help the hit prediction; the recurrent layer helps the imitation; together
  they give the largest imitation gain measured so far, larger than any crop input gave.
- **Correction to the vision probe above.** Its `vector` baseline stopped at 15 epochs: AUC 0.671,
  140.2 px, 2.615. Trained until early stopping here, the vector alone reaches 0.712, 133.5 px and
  2.426: as good as vector + crops did at 15 epochs (0.701, 131.7 px, 2.628). So that run shows the
  hybrid learning faster, not yet knowing more. Whether crops add to a converged vector is open; the
  probe must be rerun until early stopping, with the timers in the vector.
- **Right in principle, already true in the data:** the camera must be the Fixed mode (Classic and
  Action pan and zoom). In all recordings the stage is in one place (571 standing fighters with feet
  at y 613-621, the grass edge in the median of 40 stills), but nothing checks it at start.
- **Already how the code works:** a missed fighter is not called offstage: `present` and `seen` are
  entries of the vector, and the position is the last one seen (the sentence in DESIGN 5.2 is older
  than brawl/sense/track.py). Random shifts are the same for every picture of a stack.
- **Already gated:** the HUD ring as damage (R1: 10 overlay hits within 1.9, KOs exact).
- **Checked on the labels:** the 256 px window around the torso holds the fighter and its weapon in
  1,797 of 1,800 labelled boxes (farthest reach 126 px sideways, 144 px up with the spear; 3 boxes
  exceed 128). Labels do not include swing trails.
- **Right about the wording, not about what was built:** DESIGN 5.3 words the auxiliary labels as
  "a hitbox appeared", which needs overlays. The probe targets used are free without them: damage
  taken (HUD) and Mordex's displacement (tracker).
- **Not adopted now:** supervised combat-phase labels (startup, active, recovery). They need the
  overlays; the route exists only in stepped mode (replay the same keys with overlays on), and the
  free targets have not been shown to fail.
- **Noted for later:** a strong hit can disarm (the wiki has a Disarming article), so armed to
  unarmed does not prove a throw; the thrown-weapon owner tracking, when built, needs the projectile.

## 2026-11-11: timers in the observation; the probes made ready for a long run

**Timers (decision of the user).** brawl/observation.py: 13 entries after the 99 (112 in all), kept
by one `Timers` class that the builder updates each decision. Files prepared earlier got them added
in place (`prepare_demos.py`, no detector). Check on session 20261111_201232, prepared again from
its frames: the 99 older entries are identical to the stored ones in all 9,090 decisions (the
detector, tracker and HUD reader are repeatable), and the timers added in place equal the ones
built live. Policies cloned before (99 entries) no longer fit the environment.

**medium_v3** (winning Medium play, with timers; results/bc/medium_v3.md), held out against
medium_v2: loss 0.976 (0.989), changes right 0.208 (0.194), new presses 0.145 (0.133). One seed.

**Camera:** Fixed mode, confirmed by the user for every recording; written into the recorder's
instructions and each new recording's meta.

**Probes rebuilt for the long run** (scripts/probe_vision.py; scripts/probe_memory.py is removed, its
arms are now `vector, no timers`, `vector`, `vector, GRU` here). One loop for every input: windows of
128 decisions, 8 per batch, trained until the last 12% of the training sessions stops improving
(12 epochs without gain), 2 seeds per fold, runs recorded as they finish. Inputs: the three vector
arms; both crops 1 frame; both crops 4 decisions; both crops 1 frame with the GRU. Dropped after
round 1: Mordex alone, 64 px, crops without the vector, and the frame pair (new recordings keep the
decision frames only). Crops are now one frame per decision (0.8 GB per hour).
scripts/pack_probes.py packs the code with each session's decisions and crops (1.5 GB for the six
sessions) and prints the Kaggle cell. Round 1 is kept as results/probes/vision_round1.md.

Disk: Windows' page file grew to 19.7 GB under memory pressure (8 GB of RAM); 9.8 GB free.

## 2026-11-12: the long probe run bounded in time; a pair-region crop discussed

- Medium recordings: 9 sessions, 110 minutes (three new ones with decision frames only).
- scripts/probe_vision.py: `--hours` stops cleanly (no new run in the last 5 minutes; a run in
  progress keeps its best epoch and is marked), `--part i/n` shares the vector runs and the picture
  runs evenly between processes, and runs go seed by seed so that a stop leaves every input with
  the same runs. The Kaggle cell gives each GPU 9.5 h and kills at 10.25 h (Kaggle ends at 12 h).
- Idea of the user, not built: one crop of the smallest region holding both fighters. Measured on
  the 116,631 prepared Medium decisions: the opponent is already inside the agent's 256 px window
  in 43% of decisions with both present, and in 70% of those followed by a hit within 0.5 s. The
  smallest square holding both (80 px margin) has a side of 210 / 311 / 584 / 969 px at the 10th /
  50th / 90th / 99th percentile; at 96 px a fighter is then 34 / 23 / 12 / 7 px tall (28 in the
  fixed crop). The pair is more than 3 times wider than tall in 23% of decisions.

## 2026-11-12: medium_v4; the noise-floor tool

- **medium_v2 played Medium** (2026-11-11, 5 minutes, 99 entries, 61 minutes of winning play): stocks 3 - 2,
  damage 558 / 552 (medium_v1, from 10 minutes: 1 - 4, 346 / 591). One block each: no noise floor yet.
- **medium_v4:** cloned from all 9 Medium sessions with timers (166 stocks won, 36 lost; 113,889
  decisions kept, 95 of 110 minutes). Held out: loss 0.898, changes right 0.224, new presses
  0.112 (results/bc/medium_v4.md). Not comparable with medium_v3's held-out numbers: other stretches.
- **scripts/play.py --blocks n:** n blocks of `--minutes` played without stopping, each reported, with
  the spread between blocks, the standard error, and the difference in reward per minute that two
  policies need to differ beyond chance; virtual matches (first to 3 stocks) over the run; a JSON
  record in runs/play/. The loop is tested against a scripted environment, not yet in the game.

## 2026-11-12: committed; the RL loop (Stage 1 baseline)

Work since ce30e6a committed in five commits on restructure/package-layout (not pushed).

**Config system** (brawl/config.py, from PADP's config_io): configs/base.yaml holds every default;
an experiment file adds overrides, a grid and seeds; a key that base.yaml lacks is rejected.
configs/experiments/s1_ppo.yaml is Stage 1 (H1, H2): regime (realtime, oracle) x start (scratch,
bc), 5 seeds, one hour of game time each. configs/experiments/smoke.yaml: five minutes from medium_v4.

**scripts/train.py + brawl/learn.py `train_ppo`:** Stable-Baselines3 PPO on BrawlEnv, on the CPU,
with the cloned policy's weights when `train.init` is set (same network; the value head starts
untrained). Settings: rollouts of 1,200 decisions (one minute), 5 epochs, batch 300, learning rate
3e-4, gamma 0.995: the standard recipe, on purpose. What a live game needed:

- **The update stalls the agent, the game goes on.** Before the update the keys are released;
  after it `BrawlEnv.resume` counts the frames that passed (`RealtimeClock.resync`), follows the
  fighters afresh, advances the timers, and observes again. Damage and stocks lost during the
  stall are logged apart and are no action's reward. The stall's length is logged per rollout.
- **No windows:** the task continues; PPO bootstraps from the value of the last observation.
- **Focus:** every 20 decisions the window with the keyboard is checked; if it is not the game,
  the keys are released and training waits.
- **F10** stops cleanly (policy_stopped.zip). A run is never resumed: a stopped one is kept as
  seed<k>_attempt<n>.
- progress.csv per rollout: steps, game frames, wall time, stocks, damage, reward, the stall and
  its cost, late frames, perception time, decisions over the budget, pauses. Checkpoints load in
  scripts/play.py.

Not built yet: the oracle arm's use of rewind, evaluation of checkpoints over blocks, sweep and
analyze scripts, and the asynchronous learner of H3.

## 2026-11-12: noise floor (gate N1), first measurement

`play.py runs/bc/medium_v4/policy.zip --difficulty medium --minutes 5 --blocks 6`
(results/play/medium_v4_medium_realtime_x6.md): 30 minutes, real time, sampled actions.

- **medium_v4 is level with the Medium bot:** stocks 16 - 17, damage 3,210 / 3,191, virtual matches
  3 - 4, reward per minute -0.03. The human in the recordings it was cloned from: 166 - 36 stocks
  in 110 minutes, about +3 reward per minute.
- **Between 5-minute blocks** the reward per minute has a spread of 0.58 (blocks: -1.19, 0.24, 0.32,
  0.17, 0.32, -0.02); net stocks per minute 0.29. Standard error of the 30-minute mean: 0.24.
- **So:** two policies played 30 minutes each are told apart when they differ by more than about
  0.67 reward per minute; one 5-minute block (as used for medium_v1 and medium_v2) tells nothing
  below about 1.6. Halving the detectable difference takes four times the play.
- Timing over 36,014 decisions: 26.5 ms median, 32.8 ms p95, 0.17% over the budget, one decision of
  1.2 s. Each part ran once per step.
- Six consecutive blocks of one run are not six independent runs; seeds and sessions on other days
  are still to be added to this estimate.

The PPO trainer's test passes (47 tests in the new modules); not yet run in the game.

## 2026-11-12: gate A1, second run; the trainer's first run in the game

**Gate A1** (bot away from Wu Shang, Wu Shang unarmed and facing left). The run printed its whole
table, then crashed while saving the strips (sequences of 12 decisions are wider than those of 6:
my fault), so no report file was written. Read from the printed table:

- **Same both times:** 54 of 58 sequences by both measures. The other four are the dashes to the
  facing side (left, down, up-left, down-left + dodge: 170 px): identical by pixels (2.0-2.2%
  against 4.3%), with the detected torso 14-18 px apart between the two plays, a tenth of the way
  travelled. Read as detector spread on dash frames, not as a different move.
- **Every action acts** except up+none and down+none on the ground, as expected: walking 16-19 px,
  jump 110 px, lights 13-93 px, dash 122 px against the facing and 170 px with it, heavies 11-18 px
  with 4% of the window changed, throw 12-20 px (unarmed: a pickup attempt).
- **Charging:** a heavy held for 6 decisions ends 12 px below the start, still charging; one held
  for 1 or 2 decisions ends 10-11 px above it, already struck.
- **Pressing again:** a second jump pressed 4 decisions after the first adds height (227 px against
  141); pressed 2 decisions after (one decision released) it adds none (141 = 141). T2 showed that
  each frame's keys reach the game, so this is most likely the game refusing an aerial jump that
  early, not a lost press; not proven.
- Gate fixed: strips padded to one width; the numbers written to sequences.json before anything
  else; the torso spread allowed up to 12% of the way travelled; "acts" at twice the idle
  difference; a second jump 3 decisions after added. A clean run is still needed for the report.

**Trainer smoke** (configs/experiments/smoke.yaml, 5 rollouts from medium_v4, real time, Medium):
ran to the end. Stocks 3 - 3, damage 601 / 588; reward per minute by rollout 0.12, -1.31, 1.11,
-1.30, 1.50 (one-minute noise, as expected from N1). The update stalls the agent for 0.29-0.41 s
(17-24 game frames); 15.8 damage was taken during one of the four stalls. Perception 25-26 ms,
at most 0.3% of decisions over the budget; each part once per step (6,000 steps, 6,005 perceptions:
one at the start and one after each update). Too short to say anything about learning.

## 2026-11-12: the long probe run (gates S1, S2)

Kaggle T4 x2, 9 Medium sessions (131,599 decisions, 110 minutes), 3 folds x 2 seeds, every input
trained until its stop score stopped improving; 36 runs, none cut by the time limit
(results/probes/vision.md, runs/probes/vision/). Difference from `vector` (112 entries, with
timers) in the same fold and seed, mean ± spread over 6 runs:

| Input | Hit within 0.5 s: AUC | Mordex in 0.5 s (px) | Action NLL when the human changes | Action NLL |
|---|---|---|---|---|
| vector (reference) | 0.761 | 131.3 | 2.361 | 0.985 |
| vector, no timers | -0.027 ± 0.012 | +1.7 ± 0.5 | +0.026 ± 0.033 | +0.006 ± 0.010 |
| vector, GRU | +0.009 ± 0.016 | -2.4 ± 1.1 | -0.106 ± 0.048 | -0.039 ± 0.010 |
| both crops, 1 frame | +0.005 ± 0.015 | -6.3 ± 0.6 | +0.094 ± 0.051 | +0.041 ± 0.012 |
| both crops, 4 decisions | -0.006 ± 0.014 | -4.1 ± 1.6 | +0.131 ± 0.042 | +0.050 ± 0.011 |
| both crops, 1 frame, GRU | +0.007 ± 0.011 | -6.7 ± 0.9 | -0.012 ± 0.040 | +0.004 ± 0.009 |

- **Timers: confirmed** on the hit prediction (all 6 runs), as in the first memory probe.
- **A recurrent layer: confirmed** on imitation: all 6 runs better where the human changes action
  (-0.04 to -0.17), and better on Mordex's position in 5 of 6.
- **Pictures predict Mordex's position 5% better, in all 6 runs (-5.4 to -7.3 px), and nothing else.**
  No gain on the coming hit (round 1's +0.03 came from an undertrained baseline, as suspected), and
  the action prediction is worse in 5 of 6 runs. Stacking 4 decisions is worse than one frame on
  every measure. With the recurrent layer, pictures cancel its imitation gain.
- Picture inputs reach their best epoch early (26-44 against 78 for the vector): with 110 minutes
  the CNN overfits before it helps the action head. One stop score covers the three targets, so a
  target that overfits early ends the run for the others too.
- What this does not show: whether pictures help a policy that learns from far more data (RL
  produces 72,000 decisions per hour), or whether another view (one crop of the pair) would.

**Decision (proposed):** the policy's input stays the vector with timers. Pictures are set aside
until there is RL-scale data; the recurrent layer is the next addition worth an arm.

## 2026-11-12: Stage 1, first run: PPO from the clone, real time (H2)

`train.py --config configs/experiments/s1_ppo.yaml --variant realtime-bc --seed 0`: standard PPO
(configs/base.yaml) from medium_v4, against Medium, 60 one-minute rollouts (results/s1_ppo/summary.md).

| | Reward per minute (mean, standard error) | Stocks per 30 minutes |
|---|---|---|
| medium_v4 alone (N1 run, 6 x 5 min) | -0.03 (0.24) | 16 - 17 |
| PPO from medium_v4, whole hour | -0.60 (0.14) | 11 - 21 |
| PPO, first 10 minutes | -0.30 | |
| PPO, minutes 10-60 | -0.66 (0.15) | |

- In 5-minute blocks the PPO run is below the clone by 0.57 reward per minute: 2.1 standard errors;
  Mann-Whitney one-sided p = 0.012 (12 blocks against 6). It stays down, with no recovery within
  the hour (10-minute blocks: -0.30, -0.92, -0.87, -0.32, -0.46, -0.72).
- **First evidence for H2** (PPO degrades the cloned policy in real time), one seed of the five the
  prediction asks for. Caveats: the clone was measured in another session the same day; one seed.
- Mechanics held: updates stall the agent 0.30 s each (0.5% of the hour); 37 damage taken during
  them, no stock; 0.09% of decisions over the time budget; no lost focus.
- A likely cause to check, not yet measured: the clone brings a policy but no value function, so
  the first updates follow advantages from an untrained critic. progress.csv now records each
  update's approx_kl, clip fraction, entropy, value loss and explained variance, to see it.

Added: scripts/analyze.py (blocks, standard errors, change beyond noise), scripts/sweep.py (seeds
back to back, unattended).

## 2026-11-12: does a crop show what its fighter is doing? (scripts/probe_state.py)

Labels for free from the key log: Wu Shang is "attacking" if a light or heavy was pressed in the 4
decisions before (16.5% of decisions), "dodging" for a dodge in the last 4 (9.1%), "jumping" for a
jump in the last 3 (5.7%). 9 Medium sessions held out whole, 3 folds (results/probes/state.md):

| Input (Wu Shang only) | Attacking | Dodging | Jumping |
|---|---|---|---|
| kinematics (position, velocity, weapon, place) | 0.707 ± 0.007 | 0.812 ± 0.006 | 0.836 ± 0.013 |
| his 96 px crop, one frame | 0.875 ± 0.002 | 0.891 ± 0.000 | 0.929 ± 0.004 |
| crop + kinematics | 0.886 ± 0.002 | 0.901 ± 0.002 | 0.932 ± 0.003 |

- **The picture shows the action state** (AUC; 0.5 is chance): attacking +0.17 over what position
  and velocity tell, dodging +0.08, jumping +0.09. The user's intuition holds for the fighter whose
  state is labelled.
- **The reader does not carry over to Mordex.** Trained on Wu Shang, its "attacking" output on
  Mordex's crop ranks the decisions before Wu Shang takes damage at 0.566 (being close alone: 0.710).
  Two explanations cannot be told apart here: a reader of Wu Shang's sprites does not read Mordex's,
  or Mordex attacking does not predict damage because the human dodges.
- So the long probe's negative result is about the targets and the data size, not about what a
  crop contains. What would settle Mordex: labels of his own state. The key log gives them if the
  human plays Mordex against a Wu Shang bot.
- Held-out outputs written to <recording>/state_k3.npz.

## 2026-11-13: new priority, and the main engine built

**Priority (user):** an agent that reliably beats the bots comes first, as a showcase; the paper
follows from what is built. Engineering choices now favour what wins.

**Readers and vision (parked).** The recurrent readers, trained on two scripted hours with labels
shifted by the measured action-to-screen delay (3 decisions for the live agent, 2 in the human
recordings), read both fighters on human play (attacks 0.76 / 0.78 AUC, +0.16 to +0.21 over
position and velocity), but their outputs add nothing to imitation, and their hidden state hurts it
(results/probes/vision.md). Vision is parked until RL has produced far more data.

**The main engine: recurrent Q-learning from demonstrations** (brawl/rdqn.py; R2D2-style with one
actor, DQfD-style demonstrations). A GRU-256 Q-network over the 112-entry vector, dueling head over
the 54 actions, double Q, 5-step returns, sequences of 69 decisions with a 16-decision burn-in. The
15 human Wu Shang sessions (196,193 decisions, all levels) stay in replay; on their winning
stretches a margin loss (0.5) keeps the human's action ahead. The actor plays in the game process;
the learner runs in its own process on the CPU (4 threads, 74 ms an update, about 13 a second
possible; set to 0.4 per decision, 8 a second), so the detector keeps the GPU. Weights reach the
actor through a file every 100 updates (replacement retried: Windows refuses while it is read).

**Pretrained network** (scripts/pretrain_rdqn.py, 20,000 updates on the GPU; results/rdqn/pretrain_all.md),
held-out check on 5 sessions: right where the human changes action 0.213 (behaviour cloning 0.224),
buttons on 25% of decisions (human 30%), 52 of the 54 actions used: it plays like the clone and has
not collapsed. runs/rdqn/pretrain_all/qnet.pt starts the live runs.

**Also:** the tuned PPO baseline (configs/experiments/s1_ppo_tuned.yaml: value function alone for 5
minutes, learning rate 1e-4, clip 0.1, target KL 0.01); scripts/play.py plays a qnet.pt greedily;
configs are checked at every level. 52 tests pass. Not built: the difficulty macro (levels are set
by hand between runs).

## 2026-11-13: first smoke run of the main engine, and the copycat fault

`train.py --config configs/experiments/smoke_rdqn.yaml`, 5 minutes against Medium, from the network
pretrained with the last-4-actions history in its input: stocks 0 - 18, damage dealt 63, taken 790.
Bad from the first minute, before learning had changed much: the starting network, not RL.

**Cause, measured offline:** given held-out situations with the action history taken from random
other moments, the network chose the history's last action 88-90% of the time (the human repeats the
previous action 63-69%). Playing greedily, it then repeated its own last action, e.g. walking off the
stage (4 stocks lost in the first minute with 79 damage taken). The cloned policies survived because
they sample. The held-out check had fed the human's history, not the network's own, so it missed it.

**Fix:** the action history is left out of the Q-network's input (train.rdqn.drop_history); the GRU and
the timers keep what happened. Re-pretrained (results/rdqn/pretrain_all.md, held out):

| | With history | Without |
|---|---|---|
| Repeats a history taken from elsewhere | 0.88-0.90 | 0.156 |
| Right where the human changes action | 0.213 | 0.248 (cloning: 0.224) |
| Human's new button presses matched | 0.141 | 0.162 |
| Action right overall | 0.689 | 0.437 (it changes action more often than the human) |
| Decisions with a button (human 0.300) | 0.252 | 0.239 |

The copycat check is now part of scripts/pretrain_rdqn.py. The rest of the smoke run worked: learner
6.1-6.5 updates a second (asked 8: it shares the CPU with the game loop), perception 32 ms (26 without
the learner), under 1% of decisions late, one focus pause handled.

## 2026-11-14: the overnight run, its cause, and an offline test of value learning

**rdqn_medium seed 0** (7 h, 500,000 decisions, 186,000 updates; results/rdqn_medium/summary.md,
results/figures/rl_runs.png, rl_stocks.png): reward per minute fell from -1.85 (first 10 minutes) to
-4.32 (last 10), -2.47 +- 0.56 first to last block. Stocks 102 - 902. Damage taken per stock lost fell
from 73 (hour 1) to 19 (hour 7): increasingly, it lost stocks by itself. The machinery held for 7 hours
(perception 29-31 ms, 0.08% of decisions over the budget, no stall, no focus loss).

**Cause: value estimates that never tracked outcomes, then inflated** (results/figures/value_estimates.png).
Mean Q of the human's action against the true discounted return (gamma 0.995): Medium play +0.25 at
the start, +1.05 after the night, true +0.55; Chosen play +0.19, +0.78, true -0.43. Offstage moments
ended valued as high as onstage ones. The agent's own play (about -0.4 true) was valued about +1.1.

**Offline test of value learning** (scripts/value_bench.py, results/rdqn/value_bench.md, value_bench.png):
10 human sessions to train, 5 held out, 20,000 updates each; scored on Q(s, a_human) against the true
return. Regressing the true return directly: correlation 0.34-0.39, unbiased. The night's settings
(double Q, gamma 0.995, n 5, target 2000, lr 1e-4): correlation 0.25, the worst Q variant. Best:
gamma 0.99, n 20, target 500, lr 3e-5: correlation 0.43, bias +0.09, a lost stock within 3 s ranked at
0.74, a stock taken at 0.83. Without the margin loss every session is valued about +0.7 (it anchors the
scale). No variant values the held-out Hard session (lost) below zero: the bot's level is not in the
observation, and the training sessions are mostly won. Limit: the test uses human play only; the live
inflation came from the agent's own data.

**Changes:** falls are told apart from KOs (the ledger matches the HUD badges on two recordings); every
live decision is written to trace/; the live run keeps best.pt, the weights playing during its best
30 minutes; configs/experiments/rdqn_medium_v2.yaml uses the best settings, pretrained with them
(results/rdqn/pretrain_v2.md: copying 0.171, right where the human changes action 0.232, buttons 21%
against the human's 30%, its most used action 46% of decisions). scripts/plot_results.py draws every
figure from the run files.

## 2026-11-14/15: the greedy Q-network stands still; the advised clone beats the clone

**Exploration and the greedy player.** In the 5-minute run with the second network, uniform random
actions at 5% of decisions threw the weapon 18 times a minute (the human 7) and the agent was armed 47%
of the time (the human 90%). With exploration off, the greedy network stood still: trained to rank the
human's action first, its best action is the most common one, no button (70% of the human's
decisions), and standing still leads to states where it stays best. The Q-network alone is the wrong
player.

**The advised clone** (configs/experiments/rdqn_medium_v3.yaml). The agent samples from the clone
(medium_v4), its logits shifted by beta * clip(Q - E_clone[Q], 0.1); the Q-network evaluates the play
it sees (sarsa: no max, no inflation), pretrained without the margin loss. Advantages on human play are
small (median |A| 0.04, 90th percentile 0.12): beta 5 would have moved the clone's most likely action in
2.7% of decisions, beta 10 moves it in 5%, with an action's odds changed at most 2.7-fold. The first 30
minutes are the pure clone, the run's own baseline; beta then ramps over 30 minutes.

**Night of 2026-11-14** (7 h, results/rdqn_medium_v3/summary.md, results/figures/rl_runs.png):

| | Reward per minute | Stocks taken / lost per minute | Falls per minute | Damage dealt / taken per minute |
|---|---|---|---|---|
| Clone only (first 30 min) | -0.11 +- 0.19 | 0.53 / 0.57 | 0.27 | 108 / 116 |
| Advised (beta 10, 357 min) | +0.38 +- 0.06 | 0.66 / 0.57 | 0.15 | 134 / 104 |

Difference +0.49, 2.5 standard errors; Mann-Whitney over minutes p = 0.016. The clone measured on
2026-11-12: -0.03 +- 0.24. Stocks 270 - 236 over the night: the first run that takes more stocks than it
loses. By part of the night: +0.32, +0.28, +0.56 (last two hours). The values stayed honest (gap on live
play between -0.05 and +0.14, no trend); no rollback; 0.29% of decisions over the time budget.

Caveats: one seed; the in-run baseline is 30 minutes (standard error 0.19). A paired evaluation of the
frozen advised policy against the clone in one session is the next measurement. scripts/play.py now
plays the advised clone (--prior, --beta, --adv-clip).

## 2026-11-15/17: evaluation, the second night, distillation

**Paired evaluation** (2026-11-15, frozen policies back to back, 6 x 5 min each, results/play/eval_*.md): the
clone -0.08 +- 0.32 reward per minute, stocks 15 - 16, virtual matches 4 - 3; the advised clone (night 1's
best.pt, beta 10) +0.50 +- 0.21, stocks 20 - 14, virtual matches 6 - 1. +0.58, 1.5 standard errors alone,
the same gap as during the night (+0.49).

**Second night** (advised_medium_night2: continued from night 1's Q-network, beta 15, 6 h): clone alone
(first 30 min) +0.20 +- 0.25; advised +0.26 +- 0.07 (thirds +0.20, +0.46, +0.11); stocks 0.64 / 0.60 per
minute, falls 0.21 per minute (night 1: 0.15). No improvement on night 1 (+0.38). The clone's half hours
so far: -0.03, -0.08, -0.11, +0.20 (mean 0.00): the advised clone is about +0.3 to +0.5 above it, and
stops there. Reading: with the clone fixed and the advice clipped, a night is one bounded step of
improvement; a second night of the same adds nothing, and the stronger beta did not help.

**Distillation** (scripts/distill.py, runs/bc/advised_v1): behaviour cloning on both nights' advised play,
kept from stretches ending in a stock taken: 440,726 decisions (367 minutes). Held out: loss 0.887, actions
right 0.726 (repeat-previous 0.698), buttons on 23% of most-likely actions (played 28%). It becomes the
base of night 3 (configs/experiments/advised_night3.yaml), whose first 30 minutes measure it alone. Live
traces can now keep both fighters' crops (train.rdqn.trace_crops).

**Data:** the human recordings were deleted for disk space (2026-11-16); the 9 Medium sessions' decisions
and crops were restored from the probe pack, the only remaining copy.

**Vision value test** (2026-11-17, scripts/value_bench_vision.py, results/rdqn/value_bench_vision.md): the
advisor (sarsa, n 20, 10,000 updates) on the 9 Medium sessions, 3 folds. Vector: correlation 0.477, lose AUC
0.725, win AUC 0.862. Vector + both crops (CNN, lr 1e-4): 0.414 (-0.063 +- 0.021), lose AUC 0.608 (-0.117 +-
0.015), win AUC 0.828. Worse in every fold, consistently. Reading: with about 2.5 h of human play the
picture network overfits (the vector already carries positions, damage, timers); the same as the
imitation probes. Vision stays parked until the night traces with crops give 10x the data (night 3 on).

**Night 3, 5 h planned** (advised_night3: base advised_v1, night 2's Q-network, beta 10, crops traced). The
game crashed after 69 min (no Windows error logged); the HUD watchdog ended the run cleanly
(qnet_stopped.pt). Distilled clone alone (30 min): -0.29 +- 0.22 reward per minute, stocks 18 - 23, 8 falls
(the human clone's four half hours: mean 0.00). Beta ramping: -0.02 +- 0.24; full beta (9 min): +0.25.
But sensing was 7 ms slower all run from the first block (38 against 31 ms on both earlier nights; 40% more
late frames, over budget 2.3% against 0.4%): the laptop's state, plus possibly the crop shards (110 MB
compressed on the main thread every 5 minutes). Not a clean test of distillation: worse by 1.2 standard
errors, and confounded.

**Night 4 planned** (advised_night4, about 10 h): the human clone and the distilled clone take turns every
5 minutes under one advisor (train.rdqn.prior as a list, prior_every; progress.csv "prior"), so drift in
the laptop's speed falls on both arms. No crops.

## 2026-11-19: night 4, the A/B night (distilled clone against the human clone)

advised_night4: 10 h, 700,000 decisions, completed. The human clone (medium_v4) and the distilled clone
(advised_v1) took turns every 5 minutes under one advisor (night 2's Q-network, sarsa, beta 10). Crops saved
(6.2 GB). Sensing 33 ms median, over budget 1.0%, one rollback, learner lag 0.48 (night 2: 0.71).

| Phase | Human clone | Distilled | Paired (distilled - human) |
|---|---|---|---|
| No advice (15 min each) | +0.59 +- 0.31 | +0.68 +- 0.34 | +0.09 +- 0.05 (3 cycles) |
| Full advice (4.4 h each) | +0.34 +- 0.07 | +0.23 +- 0.08 | -0.09 +- 0.10 (52 cycles) |

Under full advice, stocks 182 - 174 and 179 - 173: identical. **Distillation does not help**; night 3's -0.29
was noise and the slow afternoon. Over the whole night, 399 - 386 stocks: against Medium the advised agent is
even on stocks and ahead on damage (+0.28 reward per minute, the plateau of nights 1 and 2 again). The
advisor, not the base, is what is bounded.

**Where stocks are lost.** 137 of 386 lost stocks (35%) were falls nobody was credited for (0.25 per
minute; Mordex 0.05). Before them the agent was off stage for about 3 s at low damage (median 30%, against
71% for stocks lost to a hit), steering toward the stage 66% of decisions, but using up + heavy (the
recovery) on only 7.5%. It comes back from 97% of off-stage spells; the 3% it does not cost a third of the
stocks. Recovery is rare and its failure is delayed and sparse: the kind of skill the value learning does
not find. Without those falls the agent would lose about 0.42 stocks a minute against 0.69 taken.
