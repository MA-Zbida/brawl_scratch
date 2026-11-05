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
