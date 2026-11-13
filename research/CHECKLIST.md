# Checklist

Read this first in each session; update it when a box changes. The plan behind it is in
[HYPOTHESIS.md](HYPOTHESIS.md) (why) and [DESIGN.md](DESIGN.md) (how), and the record of what was
done is in [LAB_NOTEBOOK.md](LAB_NOTEBOOK.md).

## The goal

Beat the **Chosen** bot (Wu Shang vs Mordex, Brawlhaven small, training mode) from the screen and
keyboard only, climbing Easy (90%) -> Medium (70%) -> Hard (70%) -> Extreme (50%) -> Chosen (>50%).
The paper measures how much of the gap between learning in real time and learning in a
pause-and-rewind oracle a learner can close.

## The rule that decides everything

Nothing is trained on a part until that part's gate passes. A wrong input fails every agent
built on it, and no RL result can show where the fault was.

## Your next steps (2026-11-13, night)

Run everything in the `brawl312` environment. Camera mode: Fixed. You as Wu Shang, bot Mordex on Medium.

1. Smoke run of the main engine, 5 minutes: `python scripts/train.py --config configs/experiments/smoke_rdqn.yaml`
2. Overnight, about 7 hours: `python scripts/train.py --config configs/experiments/rdqn_medium.yaml --seed 0`
3. In the morning: `python scripts/analyze.py rdqn_medium`, then evaluate the result:
   `python scripts/play.py runs/rdqn_medium/default/seed0/qnet.pt --difficulty medium --minutes 5 --blocks 6`

## Stage 0: gates (DESIGN section 9)

| Gate | Part | Status |
|---|---|---|
| T1 | Keys reach the game | Passed |
| T2 | One step = one frame | Passed |
| T3 | Rewind + replay is deterministic | Passed (by witness; positions check comes with P1) |
| T4 | Stepping throughput | Passed (about 20.7 frames/s) |
| T5 | Difficulty macro | Not built |
| P1 | Perception: position, identity, held weapon, free weapons | Passed with the tracker (2026-11-11, 960 px TensorRT engine, 11 ms per frame): found 1.00 / 0.99, torso p95 13.6 / 15.3 px (pass at 20), held 0.96 / 0.95, spawn 0.97 / 0.97, thrown 0.81 / 0.81. Weak spots, known: Mordex held weapon is on the threshold and reads 0.76 on the Easy validation frames; thrown weapons 0.40 / 0.67 there (5 labels) |
| R1 | Damage and KO events (HUD reader) | Passed on recordings (2026-11-10): KOs reproduce the final scores of 4 sessions; 10 overlay hits within 1.9 damage. Live check on lossless frames comes with the env |
| A1 | Every move in the action list is reproducible | Passes by the printed table of the second run (2026-11-12): 54 of 58 sequences identical by pixels and torso, the 4 others are dashes with the detected torso 14-18 px apart; every action acts; charging works; a second jump works 4 decisions after the first, not 2. The run crashed while saving its strips, so the report file still needs one clean run (gate fixed) |
| S1, S2 | Observation features justified; representation chosen offline | Long run done (2026-11-12, results/probes/vision.md): timers and a recurrent layer earn their place; fighter crops improve only the prediction of Mordex's position (5%) and worsen imitation at 110 minutes of data. Input: the vector with timers; pictures set aside until RL-scale data |
| L1, D1 | Loop runs each part once per step; fits the time budget | Passed in the live game (2026-11-11, two 5-minute runs): each part once per step; perception + policy 26 ms median, 34 ms p95, of 50 ms; 0.13% of decisions over |
| N1 | Noise floor: matches per evaluation, seeds per arm | First measurement 2026-11-12 (medium_v4 against Medium, 6 x 5 min): reward per minute -0.03, spread between blocks 0.58; 30 minutes per policy tell apart differences above 0.67. Still to add: other seeds and days |

## Build order (DESIGN section 12)

1. [x] Game layer: keys, clock, capture (T1-T4).
2. [ ] Recording and labelling.
   - [x] Recorder and pre-labeller.
   - [x] Label rules fixed (docs/labelling.md).
   - [x] Test set: 150 frames from 2 sessions, labelled.
   - [x] Labels fixed to one rule (weapon-inclusive box + torso point): test set done, 149 frames.
   - [x] Train/valid torso points on spear and scythe boxes (export v3).
   - [ ] Train/valid `weapon_thrown`: delete the horn and mine boxes (rule 11); test has none.
   - [x] Batch 2: 299 frames from Hard, Extreme and Chosen sessions, corrected (export v5, 943 frames).
   - [x] Item boxes removed from `weapon_thrown` (export v6); pack built, training on Kaggle (1280 and 960 px).
   - [x] Five frames with two boxes of one fighter (3 train, 2 test): fixed in the local export (LOCAL_FIXES.txt).
   - [ ] Mirror those five fixes in Roboflow, or the next export undoes them.
   - [x] Test labels found wrong by the 2026-11-11 run, fixed in the local export: Mordex box on the KO bubble (155225 row 4931); spear points left at the box centre (155225 rows 28278, 2223, 28755); thrown spear at the right edge unlabelled (151844 row 20387).
   - [ ] Mirror those five test fixes in Roboflow too (LOCAL_FIXES.txt lists all ten edits).
   - [ ] More demonstrations: hours of play for behaviour cloning, recorded on an unloaded machine.
3. [ ] Perception.
   - [x] Export reader (brawl/sense/truth.py).
   - [x] P1 scorer (scripts/gate_p1.py).
   - [x] Reader and P1 use the torso point (old detector baseline: p95 22-26 px, fails).
   - [x] Deployed detector (YOLO pose, one keypoint), trained on Kaggle; TensorRT engine by scripts/export_detector.py.
   - [x] Fighter tracker (brawl/sense/track.py): carries a hidden fighter, weighs doubtful boxes, votes the held weapon; scored by `gate_p1.py --track 30`.
   - [x] Tracker on weapons: a spawn marker reported when seen in 2 of 3 frames, one box per marker.
   - [ ] Held-weapon crop classifier.
   - [ ] Thrown-weapon owner tracking.
   - [x] HUD reader and event ledger (R1): brawl/sense/hud.py, brawl/ledger.py.
   - [x] Stage geometry (brawl/stage.py): ledges at x 605 and 1305, feet at y 617, measured from the labels.
4. [ ] Actions (A1).
   - [x] The 54 actions as keys, and recorded keys read back as actions (brawl/actions.py).
   - [ ] Gate A1 in the game: each move performed and checked on screen (`python scripts/gate.py a1`).
5. [ ] Observation, ledger, env (S1, L1, D1).
   - [x] Perception per decision (brawl/sense/perception.py): detector, tracker, HUD and ledger in one call, used by the environment and by the demonstration reader.
   - [x] Observation vector (brawl/observation.py) and environment (brawl/brawlenv.py; `brawl/env.py` once the legacy `brawl/env/` folder is deleted).
   - [x] Live run: `scripts/play.py` in the game (L1, D1).
   - [ ] Live check of R1: stocks and damage counted by the run against the HUD score read by hand.
6. [ ] Config, learners, train/sweep/analyze/evaluate (N1), then Stage 1.
   - [x] Demonstrations to decisions (scripts/prepare_demos.py) and behaviour cloning on winning stretches (brawl/learn.py, scripts/train_bc.py).
   - [x] Demonstrations against Medium: 70 minutes (6 sessions); learning curve still rising, slowly.
   - [x] Vision probes (S2), first round: pictures of both fighters improve Mordex's position 0.25-0.5 s ahead and the hit prediction in every fold; not yet the imitation (results/probes/vision_round1.md).
   - [x] Memory probe: 13 timers in the vector (results/probes/memory.md); a recurrent layer helps imitation and is an arm of the long run.
   - [x] Long probe run on Kaggle: pictures add 5% on Mordex's position, nothing on defence, and cost imitation; the recurrent layer helps imitation.
   - [ ] Recurrent policy as an arm (cloning and RecurrentPPO), after the Stage 1 baseline.
   - [x] First cloned policies played in the game: 4 - 3 against Easy, 1 - 4 against Medium in 5 minutes.
   - [ ] Noise floor (N1): the same policy over several 5-minute runs.
   - [x] Config system (brawl/config.py, configs/base.yaml, configs/experiments/) and the PPO trainer (scripts/train.py): tested against a drawn game.
   - [x] Smoke run in the game (2026-11-12): 5 rollouts, update stall 0.3-0.4 s, timing within budget.
   - [ ] Stage 1 runs (configs/experiments/s1_ppo.yaml): realtime-bc seed 0 done (2026-11-12): below the clone by 0.57 reward per minute (2.1 standard errors). Seeds 1-4 next (`scripts/sweep.py`).

## Open decisions

| Decision | How it gets decided |
|---|---|
| Held weapon: nine-class pose detector, or a crop classifier if its class fails | P1, on the same test labels |
| P1 thresholds | Reviewed 2026-11-11: torso error p95 20 px instead of 12 (hand-placed points agree to about 15 px); the others unchanged |
| Deployed detector size and input resolution | P1 accuracy against the D1 time budget. Measured 2026-11-11 on the laptop GPU, per frame, of 50 ms per decision: PyTorch 1280 px 33 ms, 960 px 21-24 ms; TensorRT 960 px 11 ms with the same answers |
| Red swirl before a weapon appears | Decided 2026-11-11: `weapon_spawn`, as the train labels already do (docs/labelling.md rule 9); two test frames corrected |
| Confidence thresholds | In the tracker: fighters 0.25 alone, or lower when in place; weapons 0.25 (0.4 and 0.5 lose spawn recall on validation) |
| Where the policy runs | CPU for the small MLP, GPU for the detector only (your earlier measurement with SB3); the time-budget gate D1 confirms it |
| Crop resolution (64 or 96) and stack depth | S2 offline benchmark; the first runs use the structured vector alone (2026-11-11) |
| Charged heavy attacks | Built 2026-11-11: a button is held for its decision and stays held while chosen again; pressing it twice needs one decision in between. Gate A1 checks both |
| Which stretches the policy is cloned from | Winning stretches only (cut at each KO; those ending in Wu Shang's KO, and the unfinished last one, are dropped). `--keep` in scripts/train_bc.py changes it |
| Self-destructs (-3 on the HUD): rewarded like any lost stock, or apart | Default: the same, counted separately in the log. Yours to change |
| Offscreen bubble not labelled (2026-11-10): beyond the edge = not detected and no stock lost | Revisit if recovery is weak: the bubble shows where along the edge the fighter is, and a missed detection reads as offscreen |
| Recorder drops under machine load | Cause found: many open browser tabs. Record with them closed |
