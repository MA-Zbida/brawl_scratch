# System design

> **DRAFT 2026-11-05, for review.** The whole system, from the screen to the key press, and the
> minimal codebase that implements it. The research questions it serves are in
> [HYPOTHESIS.md](HYPOTHESIS.md). Nothing from the old system is assumed to work: where an old
> piece is reused, it is because it passed the gate for its part (section 9).

## 1. Principles

1. **One flow.** One environment, one step function, one observation builder, one reward
   ledger, one recording format. Methods differ only in configuration.
2. **Everything once per decision.** Each frame the agent decides on is captured once, perceived
   once, turned into one observation and one reward. No wrapper recomputes or overwrites it.
3. **Measured before trusted.** Every sensor and actuator has a gate measured against ground truth
   before anything is trained on it.
4. **Minimal.** About 20 modules instead of 165. If a module has no caller in the experiment path,
   it does not exist.

## 2. What the game gives us (verified in game, 2026-11-05)

| Fact | Consequence |
|---|---|
| Training mode runs one endless session against a configurable bot | Continuing task; no match restarts |
| **Pause** (F7) freezes everything: game, bot and agent | Learning can happen while the world waits |
| **Frame step forward** (F6): pressing the action keys at the same moment as F6 applies them to that frame | A true step function: the game advances only when the agent has decided |
| **Frame step backward** (F5) reaches back over the **whole session**; stepping forward again from there changes the past | Resets: return to any earlier frame and try something else |
| **Deterministic**: after a rewind, the bot repeats exactly what it did given the same inputs | Reproducible start states; paired comparisons from identical situations |
| Bot difficulty: Tab+D steps up (Easy, Medium, Hard, Extreme, Chosen, then Evasive), Tab+A steps down | The ladder runs unattended, with the policy suspended during the switch |
| Overlays: hurtboxes, hitboxes, damage numbers, combo counter, force, stun flash | Displayed signals instead of inferred ones; damage numbers clutter the screen mid-combo |
| The HUD shows damage as a ring colour and stocks as a digit beside each portrait | The damage source for reward; the old RGB decoder is a candidate |

On a laptop the keys are pressed with FN (FN+F5/F6/F7). Software sends the F-key scan codes
directly; Gate T1 checks that the game accepts them.

## 3. Time control: three regimes, one code path

The environment's `Clock` has one interface, `advance(keys, frames)`, with two implementations
selected by configuration:

| Regime | `advance` does | Learning happens |
|---|---|---|
| `realtime` | Holds the keys and waits until `frames` game frames (1/60 s each) have passed | While the game runs (stalls are real) |
| `stepped` | Game paused; for each frame, holds the keys and presses F6, then waits for the frame counter to change (re-pressing if the game dropped the press) | Between steps, while paused (no stalls) |

**Rewind** (`stepped` only): `rewind(n)` steps back `n` frames with F5, each confirmed the same way.

**Confirmed steps.** The paused toolbar shows the game's frame counter. The game drops some F6
presses, mostly when fight keys change on the same step (gate T0: 166/200 applied unconfirmed). With
every press confirmed against the counter, 800 of 800 presses landed exactly, as read by a human
(T0). A 10 ms gap between a key change and the following F6 reduces drops; confirmation removes
them. The game also lands some presses late (T3: a press arriving after its retry moved the game
two frames), so the clock reads the counter's **number** (template-matched digits, calibrated once
by `gate.py counter`). The counter restarts at 0 on every pause and goes negative below it. The
clock's `frame` is that number. Retries wait longer each time, and a double landing is undone
exactly: F5 back, or F6 forward with the keys recorded for that frame (`keys_at`). Rewinding across
weapon throws works (witnessed).

**Decision period.** The agent decides every `k` game frames (config, default `k = 3`, i.e. 20
decisions per game second). The same `k` is used in both regimes, so a policy trained in `stepped`
mode sees exactly the same timing in `realtime`, as long as perception plus inference fit within
`k / 60` s (50 ms at `k = 3`). This budget is a deployment gate (D1).

**Roles of the regimes** ([HYPOTHESIS.md](HYPOTHESIS.md)). `realtime` is the **target setting**:
the method must work there and the headline Chosen result is played there. `stepped` with rewind is
the **oracle**: the same game with full control, used as the upper bound that every real-time
result is measured against, and as an alternative training route (train in the oracle, deploy in
real time, hypothesis H6).

## 4. The step function

One `BrawlEnv(gym.Env)`. Everything below happens exactly once per `step`:

```
step(action)
  keys   = actions.to_keys(action, mirrored)        # one mapping, one place
  clock.advance(keys, frames=k)                     # realtime wait, or k paused frame steps
  frame  = capture.grab()                           # one capture, after the last frame
  sense  = perception(frame)                        # fighters, weapons, hitboxes, HUD: one pass
  obs    = observation.build(sense, history)        # one vector; history is a ring buffer
  events = ledger.update(sense.hud, frame_index)    # damage and KO events, virtual matches
  reward = reward_fn(events)                        # from events only
  record.append(frame_index, obs, action, events, timing)
  return obs, reward, terminated=False, truncated=window_end, info=events
```

`reset(options)` chooses a start state instead of pretending to reset the game:

| `reset` mode | What happens |
|---|---|
| `continue` | No rewind: the next window starts where the last one ended (always used in `realtime`) |
| `rewind_to_failure` | Rewind to `m` frames before a recent lost stock (for rare situations such as recovery) |
| `rewind_to_anchor` | Rewind to a fixed frame; used by paired evaluation |

Episodes are only windows. `terminated` is always false; `truncated` marks a window end or a
rewind, so value targets keep bootstrapping.

## 5. Observation: hybrid (structured state + fighter vision)

The policy sees two things each decision: a **structured vector** of quantities we can measure
reliably, and **stacked images around each fighter** for what only pixels show. Both are fused
inside one network.

### 5.1 What the literature says, and what follows for us

| Finding | Source | Consequence here |
|---|---|---|
| The strongest fighting-game agents use true game state, not pixels: Melee (positions, velocities and the **action state**, i.e. which animation is playing, read from memory), Blade & Soul (game-state features), FightingICE (about 165 simulator features) | Firoiu et al. 2017; Oh et al. 2019/2022; Kim et al. 2020 | We cannot read memory. The structured vector covers what we can measure; the one field we cannot measure, the **action state**, is what the image branch must recover |
| Pixels alone learn much more slowly than state, though they can match it eventually | Yarats et al. (SAC+AE); Google Research Football (Kurach et al. 2019) | Never pixels alone at our budget: images complement the vector, they do not replace it |
| Large game agents fuse a vector branch with an image-like branch: scalar + entity + spatial encoders concatenated (AlphaStar); vector features + image-like features (Honor of Kings); visual input plus game features speeds up learning (Doom) | Vinyals et al. 2019; Ye et al. 2020; Lample & Chaplot 2017 | Late fusion by concatenation is the standard, proven design. We use it |
| Adding images to proprioceptive state improves sample efficiency in robot learning | Multimodal information bottleneck (2024); others | Hybrid is the right default; whether it helps *here* is measured (5.4) |
| A small CNN trained from scratch with random-shift augmentation matches frozen pretrained encoders (R3M, MVP, PVR) | Hansen et al. 2023 (DrQ-v2 baseline) | No pretrained ResNet: a small CNN with augmentation, which also fits the time budget |
| Predicting the future in representation space is what makes pixel RL data-efficient (Atari 100k); BBF builds on it | Schwarzer et al. 2021 (SPR); 2023 (BBF) | The image branch gets an auxiliary prediction loss, not reward alone (5.3) |
| Reaction delay hurts sharply; a predictive model of the opponent offsets it | Firoiu et al. 2018 ("At Human Speed") | The image branch's auxiliary target is the opponent's next option |
| A critic may see privileged state the actor does not | Pinto et al. 2018 (asymmetric actor-critic) | Available if overlays are tried later (critic only); not used by default |

The structured-plus-vision design you proposed is the standard one for game agents. What has to
be adapted is the image branch: Brawlhalla's fighters are small (roughly 60-90 pixels tall at 1080p in your screenshots), so
the usual full-screen 84 x 84 Atari frame would shrink a fighter to about 4 pixels and lose the
animation. We look at the fighters instead.

### 5.2 Structured vector (measured quantities only)

Every entry must pass its accuracy gate (P1, R1). "True" means measured and validated, not assumed.

| Feature, per fighter (agent and opponent) | Sensor |
|---|---|
| Position (body box centre) and velocity over the decision period | Detector |
| Grounded, airborne, offstage; signed distance to each ledge and blast zone | Position + fixed stage geometry |
| Damage (0-1), stocks | HUD ring colour and digit |

Plus: the nearest free weapon (on the ground or thrown: relative position, present or not); the
last `H = 4` actions (jumps left and dodge cooldown are not observable, so the action history
stands in for them). Facing, the held weapon and attack or stun states are not structured
features: the fighter crops show them (5.3). The detector's classes and the labels that measure
it are fixed in [../docs/labelling.md](../docs/labelling.md).
Positions are relative to the stage centre; mirroring is an ablation, off by default.

### 5.3 Fighter vision (the missing action state)

- **Crops:** a window of about 160 x 160 screen pixels centred on each fighter (from the
  structured position), downscaled to 64 x 64 grey; the exact size is fixed from the labelled frames; one context view of the whole screen at 96 x 54 for items and
  weapons on the ground.
- **Stack:** the last 4 decisions (at `k = 3`, 12 game frames, 0.2 s), the window over which
  move startups and jump squats show.
- **Encoder:** a small CNN shared by both fighter crops (Nature-DQN size), trained from scratch with
  random-shift augmentation.
- **Auxiliary loss:** each recorded frame is labelled by its own future, for free: within the next
  `f` frames, did the opponent attack (a hitbox appeared), jump, dodge, or hit the agent (a damage
  event)? A prediction head trained on these labels gives the encoder a dense signal; an SPR-style
  latent prediction loss is the alternative if explicit labels prove too coarse.
- **Fusion:** CNN embeddings concatenated with an MLP embedding of the structured vector, then a
  shared MLP to policy and value heads.

### 5.4 Benchmark it offline, not with RL runs

Choosing the representation does not need weeks of live play. All candidates are compared on
**recorded data** (your demonstrations plus agent play) by supervised probes, in hours of compute:

1. **Information:** predict the future-labelled events (agent hit within `f` frames; opponent
   attack, jump, dodge) and your next action (behaviour cloning accuracy).
2. **Candidates:** structured only; structured + crops; structured + crops + context view;
   crops only. Stack depth 1, 2 or 4.
3. **Cost:** inference time per decision, which must fit the deployment budget (D1).

The cheapest candidate within noise of the best is chosen (Gate S2). Only that one goes to RL,
plus one RL ablation (structured only vs hybrid) in Stage 2 to confirm the probe's choice.

**Raw frames** are recorded every step (downscaled), so all of this can be redone later without
new data.

## 6. Action

From the controls, not from the old space:

- **Direction** (9): none, left, right, up, down, and the four diagonals. Held for the whole decision
  period: it moves, aims attacks and steers recovery.
- **Button** (6): none, jump, light attack, heavy attack, dodge, pick up or throw. Pressed on the
  first frame of the period, released after `press_frames` frames (config, default 1).

That is 54 actions as `MultiDiscrete([9, 6])`. Direction plus button gives every move class:
neutral, side and down lights and heavies (signatures) on the ground; aerials and recovery in the
air; directional dodges. The old `Discrete(27)` could not express directional attacks.

Gate A1 decides whether this is enough: a scripted test performs each move in a target list and
checks it on screen. Masks: none at first (only "throw" is meaningless while unarmed; the policy
can learn that).

## 7. Reward and evaluation

**Events** (the ledger, from the HUD every step):

- `damage_dealt`, `damage_taken`: positive changes of each fighter's damage reading.
- `ko_scored`, `ko_lost`: stock changes or respawn.

**Reward** (no shaping):

```
r = (damage_dealt - damage_taken) / 100 + 1.0 * (ko_scored - ko_lost)
```

The weights are config; this is the only reward. Gate R1 validates the events against the
damage-number overlay and hand counts.

**Discount:** gamma = 0.995 per decision (half-life of about 7 s of game time at 20 Hz). Config.

**Virtual matches:** the ledger counts a race to 3 stocks, as in [HYPOTHESIS.md](HYPOTHESIS.md).
Evaluation reports win rate (with confidence intervals), net stocks and damage per minute.

**Paired evaluation:** two policies start from the same rewind anchors, so a difference cannot
come from easier starting situations. This needs fewer evaluation minutes for the same confidence.

## 8. Learning

**Data.**

- **Human demonstrations:** you play the setting in real time; your key presses are logged with
  each decision frame and mapped to the 54 actions. These replace the old scripted teachers.
- **Agent experience:** every step is recorded in the same format (section 10).

**Learners**, all built from configuration with Stable-Baselines3 (no hand-written training loops):

| Learner | Role |
|---|---|
| PPO | On-policy baseline |
| BC -> PPO | Reproduces the old degradation (H2) |
| QR-DQN (or DQN) with replay prefilled by demos | Main off-policy candidate; reuses every frame |

In `stepped` mode (the oracle), SB3's synchronous loop is automatically correct: the game only
advances when SB3 calls `step`, so updates never stall the world. In `realtime` mode the same code
suffers real stalls, which is exactly the baseline the gap (H1) is measured from.

**The real-time method** (H3) needs one component SB3 does not have: **asynchronous learning**. An
actor thread keeps stepping the live game at the decision rate with the latest policy weights,
while the learner updates from the replay buffer in parallel; weights are copied to the actor
after each update. This is one small module (`learn.py`'s async runner), written in Stage 2 and
used only by real-time runs; the learner algorithms themselves stay SB3's. Its other ingredients
(action history, the opponent-prediction head, demonstration-seeded replay) are already part of
the observation and data design.

## 9. Gates (Stage 0, before any training)

| Gate | Part | Pass condition |
|---|---|---|
| T1 | Keys | F5/F6/F7 and every action key are accepted by the game when sent by software |
| T2 | Stepping | One step with keys held moves the game exactly one frame with those inputs, in 1,000 of 1,000 steps |
| T3 | Determinism | Rewind `n` frames, replay the same actions: the captured frames match the originals |
| T4 | Throughput | Steps per second in `stepped` mode, and cost of a rewind per frame |
| T5 | Operator | The difficulty macro switches level 50 of 50 times, confirmed on screen |
| P1 | Perception | Position error, identity and weapon accuracy on the labelled frames, for each candidate |
| P2 | Overlays (optional) | Only if overlays are tried: hitbox, hurtbox and stun detection accuracy, and whether overlays hurt fighter detection |
| R1 | Reward | Damage and KO events agree with the damage overlay and hand counts |
| A1 | Actions | Every move in the target list is reproducible |
| S1 | Observation | Each feature is justified by a probe or an ablation |
| S2 | Representation | Offline probes (section 5.4) pick the cheapest candidate within noise of the best, within the time budget |
| L1 | Loop | Each part runs once per step (counted); per-step timing logged |
| D1 | Deployment | Perception + inference fit within `k / 60` s at the deployment `k` |
| N1 | Noise floor | Variance of virtual-match outcomes for a fixed policy, giving matches per evaluation and seeds per arm |

Each gate is one script that writes `results/gates/<gate>.md`. Gate code is the system code: a gate
that needs special code is a sign the module is wrong.

## 10. Recording format

One format for demonstrations, training experience and evaluation: shards of `npz` with one row per
decision step: `frame_index`, `regime`, `obs`, `action`, `events`, `reward`, `timing`, plus a
downscaled frame. BC, replay prefill, evaluation and analysis all read it. The old world-replay,
demo and Round A dataset formats are not carried over.

## 11. Minimal codebase

```
brawl/
  paths.py          repository locations (kept)
  config.py         YAML base + experiments + grid + --set (ported from PADP's config_io)
  stats.py          IQM, bootstrap CI, Mann-Whitney (from PADP)
  game/
    capture.py      DXGI frame grab; acquired before torch loads (from capture_first + dxgi)
    keys.py         the only module that touches the keyboard: hold, tap, F5/F6/F7, macros
    clock.py        realtime | stepped; advance, pause, rewind; exact frame counter
  sense/
    perception.py   one call per frame: fighters, weapons, hitboxes, hurtboxes, stun
    hud.py          damage rings and stock digits
  actions.py        MultiDiscrete([9, 6]) <-> keys; mirroring
  observation.py    the feature list (one table) and the builder
  ledger.py         events, reward, virtual matches
  env.py            BrawlEnv: the step and reset above
  record.py         the recording format
  learn.py          builds SB3 learners and BC from config
scripts/
  gate.py           run one gate (section 9)
  record_demos.py   log your play as demonstrations
  label.py          label frames for the ground-truth set
  train.py  sweep.py  analyze.py  evaluate.py   (as in PADP)
configs/
  base.yaml  experiments/*.yaml
tests/              unit tests against a fake game (see below)
```

**Testing without the game.** A `FakeGame` implements the same capture/keys/clock interfaces with a
tiny deterministic simulation (two points, damage counters, stocks). Every module is tested against
it, so CI runs without Windows or a GPU.

**What happens to the old code.** Before the rewrite starts, the current tree is tagged
`legacy-eras`, so every old pipeline stays reachable. Each new module is written fresh; an old
piece is copied in only after it passes its gate (likely candidates: DXGI capture and the
capture-before-torch rule, the HUD colour decoder, the key injection). When Stage 0 passes,
`brawl/pipelines/`, the old `scripts/`, `configs/legacy/` and their tests are deleted from the tree.
The old checkpoints are not needed by the new system (new observation and action spaces), so
`brawl/compat.py` goes too.

## 12. Build order

Each step ends with its gates passing:

1. `game/keys.py`, `game/clock.py`, `game/capture.py` -> T1, T2, T3, T4, T5.
2. `record.py`, `scripts/record_demos.py`, `scripts/label.py` -> the ground-truth set and your
   first demonstrations.
3. `sense/hud.py`, `sense/perception.py` -> P1, P2, R1.
4. `actions.py` -> A1.
5. `observation.py`, `ledger.py`, `env.py` -> S1 (partly), L1, D1.
6. `config.py`, `learn.py`, `scripts/{train,sweep,analyze,evaluate}.py` -> N1, then Stage 1.

## 13. Decisions for review

1. `k = 3` (20 decisions per game second) as the default decision period.
2. `MultiDiscrete([9, 6])` actions, button pressed for 1 frame.
3. Reward `(dealt - taken) / 100 + 1.0 * (KO scored - KO lost)`, gamma 0.995.
4. **No overlays during play** (revised 2026-11-05): the existing labelled data and detector have none, the
   fighter crops are meant to recover the action state from the animation, and "what any player sees" is the
   cleaner claim. Hitbox, hurtbox and stun overlays become an optional experiment if the offline benchmark
   (S2) shows the vision missing attacks. Damage numbers only for validating the HUD
   reader (they clutter the screen).
5. `realtime` is the target setting and the headline; `stepped` with rewind is the oracle and an
   alternative training route.
6. Delete the old pipelines from the tree once Stage 0 passes (kept under the tag `legacy-eras`).
