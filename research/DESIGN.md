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
| `stepped` | Game paused; for each frame, presses the keys together with F6 | Between steps, while paused (no stalls) |

**Rewind** (`stepped` only): `rewind(n)` steps back `n` frames with F5. The clock counts every
frame it advances, so the environment always knows the exact frame index and can return to it.

**Decision period.** The agent decides every `k` game frames (config, default `k = 3`, i.e. 20
decisions per game second). The same `k` is used in both regimes, so a policy trained in `stepped`
mode sees exactly the same timing in `realtime`, as long as perception plus inference fit within
`k / 60` s (50 ms at `k = 3`). This budget is a deployment gate (D1).

**Train stepped, deploy in real time.** Stepped training removes stalls, late decisions and the
no-reset problem. The final evaluation, and the headline Chosen result, run in `realtime`, so the
claim is about an agent that plays at game speed.

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

## 5. Observation

Designed from game mechanics. Each feature names its sensor, and each must pass a probe or an
ablation (Gate S1) to stay.

**Per fighter (agent and opponent), from perception:**

| Feature | Sensor |
|---|---|
| Position (hurtbox centre), velocity over the last `k` frames | Hurtbox overlay or detector |
| Facing direction | Detector or sprite crop |
| Grounded, airborne, offstage; signed distance to each ledge and blast zone | Position + fixed stage geometry (one map) |
| Damage (0-1), stocks | HUD ring colour and digit |
| Weapon held: none, or which of the legend's two weapons | Detector classes |
| In hitstun | Stun flash overlay |
| Active hitbox: present, position relative to the other fighter | Hitbox overlay |

**World:** nearest ground weapon (relative position, present or not).

**Fighter crops (the policy's own vision).** Boxes and positions cannot show what a fighter is
*starting* to do: the first frames of an attack, a jump squat, a dodge. Hitbox overlays appear
only once an attack is already active, which is too late to react to. So the policy also sees two
small crops per decision: a window around each fighter (default 64 x 64, grey, the current and
previous decision frame), encoded by a small CNN that is part of the policy.

That encoder gets a training signal beyond reward. Every recorded frame is labelled by its own
future, at no cost: within the next `f` frames, did the opponent attack (a hitbox appeared), jump
(vertical velocity reversed), dodge, or hit the agent (a damage event)? An auxiliary head learns
to predict these from the crops. Rewind makes this a controlled measurement: from one paused
frame, we can see what the bot does next under different agent actions.

What does not need vision: jumps and airborne state follow from exact velocities (frames are
stepped, so velocities are exact); hitstun comes from the stun overlay; landed hits come from the
HUD. Gate S2 decides whether the crops earn their place: the probe "agent is hit within `f`
frames" must be predicted measurably better with crops than with positions and overlays alone.

**Memory:** the last `H` observations and actions (default `H = 4`), stacked. Jumps left and dodge
cooldown are not observable; instead of the old estimates, the action history lets the policy
infer them. A recurrent policy is a later ablation.

**Frame of reference:** positions relative to the stage centre; mirroring (flip so the agent
faces right) is an ablation, off by default.

**Raw pixels:** the captured frame is recorded every step (downscaled), so a pixel or crop encoder
can be tried later without new data.

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

In `stepped` mode, SB3's synchronous loop is automatically correct: the game only advances when
SB3 calls `step`, so updates never stall the world. In `realtime` mode the same code suffers real
stalls. **The regime is the experimental variable; the learner code is identical.** A custom
learner is written only if Stage 2 shows SB3's options are not enough.

## 9. Gates (Stage 0, before any training)

| Gate | Part | Pass condition |
|---|---|---|
| T1 | Keys | F5/F6/F7 and every action key are accepted by the game when sent by software |
| T2 | Stepping | One step with keys held moves the game exactly one frame with those inputs, in 1,000 of 1,000 steps |
| T3 | Determinism | Rewind `n` frames, replay the same actions: the captured frames match the originals |
| T4 | Throughput | Steps per second in `stepped` mode, and cost of a rewind per frame |
| T5 | Operator | The difficulty macro switches level 50 of 50 times, confirmed on screen |
| P1 | Perception | Position error, identity and weapon accuracy on the labelled frames, for each candidate |
| P2 | Overlays | Hitbox, hurtbox and stun detection accuracy on the labelled frames; and whether overlays hurt detection |
| R1 | Reward | Damage and KO events agree with the damage overlay and hand counts |
| A1 | Actions | Every move in the target list is reproducible |
| S1 | Observation | Each feature is justified by a probe or an ablation |
| S2 | Fighter crops | Crops predict "agent is hit within `f` frames" measurably better than positions and overlays alone |
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
4. Overlays on during play: hurtboxes, hitboxes, stun; damage numbers only for validating the HUD
   reader (they clutter the screen).
5. Train in `stepped` mode, report the headline in `realtime`.
6. Delete the old pipelines from the tree once Stage 0 passes (kept under the tag `legacy-eras`).
