# Hypothesis, objective and experimental design

> **DRAFT 2026-11-05, for review.** Nothing below has been run. Once agreed, the predictions are
> fixed before any result, as in PADP; later changes are recorded in [LAB_NOTEBOOK.md](LAB_NOTEBOOK.md)
> and marked "added after Stage N". The system that implements it is in [DESIGN.md](DESIGN.md).

## Objective

Beat Brawlhalla's **Chosen** bot from screen pixels, on one laptop, with no game API and no
parallel environments. Setting fixed for this paper: **Wu Shang against a Mordex bot on Brawlhaven
(small), in training mode.**

Ladder, one rung per bot difficulty (targets confirmed in review):

| Rung | Bot | Target: virtual-match win rate |
|---|---|---|
| R1 | Easy | >= 90% |
| R2 | Medium | >= 70% |
| R3 | Hard | >= 70% |
| R4 | Extreme | >= 50% |
| R5 | Chosen | > 50% (headline result, played in real time) |

## The environment, as it really is (verified in game, 2026-11-05)

- **One endless training-mode session** against a configurable bot. Health and positions are never
  reset by the game; we never choose where the agent spawns.
- **Pause** freezes everything. **Frame step forward** applies the keys held at the same moment to
  exactly one frame. **Frame step backward** reaches back over the whole session, and stepping
  forward from there changes the past.
- **Deterministic:** after a rewind, the bot repeats exactly what it did given the same inputs.
- **No self-play:** one laptop, one keyboard, one screen; the opponent is always a built-in bot.
- **Bot difficulty** steps up with Tab+D and down with Tab+A, so the ladder runs unattended.

So the same live game can be played in three regimes:

| Regime | Time | Resets |
|---|---|---|
| Real time | The game never waits | None |
| Stepped | The game advances only when the agent has decided | None |
| Stepped + rewind | The game waits | Return to any earlier frame and retry |

Few real-world learning problems allow this: the real-time constraint and the reset constraint can
each be switched on and off, against the same opponent, in the same environment.

## Win definition (virtual matches)

A **virtual match** is a race to 3 stocks, counted from KO events in the continuous stream: **win**
if the agent takes 3 stocks before losing 3, **loss** otherwise. The counters then reset; the game
is never interrupted. **Net stocks per minute** comes from the same event log.

## Research question

**When an agent can only learn by playing a live game, with no API and no parallel copies, what do
real-time pressure and the lack of resets cost it, and how much of that can be recovered?**

At 20 decisions per game second, one machine produces about 1.7M decisions per day of play: about
one Atari-200M budget every four months. Emulator results (Atari 100k, about 2 hours of game time)
show human-level learning in hours when the environment waits and resets. Here we can measure what
happens when it does not, and when it partly does.

## Main hypothesis (H)

Most of what makes live, single-instance RL hard is **time and resets, not the policy**:

1. **Real time wastes learning.** Updates stall the agent and decisions arrive late; stepping the
   game removes both.
2. **On-policy learning wastes data.** PPO uses each expensive frame a few times and discards it;
   a replay-based learner reuses every frame, including the rare ones.
3. **No resets starve rare skills.** Recovery and weapon situations arise only when the game
   produces them; rewinding to just before a failure lets the agent practise them directly.
4. **Stepped training transfers.** With the same decision period in both regimes, a policy trained
   stepped loses little when deployed in real time, provided perception fits in the time budget.

## Predictions

Each comes with the result that would refute it. Thresholds marked * are fixed after the Stage 0
noise floor, before Stage 1 starts.

| ID | Prediction | Falsified if | Stage |
|---|---|---|---|
| H1 Real-time cost | The same learner learns more per interaction stepped than in real time (win rate after a fixed budget, higher by a margin*) | No difference within the noise floor | 1 |
| H2 Degradation | BC -> PPO falls below the BC policy's win rate within 100k steps in at least 4 of 5 seeds, in real time | BC -> PPO matches or beats BC in most seeds | 1 |
| H3 Replay | In the stepped regime, a replay learner seeded with demos reaches each rung in fewer interactions than PPO and BC -> PPO | No significant difference in interactions-to-target or area under the win-rate curve | 2 |
| H4 Rewind | Rewinding to before lost stocks improves recovery success and overall win rate per interaction, compared with stepped training without rewind | No difference in recovery rate or win rate | 2 |
| H5 Transfer | A policy trained stepped keeps most of its win rate when evaluated in real time (gap below a margin*) | The real-time win rate collapses | 2 |
| H6 Ladder | The best regime and learner reach R3, R4 and R5, the last in real time | Plateau below R3 within the time budget | 3 |

H2 turns "PPO made BC worse in every old pipeline" into the motivating observation. H1 and H4
together measure the two constraints separately, which is the paper's main contribution.

## Contributions (if the predictions hold)

- **C1 Measurement.** What real-time pressure and the lack of resets each cost a single-instance
  learner in a live game, measured separately in one environment.
- **C2 System.** One clean loop that runs the same learner in real time or stepped, with exact
  rewind-based resets, from screen pixels and keyboard only.
- **C3 Result.** A seeded, statistically reported path up the bot ladder to Chosen, played in real
  time.

**Not claimed:** the first agent for a commercial game from pixels, a new RL algorithm, or
generality to other maps, legends or games.

## Closest prior work (to verify in a dated literature search, Stage 0)

| Area | Examples | What it leaves open here |
|---|---|---|
| Sample-efficient RL with replay | BBF (Schwarzer et al. 2023), Atari 100k | Emulators always wait and reset |
| Online RL with demos | RLPD (Ball et al. 2023), DQfD (Hester et al. 2018) | Simulated or paused settings |
| Real-time and delayed RL | Ramstedt & Pal 2019; Bouteiller et al. 2021 | Simulated delays, not a live system |
| Resets and exploration from saved states | Go-Explore (Ecoffet et al. 2021); reset-free RL | Emulator save states, or robots without resets |
| Fast real-world robot RL | SERL (Luo et al. 2024) | No controllable time or resets |
| Fine-tuning forgetting | Wołczyk et al. 2024 | Not real-time |
| Non-embedded game agents | e.g. the Clash Royale work | To compare against in the search |

## Design

The whole system (time control, step function, observation, actions, reward, learners, gates and
codebase) is in [DESIGN.md](DESIGN.md). In short: one environment whose clock is `realtime` or
`stepped`; one observation built once per decision from displayed signals where training mode
offers them; `MultiDiscrete([9, 6])` actions (direction x button); reward from HUD damage and KOs
only; learners built from configuration; every part gated before training.

**Evaluation:** virtual matches, deterministic policy, a pre-declared number per evaluation (from
the noise floor), win rate with confidence intervals, plus net stocks and damage per minute.
Comparisons between policies use paired starts from the same rewind anchors.

**Reporting:** curves against interactions and against hours of play; IQM with bootstrap
confidence intervals; every seed; Mann-Whitney tests per matched pair (PADP's statistics).

## Stages

| Stage | What | Tests |
|---|---|---|
| 0 | Gates for every part ([DESIGN.md](DESIGN.md) section 9), ground-truth labels, your demonstrations, noise floor, literature search | Nothing trained on a part before its gate passes |
| 1 | Easy and Medium: PPO and BC -> PPO, each in real time and stepped, 3-5 seeds | H1, H2 |
| 2 | Medium and Hard: replay learner, with and without rewind; real-time evaluation of stepped policies | H3, H4, H5 |
| 3 | The ladder, Hard to Chosen, with the best combination | H6 |

**Budget.** About 70 h/week unattended. Stepped mode runs slower than the game (perception sets the
pace), so interactions per hour are measured in Gate T4 and the stage budgets are fixed after it.

## Threats to validity

- **Stepped play is not real play.** Training is stepped, but the headline result is evaluated in
  real time, and H5 measures the gap directly.
- **Overlays are training-mode aids.** They are visible to any player, not memory reading. Stated
  as scope; an overlays-off ablation measures their value.
- **Carried-over health.** A virtual match can start with a fighter already damaged; report the
  damage at each start, and check results hold without large-gap starts. Paired anchors remove it
  from comparisons.
- **Training-mode bots vs match bots.** Assumed equal at a given difficulty; spot-checked with a few
  real matches at the end.
- **Exploitable bots.** Report the action distribution; a spam strategy that also beats Chosen is a
  result, but we name it as one.
- **HUD and perception errors.** Gates P1, P2 and R1; misreads logged per hour.
- **Game patches.** Record the game version with every run.
- **One map, opponent and legend.** Stated as scope.
