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
- **Deterministic:** after a rewind, the bot repeats exactly what it did given the same inputs
  (seen by eye, and measured on replayed frames in gate T1).
- **No self-play:** one laptop, one keyboard, one screen; the opponent is always a built-in bot.
- **Bot difficulty** steps up with Tab+D and down with Tab+A, so the ladder runs unattended.

So the same live game can be played in three regimes, and each has a role in the paper:

| Regime | Time | Resets | Role |
|---|---|---|---|
| Real time | The game never waits | None | **Target setting**: what any live game offers. The method must work here, and the headline result is played here |
| Stepped | The game advances only when the agent has decided | None | Removes real-time pressure only: separates what time costs from what resets cost |
| Stepped + rewind | The game waits | Return to any earlier frame and retry | **Oracle**: the upper bound with full control of the same game |

**Instruments.** The paused toolbar's frame counter is ground truth for every step (presses are
confirmed against it; the game drops about 2-17% of unconfirmed presses, gate T0). Game state is
deterministic: replays after a rewind show identical fighters; only cosmetic animations (weapon
spawn effects, background phantoms) differ (gate T1). So comparisons can use paired starts from
identical frames.

This is like sim-to-real work with one difference: the reference is not a simulator but the real
game in a controlled mode, so there is no reality gap between the oracle and the target.

## Win definition (virtual matches)

A **virtual match** is a race to 3 stocks, counted from KO events in the continuous stream: **win**
if the agent takes 3 stocks before losing 3, **loss** otherwise. The counters then reset; the game
is never interrupted. **Net stocks per minute** comes from the same event log.

## Research question

**Using only the screen and the keyboard of a live game, how much of the gap between real-time,
reset-free learning and an oracle that can pause and rewind the same game can a learner close, and
what closes it?**

At 20 decisions per game second, one machine produces about 1.7M decisions per day of play: about
one Atari-200M budget every four months. Emulator results (Atari 100k, about 2 hours of game time)
show human-level learning in hours when the environment waits and resets. Most live settings
cannot measure how far they are from that; here the oracle makes the gap measurable.

## Main hypothesis (H)

Standard learners leave a large gap between the real-time setting and the oracle, and most of it
comes from **time and data waste, not from the policy**. Each of these can be attacked using only
screen and keyboard:

1. **Stalls and late decisions.** Synchronous updates stop acting while the game runs, and
   decisions arrive late. Remedy: learn asynchronously while acting, and make decisions
   delay-aware (action history; a predictive model of the opponent).
2. **Discarded data.** On-policy PPO uses each expensive frame a few times and discards it. Remedy:
   a replay-based learner seeded with demonstrations.
3. **Rare situations.** Without resets, recovery and weapon situations arise only when the game
   produces them. In real time, replay that keeps rare events longer partly substitutes; the
   oracle's rewind measures how much remains out of reach.
4. **Transfer is an alternative route.** Training in the oracle and deploying in real time, with the
   same decision period, should lose little if perception fits the time budget.

## Predictions

Each comes with the result that would refute it. Thresholds marked * are fixed after the Stage 0
noise floor, before Stage 1 starts.

| ID | Prediction | Falsified if | Stage |
|---|---|---|---|
| H1 Gap | PPO in real time reaches a lower win rate per interaction than the same PPO in the oracle, by a margin* | No gap within the noise floor: real time costs nothing here | 1 |
| H2 Degradation | BC -> PPO in real time falls below the BC policy's win rate within 100k steps in at least 4 of 5 seeds | BC -> PPO matches or beats BC in most seeds | 1 |
| H3 Method | The real-time method (asynchronous, delay-aware, replay with demos) closes at least half* of the H1 gap per interaction, and more per hour | It closes less than that | 2 |
| H4 Ingredients | Removing any one ingredient reopens a measurable part of the gap (ablation) | An ingredient's removal changes nothing: drop it from the method | 2 |
| H5 Resets | In the oracle, rewinding to before lost stocks improves recovery success and win rate per interaction over stepped training without rewind: the share of the gap only resets recover | No difference | 2 |
| H6 Transfer | An oracle-trained policy keeps most of its win rate when evaluated in real time (loss below a margin*) | The real-time win rate collapses | 2 |
| H7 Ladder | The better route (H3 or H6) reaches R3, R4 and R5, with R5 played in real time | Plateau below R3 within the time budget | 3 |

H2 turns "PPO made BC worse in every old pipeline" into the motivating observation. H1 and H5
measure the gap and split it between time and resets; H3 and H4 are the method's contribution.

## Contributions (if the predictions hold)

- **C1 Measurement.** The size of the gap between live, real-time, reset-free RL and a pause/rewind
  oracle of the same game, and how it splits between time pressure and resets.
- **C2 Method.** A learner for the real-time setting, using only screen and keyboard, and the share
  of the gap each of its ingredients closes.
- **C3 System.** One loop that runs the same learner in either regime, with frame-exact control
  (every step confirmed against the game's frame counter).
- **C4 Result.** A seeded, statistically reported path up the bot ladder to Chosen, played in real
  time.

**Not claimed:** the first agent for a commercial game from pixels, a new RL algorithm from scratch,
or generality to other maps, legends or games.

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
| 1 | Easy and Medium: PPO and BC -> PPO, in real time and in the oracle, 3-5 seeds | H1, H2 |
| 2 | Medium and Hard: the real-time method and its ablations; rewind vs no rewind in the oracle; real-time evaluation of oracle-trained policies | H3, H4, H5, H6 |
| 3 | The ladder, Hard to Chosen, by the better route | H7 |

**Budget.** About 70 h/week unattended. Gate T4: the oracle runs at about 20.7 game frames per
second with every press confirmed (6.9 decisions/s at `k = 3`), roughly a third of the game's
speed if the game simulates 60 frames per second (to be measured). Oracle runs are therefore the
expensive arm; stage budgets are fixed from these numbers.

## Threats to validity

- **The oracle is not real play.** It is the reference, not the target: the headline result and the
  method are evaluated in real time, and H6 measures what oracle training loses when deployed there.
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
