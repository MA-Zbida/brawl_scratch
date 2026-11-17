# Play: eval_clone

2026-11-15 11:04. Policy runs/bc/medium_v4/policy.zip (sampled), bot medium, realtime, k = 3, 6 x 5 min. 36004 decisions in 1800 s (20.0 per second; 1800 s of game time).

| Measure | Value |
|---|---|
| Stocks taken - lost | 15 - 16 |
| Virtual matches (first to 3 stocks) won - lost | 4 - 3 (57%) |
| Damage dealt / taken | 3359 / 3496 |
| Reward per minute of game time | -0.08 |
| Decisions with a button | 28.0% |
| Wu Shang / Mordex present | 98.2% / 98.2% |
| Perception + policy per decision (ms): median / p95 / max | 24.4 / 33.0 / 123.6 (budget 50) |
| of which policy (ms): median / p95 | 0.96 / 2.02 |
| Decisions over the budget | 0.18% |
| Frames whose keys were set late (realtime) | 38137 of 108012 |
| Each part once per step (advance, grab, perceive, build) | {'advance': 36004, 'grab': 36005, 'perceive': 36005, 'build': 36005} for 36004 steps |

| Block | Stocks taken - lost | Damage dealt / taken | Reward per minute |
|---|---|---|---|
| 1 | 3 - 2 | 680 / 629 | 0.30 |
| 2 | 2 - 2 | 585 / 511 | 0.15 |
| 3 | 3 - 2 | 607 / 474 | 0.47 |
| 4 | 1 - 4 | 410 / 580 | -0.94 |
| 5 | 5 - 2 | 650 / 594 | 0.71 |
| 6 | 1 - 4 | 427 / 708 | -1.16 |

Over the 6 full blocks, per minute of game time (mean ± spread between blocks; standard error of the mean):

- Reward: -0.08 ± 0.78 (standard error 0.32)
- Net stocks: -0.03 ± 0.48 (standard error 0.20)
- Damage dealt: 112 ± 23; taken: 117 ± 17
- Two policies each played for 6 such blocks differ beyond chance (about 2 standard errors of the difference) when their reward per minute differs by more than 0.90.
- Blocks follow each other without a pause, so damage carries from one into the next.
