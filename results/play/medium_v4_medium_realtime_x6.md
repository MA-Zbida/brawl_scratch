# Play: medium_v4_medium_realtime_x6

2026-11-12 13:09. Policy runs/bc/medium_v4/policy.zip (sampled), bot medium, realtime, k = 3, 6 x 5 min. 36014 decisions in 1800 s (20.0 per second; 1801 s of game time).

| Measure | Value |
|---|---|
| Stocks taken - lost | 16 - 17 |
| Virtual matches (first to 3 stocks) won - lost | 3 - 4 (43%) |
| Damage dealt / taken | 3210 / 3191 |
| Reward per minute of game time | -0.03 |
| Decisions with a button | 27.7% |
| Wu Shang / Mordex present | 98.1% / 98.3% |
| Perception + policy per decision (ms): median / p95 / max | 26.5 / 32.8 / 1219.0 (budget 50) |
| of which policy (ms): median / p95 | 0.90 / 1.78 |
| Decisions over the budget | 0.17% |
| Frames whose keys were set late (realtime) | 38454 of 108042 |
| Each part once per step (advance, grab, perceive, build) | {'advance': 36014, 'grab': 36015, 'perceive': 36015, 'build': 36015} for 36014 steps |

| Block | Stocks taken - lost | Damage dealt / taken | Reward per minute |
|---|---|---|---|
| 1 | 2 - 5 | 385 / 681 | -1.19 |
| 2 | 3 - 3 | 727 / 606 | 0.24 |
| 3 | 3 - 2 | 557 / 498 | 0.32 |
| 4 | 3 - 3 | 564 / 480 | 0.17 |
| 5 | 3 - 2 | 545 / 483 | 0.32 |
| 6 | 2 - 2 | 432 / 443 | -0.02 |

Over the 6 full blocks, per minute of game time (mean ± spread between blocks; standard error of the mean):

- Reward: -0.03 ± 0.58 (standard error 0.24)
- Net stocks: -0.03 ± 0.29 (standard error 0.12)
- Damage dealt: 107 ± 24; taken: 106 ± 18
- Two policies each played for 6 such blocks differ beyond chance (about 2 standard errors of the difference) when their reward per minute differs by more than 0.67.
- Blocks follow each other without a pause, so damage carries from one into the next.
