# Play: eval_advised

2026-11-15 10:30. Policy runs/rdqn_medium_v3/default/seed0/best.pt (sampled), bot medium, realtime, k = 3, 6 x 5 min. 36004 decisions in 1800 s (20.0 per second; 1800 s of game time).

| Measure | Value |
|---|---|
| Stocks taken - lost | 20 - 14 |
| Virtual matches (first to 3 stocks) won - lost | 6 - 1 (86%) |
| Damage dealt / taken | 4060 / 3158 |
| Reward per minute of game time | 0.50 |
| Decisions with a button | 26.7% |
| Wu Shang / Mordex present | 98.4% / 97.7% |
| Perception + policy per decision (ms): median / p95 / max | 25.0 / 33.6 / 152.9 (budget 50) |
| of which policy (ms): median / p95 | 1.85 / 3.86 |
| Decisions over the budget | 0.15% |
| Frames whose keys were set late (realtime) | 38529 of 108012 |
| Each part once per step (advance, grab, perceive, build) | {'advance': 36004, 'grab': 36005, 'perceive': 36005, 'build': 36005} for 36004 steps |

| Block | Stocks taken - lost | Damage dealt / taken | Reward per minute |
|---|---|---|---|
| 1 | 3 - 2 | 668 / 506 | 0.52 |
| 2 | 4 - 2 | 800 / 494 | 1.01 |
| 3 | 3 - 3 | 563 / 539 | 0.05 |
| 4 | 5 - 2 | 854 / 550 | 1.21 |
| 5 | 2 - 2 | 609 / 507 | 0.20 |
| 6 | 3 - 3 | 566 / 560 | 0.01 |

Over the 6 full blocks, per minute of game time (mean ± spread between blocks; standard error of the mean):

- Reward: 0.50 ± 0.51 (standard error 0.21)
- Net stocks: 0.20 ± 0.25 (standard error 0.10)
- Damage dealt: 135 ± 25; taken: 105 ± 5
- Two policies each played for 6 such blocks differ beyond chance (about 2 standard errors of the difference) when their reward per minute differs by more than 0.59.
- Blocks follow each other without a pause, so damage carries from one into the next.
