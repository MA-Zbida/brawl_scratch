# s1_ppo: summary

2026-11-12 14:57. Reward per minute of game time in blocks of 10 rollouts (mean ± standard error over the block's rollouts). scripts/analyze.py.

## realtime-bc / seed0

60 rollouts, 60 minutes of game time. Stocks 22 - 42; damage 4423 / 6016.

| Block (minutes) | Reward per minute | Net stocks per minute |
|---|---|---|
| 0-10 | -0.30 ± 0.40 | -0.20 |
| 10-20 | -0.92 ± 0.34 | -0.70 |
| 20-30 | -0.87 ± 0.36 | -0.40 |
| 30-40 | -0.32 ± 0.37 | -0.20 |
| 40-50 | -0.46 ± 0.31 | -0.30 |
| 50-60 | -0.72 ± 0.28 | -0.20 |

- Change: -0.42 ± 0.48 from the first block to the last: within 2 standard errors, so not distinguishable from no change.
- Updates: 0.30 s each (median), 0.50% of the run; during them 37 damage taken and 0 stocks lost.
- Decisions over the time budget: 0.09%; pauses for lost focus: 0.

