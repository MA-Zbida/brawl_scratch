# Play: scripted_mordex_medium_realtime

2026-11-13 12:09. Policy scripted (sampled), bot medium, realtime, k = 3, 1 x 60 min. 72006 decisions in 3600 s (20.0 per second; 3600 s of game time).

| Measure | Value |
|---|---|
| Stocks taken - lost | 46 - 16 |
| Virtual matches (first to 3 stocks) won - lost | 15 - 0 (100%) |
| Damage dealt / taken | 9674 / 2754 |
| Reward per minute of game time | 1.65 |
| Decisions with a button | 14.5% |
| Wu Shang / Mordex present | 99.1% / 97.0% |
| Perception + policy per decision (ms): median / p95 / max | 28.7 / 37.0 / 147.1 (budget 50) |
| of which policy (ms): median / p95 | 0.57 / 1.07 |
| Decisions over the budget | 0.27% |
| Frames whose keys were set late (realtime) | 87316 of 216018 |
| Each part once per step (advance, grab, perceive, build) | {'advance': 72006, 'grab': 72007, 'perceive': 72007, 'build': 72007} for 72006 steps |
