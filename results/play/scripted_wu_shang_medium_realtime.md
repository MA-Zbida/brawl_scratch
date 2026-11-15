# Play: scripted_wu_shang_medium_realtime

2026-11-13 13:13. Policy scripted (sampled), bot medium, realtime, k = 3, 1 x 60 min. 72005 decisions in 3600 s (20.0 per second; 3600 s of game time).

| Measure | Value |
|---|---|
| Stocks taken - lost | 15 - 45 |
| Virtual matches (first to 3 stocks) won - lost | 0 - 15 (0%) |
| Damage dealt / taken | 2997 / 8797 |
| Reward per minute of game time | -1.47 |
| Decisions with a button | 13.9% |
| Wu Shang / Mordex present | 97.5% / 99.2% |
| Perception + policy per decision (ms): median / p95 / max | 29.8 / 38.9 / 124.4 (budget 50) |
| of which policy (ms): median / p95 | 0.61 / 1.16 |
| Decisions over the budget | 0.33% |
| Frames whose keys were set late (realtime) | 93596 of 216015 |
| Each part once per step (advance, grab, perceive, build) | {'advance': 72005, 'grab': 72006, 'perceive': 72006, 'build': 72006} for 72005 steps |
