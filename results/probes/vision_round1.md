# Gate S2 probes: what a picture of the fighters adds

2026-11-11 22:19. 6 medium sessions, held out two at a time (3 folds); mean ± spread over the folds. Lower is better except AUC. scripts/probe_vision.py.

- Wu Shang is hit within 0.5 s after 5.0% of decisions (log loss of always saying that rate: 0.1978).
- Guessing that Mordex keeps his velocity misses by 117.1 px at 0.25 s and 246.4 px at 0.5 s.

| Input | Folds | Hit within 0.5 s: log loss | AUC | Mordex in 0.25 s (px) | in 0.5 s (px) | Action NLL when the human changes | Action NLL | Cost per decision, CPU / GPU (ms) |
|---|---|---|---|---|---|---|---|---|
| vector | 3 | 0.1962 ± 0.0103 | 0.671 ± 0.006 | 81.2 ± 1.9 | 140.2 ± 2.2 | 2.615 ± 0.057 | 1.089 | 0.08 / 0.31 |
| Mordex, 1 frame | 3 | 0.1962 ± 0.0111 | 0.677 ± 0.012 | 74.6 ± 2.3 | 133.4 ± 2.1 | 2.654 ± 0.058 | 1.105 | 1.22 / 0.99 |
| both, 1 frame | 3 | 0.1943 ± 0.0113 | 0.701 ± 0.015 | 73.6 ± 2.2 | 131.7 ± 2.2 | 2.628 ± 0.058 | 1.101 | 1.10 / 0.73 |
| both, 2 frames 1/60 s apart | 3 | 0.1938 ± 0.0113 | 0.702 ± 0.010 | 72.2 ± 1.3 | 129.9 ± 0.8 | 2.601 ± 0.071 | 1.087 | 1.25 / 1.12 |
| both, 4 decisions (0.15 s) | 3 | 0.1959 ± 0.0114 | 0.689 ± 0.010 | 72.2 ± 0.8 | 130.7 ± 1.5 | 2.593 ± 0.061 | 1.089 | 1.37 / 0.74 |
| both, 4 decisions, 64 px | 3 | 0.1967 ± 0.0115 | 0.674 ± 0.015 | 74.6 ± 1.5 | 133.5 ± 2.1 | 2.640 ± 0.068 | 1.102 | 0.69 / 0.93 |
| crops only, both, 4 decisions | 3 | 0.1938 ± 0.0110 | 0.696 ± 0.008 | 95.3 ± 3.5 | 185.4 ± 4.9 | 2.830 ± 0.128 | 2.297 | 1.43 / 0.92 |

A candidate earns its place when it beats `vector` by more than the spread between folds.
