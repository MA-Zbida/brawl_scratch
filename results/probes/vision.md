# Gates S1 and S2 probes: what memory and a picture of the fighters add

2026-11-12 13:41. medium sessions held out whole (3 folds), trained until the stop score stops improving. Each cell: mean ± spread over the runs, then (for inputs other than `vector`) the mean difference from `vector` in the same fold and seed ± its spread. Lower is better except AUC. scripts/probe_vision.py.

- Wu Shang is hit within 0.5 s after 5.1% of decisions.
- Guessing that Mordex keeps his velocity misses by 118.5 px at 0.25 s and 250.0 px at 0.5 s.
- Choosing among the 54 actions at random gives an action NLL of 3.99.
- Runs interrupted by the time limit (their best epoch so far is used): 0 of 36.

| Input | Runs | Hit within 0.5 s: AUC | Mordex in 0.25 s (px) | Mordex in 0.5 s (px) | Action NLL when the human changes | Action NLL | Best epoch | Cost per decision here, CPU / GPU (ms) |
|---|---|---|---|---|---|---|---|---|
| vector, no timers | 6 | 0.734 ± 0.008 (-0.027 ± 0.012) | 74.2 ± 0.6 (+0.7 ± 0.4) | 132.9 ± 1.3 (+1.7 ± 0.5) | 2.388 ± 0.046 (+0.026 ± 0.033) | 0.991 ± 0.045 (+0.006 ± 0.010) | 69 | 0.13 / 0.27 |
| vector | 6 | 0.761 ± 0.014 | 73.5 ± 0.6 | 131.3 ± 1.4 | 2.361 ± 0.073 | 0.985 ± 0.052 | 78 | 0.13 / 0.25 |
| vector, GRU | 6 | 0.770 ± 0.021 (+0.009 ± 0.016) | 71.6 ± 0.8 (-1.9 ± 0.6) | 128.9 ± 1.6 (-2.4 ± 1.1) | 2.256 ± 0.044 (-0.106 ± 0.048) | 0.945 ± 0.048 (-0.039 ± 0.010) | 46 | 0.33 / 0.48 |
| both, 1 frame | 6 | 0.767 ± 0.006 (+0.005 ± 0.015) | 68.2 ± 0.5 (-5.3 ± 0.4) | 125.0 ± 1.0 (-6.3 ± 0.6) | 2.456 ± 0.072 (+0.094 ± 0.051) | 1.026 ± 0.052 (+0.041 ± 0.012) | 44 | 1.31 / 0.57 |
| both, 4 decisions (0.15 s) | 6 | 0.755 ± 0.008 (-0.006 ± 0.014) | 69.5 ± 0.6 (-4.1 ± 0.7) | 127.2 ± 1.3 (-4.1 ± 1.6) | 2.493 ± 0.066 (+0.131 ± 0.042) | 1.034 ± 0.050 (+0.050 ± 0.011) | 32 | 1.37 / 0.55 |
| both, 1 frame, GRU | 6 | 0.768 ± 0.008 (+0.007 ± 0.011) | 67.9 ± 0.8 (-5.6 ± 0.8) | 124.6 ± 1.5 (-6.7 ± 0.9) | 2.350 ± 0.055 (-0.012 ± 0.040) | 0.989 ± 0.050 (+0.004 ± 0.009) | 26 | 1.60 / 0.82 |

A candidate earns its place when its difference from `vector` is larger than that difference's spread.
