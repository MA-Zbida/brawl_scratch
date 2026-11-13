# Gates S1 and S2 probes: what memory and a picture of the fighters add

2026-11-13 17:21. medium sessions held out whole (3 folds), trained until the stop score stops improving. Each cell: mean ± spread over the runs, then (for inputs other than `vector`) the mean difference from `vector` in the same fold and seed ± its spread. Lower is better except AUC. scripts/probe_vision.py.

- Wu Shang is hit within 0.5 s after 5.1% of decisions.
- Guessing that Mordex keeps his velocity misses by 118.5 px at 0.25 s and 250.0 px at 0.5 s.
- Choosing among the 54 actions at random gives an action NLL of 3.99.
- Runs interrupted by the time limit (their best epoch so far is used): 0 of 60.

| Input | Runs | Hit within 0.5 s: AUC | Mordex in 0.25 s (px) | Mordex in 0.5 s (px) | Action NLL when the human changes | Action NLL | Best epoch | Cost per decision here, CPU / GPU (ms) |
|---|---|---|---|---|---|---|---|---|
| vector, no timers | 6 | 0.734 ± 0.008 (-0.022 ± 0.012) | 74.2 ± 0.6 (+0.6 ± 0.2) | 132.9 ± 1.3 (+1.6 ± 0.3) | 2.388 ± 0.046 (+0.017 ± 0.026) | 0.991 ± 0.045 (+0.010 ± 0.005) | 69 | 0.13 / 0.27 |
| vector | 6 | 0.756 ± 0.017 | 73.6 ± 0.7 | 131.4 ± 1.3 | 2.370 ± 0.063 | 0.981 ± 0.045 | 64 | 0.12 / 0.40 |
| vector, GRU | 6 | 0.769 ± 0.026 (+0.013 ± 0.020) | 71.7 ± 0.5 (-1.9 ± 0.8) | 129.0 ± 1.3 (-2.4 ± 1.2) | 2.248 ± 0.075 (-0.122 ± 0.055) | 0.948 ± 0.051 (-0.033 ± 0.009) | 46 | 0.28 / 0.63 |
| both, 1 frame | 6 | 0.767 ± 0.006 (+0.010 ± 0.017) | 68.2 ± 0.5 (-5.4 ± 0.7) | 125.0 ± 1.0 (-6.4 ± 0.8) | 2.456 ± 0.072 (+0.085 ± 0.051) | 1.026 ± 0.052 (+0.045 ± 0.015) | 44 | 1.31 / 0.57 |
| both, 4 decisions (0.15 s) | 6 | 0.755 ± 0.008 (-0.001 ± 0.017) | 69.5 ± 0.6 (-4.2 ± 0.7) | 127.2 ± 1.3 (-4.2 ± 1.6) | 2.493 ± 0.066 (+0.122 ± 0.043) | 1.034 ± 0.050 (+0.053 ± 0.009) | 32 | 1.37 / 0.55 |
| both, 1 frame, GRU | 6 | 0.768 ± 0.008 (+0.012 ± 0.013) | 67.9 ± 0.8 (-5.7 ± 1.0) | 124.6 ± 1.5 (-6.8 ± 1.2) | 2.350 ± 0.055 (-0.021 ± 0.032) | 0.989 ± 0.050 (+0.008 ± 0.009) | 26 | 1.60 / 0.82 |
| vector + reader states | 6 | 0.759 ± 0.014 (+0.003 ± 0.007) | 73.4 ± 0.7 (-0.3 ± 0.4) | 131.1 ± 1.4 (-0.3 ± 0.7) | 2.363 ± 0.071 (-0.007 ± 0.015) | 0.983 ± 0.047 (+0.002 ± 0.006) | 72 | 0.12 / 0.38 |
| vector + reader states, GRU | 6 | 0.772 ± 0.012 (+0.016 ± 0.012) | 71.5 ± 0.7 (-2.2 ± 1.1) | 128.6 ± 1.7 (-2.7 ± 1.6) | 2.256 ± 0.083 (-0.114 ± 0.047) | 0.950 ± 0.052 (-0.030 ± 0.009) | 42 | 0.25 / 0.61 |
| vector + reader memory | 6 | 0.748 ± 0.014 (-0.008 ± 0.010) | 73.2 ± 0.5 (-0.4 ± 0.7) | 133.1 ± 1.2 (+1.7 ± 1.6) | 2.464 ± 0.043 (+0.094 ± 0.044) | 1.025 ± 0.048 (+0.045 ± 0.011) | 35 | 0.10 / 0.37 |
| vector + reader memory, GRU | 6 | 0.744 ± 0.023 (-0.013 ± 0.016) | 72.4 ± 0.3 (-1.2 ± 0.7) | 132.4 ± 0.6 (+1.0 ± 0.8) | 2.397 ± 0.067 (+0.027 ± 0.031) | 1.007 ± 0.057 (+0.026 ± 0.013) | 24 | 0.24 / 0.60 |

A candidate earns its place when its difference from `vector` is larger than that difference's spread.
