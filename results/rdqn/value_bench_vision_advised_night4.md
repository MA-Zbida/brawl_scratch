# Does the advisor value outcomes better with pictures?

2026-11-19 16:30. scripts/value_bench_vision.py. The advisor of rdqn_medium_v3 (sarsa, gamma 0.99, n 20, target every 500), 20000 updates, on the agent's own play in runs/advised_night4/default/seed0 (8 segments of time; each fold holds out two, half the run apart), 700000 decisions. Mean over folds; in brackets the mean difference from `vector` in the same fold, and its spread.

| Input | Correlation | Bias | Lose AUC | Win AUC |
|---|---|---|---|---|
| vector | 0.442 | 0.01 | 0.771 | 0.846 |
| vector + crops | 0.457 (+0.015 ± 0.003) | 0.00 (-0.01 ± 0.01) | 0.771 (+0.001 ± 0.007) | 0.844 (-0.002 ± 0.005) |
| vector + crops, lr 1e-4 | 0.420 (-0.022 ± 0.007) | -0.03 (-0.03 ± 0.03) | 0.776 (+0.006 ± 0.006) | 0.794 (-0.051 ± 0.024) |

Run on Kaggle (GPU, Python 3.13): about 10 ms per update for the vector, 205 ms with crops.
