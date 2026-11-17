# Does the advisor value outcomes better with pictures?

2026-11-17 12:42. scripts/value_bench_vision.py. The advisor of rdqn_medium_v3 (sarsa, gamma 0.99, n 20, target every 500), 10000 updates, on 9 human Medium sessions with crops, 3 folds (a third held out in turn). Mean over folds; in brackets the mean difference from `vector` in the same fold, and its spread.

| Input | Correlation | Bias | Lose AUC | Win AUC |
|---|---|---|---|---|
| vector | 0.477 | -0.02 | 0.725 | 0.862 |
| vector + crops, lr 1e-4 | 0.414 (-0.063 ± 0.021) | -0.00 (+0.02 ± 0.03) | 0.608 (-0.117 ± 0.015) | 0.828 (-0.034 ± 0.027) |
