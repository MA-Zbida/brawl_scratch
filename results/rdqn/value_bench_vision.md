# Does the advisor value outcomes better with pictures?

2026-11-17 11:28. scripts/value_bench_vision.py. The advisor of rdqn_medium_v3 (sarsa, gamma 0.99, n 20, target every 500), 60 updates, on 9 human Medium sessions with crops, 1 folds (a third held out in turn). Mean over folds; in brackets the mean difference from `vector` in the same fold, and its spread.

| Input | Correlation | Bias | Lose AUC | Win AUC |
|---|---|---|---|---|
| vector | 0.144 | -0.27 | 0.563 | 0.638 |
| vector + crops | 0.166 (+0.022 ± 0.000) | -0.27 (+0.00 ± 0.00) | 0.488 (-0.075 ± 0.000) | 0.665 (+0.027 ± 0.000) |
