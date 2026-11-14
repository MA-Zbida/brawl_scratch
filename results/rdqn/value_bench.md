# Offline test of value learning

2026-11-14 09:23. scripts/value_bench.py. Trained on 10 human sessions, 20000 updates each; scored on 5 held-out sessions on Q(s, a_human) against the true discounted return. Lose / win AUC: the value before a stock lost / taken within 60 decisions.

| Variant | Correlation | Bias | Lose AUC | Win AUC | Sessions ranked | Headroom |
|---|---|---|---|---|---|---|
| mc regression, g 0.995 | 0.339 | -0.01 | 0.629 | 0.754 | 0.97 | 0.24 |
| mc regression, g 0.99 | 0.385 | +0.01 | 0.692 | 0.787 | 0.99 | 0.20 |
| sarsa, g 0.995, n 5 | 0.313 | -0.45 | 0.712 | 0.798 | 0.92 | 0.07 |
| double_q, g 0.995, n 5, target 2000 (the night's) | 0.253 | -0.21 | 0.682 | 0.727 | 0.93 | 0.07 |
| double_q, same, no margin loss | 0.298 | +0.20 | 0.579 | 0.885 | -0.51 | 0.11 |
| double_q, g 0.99, n 5 | 0.293 | +0.01 | 0.671 | 0.718 | 0.93 | 0.07 |
| double_q, g 0.99, n 20 | 0.351 | +0.07 | 0.669 | 0.793 | 0.87 | 0.07 |
| double_q, g 0.995, n 20, target 500 | 0.362 | +0.35 | 0.708 | 0.801 | 0.85 | 0.07 |
| double_q, g 0.99, n 20, target 500, lr 3e-5 | 0.430 | +0.09 | 0.743 | 0.826 | 0.82 | 0.03 |

Per held-out session (mean Q of the human's action against the mean true return):

- mc regression, g 0.995: 20261106_151844 (easy) Q +0.53 vs true +0.66; 20261110_194554 (hard) Q +0.22 vs true -0.34; 20261111_183313 (medium) Q +0.45 vs true +0.55; 20261111_200054 (medium) Q +0.54 vs true +0.62; 20261112_103941 (medium) Q +0.51 vs true +0.52
- mc regression, g 0.99: 20261106_151844 (easy) Q +0.30 vs true +0.34; 20261110_194554 (hard) Q +0.12 vs true -0.17; 20261111_183313 (medium) Q +0.25 vs true +0.28; 20261111_200054 (medium) Q +0.28 vs true +0.31; 20261112_103941 (medium) Q +0.26 vs true +0.26
- sarsa, g 0.995, n 5: 20261106_151844 (easy) Q +0.05 vs true +0.66; 20261110_194554 (hard) Q -0.04 vs true -0.34; 20261111_183313 (medium) Q +0.01 vs true +0.55; 20261111_200054 (medium) Q +0.05 vs true +0.62; 20261112_103941 (medium) Q +0.05 vs true +0.52
- double_q, g 0.995, n 5, target 2000 (the night's): 20261106_151844 (easy) Q +0.28 vs true +0.66; 20261110_194554 (hard) Q +0.21 vs true -0.34; 20261111_183313 (medium) Q +0.26 vs true +0.55; 20261111_200054 (medium) Q +0.28 vs true +0.62; 20261112_103941 (medium) Q +0.29 vs true +0.52
- double_q, same, no margin loss: 20261106_151844 (easy) Q +0.70 vs true +0.66; 20261110_194554 (hard) Q +0.70 vs true -0.34; 20261111_183313 (medium) Q +0.67 vs true +0.55; 20261111_200054 (medium) Q +0.65 vs true +0.62; 20261112_103941 (medium) Q +0.69 vs true +0.52
- double_q, g 0.99, n 5: 20261106_151844 (easy) Q +0.26 vs true +0.34; 20261110_194554 (hard) Q +0.20 vs true -0.17; 20261111_183313 (medium) Q +0.24 vs true +0.28; 20261111_200054 (medium) Q +0.26 vs true +0.31; 20261112_103941 (medium) Q +0.27 vs true +0.26
- double_q, g 0.99, n 20: 20261106_151844 (easy) Q +0.33 vs true +0.34; 20261110_194554 (hard) Q +0.25 vs true -0.17; 20261111_183313 (medium) Q +0.29 vs true +0.28; 20261111_200054 (medium) Q +0.31 vs true +0.31; 20261112_103941 (medium) Q +0.33 vs true +0.26
- double_q, g 0.995, n 20, target 500: 20261106_151844 (easy) Q +0.87 vs true +0.66; 20261110_194554 (hard) Q +0.75 vs true -0.34; 20261111_183313 (medium) Q +0.81 vs true +0.55; 20261111_200054 (medium) Q +0.82 vs true +0.62; 20261112_103941 (medium) Q +0.86 vs true +0.52
- double_q, g 0.99, n 20, target 500, lr 3e-5: 20261106_151844 (easy) Q +0.36 vs true +0.34; 20261110_194554 (hard) Q +0.28 vs true -0.17; 20261111_183313 (medium) Q +0.32 vs true +0.28; 20261111_200054 (medium) Q +0.31 vs true +0.31; 20261112_103941 (medium) Q +0.35 vs true +0.26
