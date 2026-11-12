# Behaviour cloning: medium_v1

2026-11-11 18:55. Levels: medium; stretches kept: won; k = 3; network 256,256; seed 0. 17 stretches, 9296 decisions to train on, 2293 held out (whole stretches). Best epoch 67.

| Measure | Held out | Train |
|---|---|---|
| Loss (negative log likelihood) | 1.302 | 1.063 |
| Direction right | 0.765 | 0.793 |
| Button right | 0.816 | 0.850 |
| Both right | 0.629 | 0.676 |
| Baseline: repeat the previous action | 0.610 | 0.638 |
| Both right when the human changes action | 0.170 | 0.174 |
| New button presses predicted | 0.114 | 0.120 |
| Share of decisions with a button, human | 0.373 | 0.331 |
| Share of decisions with a button, policy's most likely action | 0.303 | 0.291 |

Recordings:

- 20261111_183313: 17 stocks won, 4 lost; kept 11589 of 13702 decisions (9.7 of 11.4 min)
