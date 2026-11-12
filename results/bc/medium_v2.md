# Behaviour cloning: medium_v2

2026-11-11 21:15. Levels: medium; stretches kept: won; k = 3; network 256,256; seed 0. 106 stretches, 62455 decisions to train on, 10351 held out (whole stretches). Best epoch 50.

| Measure | Held out | Train |
|---|---|---|
| Loss (negative log likelihood) | 0.989 | 0.888 |
| Direction right | 0.819 | 0.827 |
| Button right | 0.851 | 0.870 |
| Both right | 0.701 | 0.722 |
| Baseline: repeat the previous action | 0.689 | 0.681 |
| Both right when the human changes action | 0.194 | 0.224 |
| New button presses predicted | 0.133 | 0.143 |
| Share of decisions with a button, human | 0.303 | 0.292 |
| Share of decisions with a button, policy's most likely action | 0.253 | 0.248 |

Recordings:

- 20261111_183313: 17 stocks won, 4 lost; kept 11589 of 13702 decisions (9.7 of 11.4 min)
- 20261111_190954: 22 stocks won, 6 lost; kept 15895 of 17981 decisions (13.2 of 15.0 min)
- 20261111_193322: 24 stocks won, 5 lost; kept 15770 of 17998 decisions (13.1 of 15.0 min)
- 20261111_200054: 16 stocks won, 2 lost; kept 10679 of 11727 decisions (8.9 of 9.8 min)
- 20261111_201232: 11 stocks won, 2 lost; kept 8032 of 9090 decisions (6.7 of 7.6 min)
- 20261111_202651: 16 stocks won, 4 lost; kept 10841 of 13128 decisions (9.0 of 10.9 min)
