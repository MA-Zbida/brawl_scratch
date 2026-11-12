# Behaviour cloning: medium_v4

2026-11-12 12:28. Levels: medium; stretches kept: won; k = 3; network 256,256; seed 0. 166 stretches, 97647 decisions to train on, 16242 held out (whole stretches). Best epoch 56.

| Measure | Held out | Train |
|---|---|---|
| Loss (negative log likelihood) | 0.898 | 0.833 |
| Direction right | 0.829 | 0.833 |
| Button right | 0.864 | 0.874 |
| Both right | 0.720 | 0.731 |
| Baseline: repeat the previous action | 0.699 | 0.686 |
| Both right when the human changes action | 0.224 | 0.251 |
| New button presses predicted | 0.112 | 0.155 |
| Share of decisions with a button, human | 0.288 | 0.290 |
| Share of decisions with a button, policy's most likely action | 0.231 | 0.238 |

Recordings:

- 20261111_183313: 17 stocks won, 4 lost; kept 11589 of 13702 decisions (9.7 of 11.4 min)
- 20261111_190954: 22 stocks won, 6 lost; kept 15895 of 17981 decisions (13.2 of 15.0 min)
- 20261111_193322: 24 stocks won, 5 lost; kept 15770 of 17998 decisions (13.1 of 15.0 min)
- 20261111_200054: 16 stocks won, 2 lost; kept 10679 of 11727 decisions (8.9 of 9.8 min)
- 20261111_201232: 11 stocks won, 2 lost; kept 8032 of 9090 decisions (6.7 of 7.6 min)
- 20261111_202651: 16 stocks won, 4 lost; kept 10841 of 13128 decisions (9.0 of 10.9 min)
- 20261112_103941: 21 stocks won, 6 lost; kept 14961 of 17974 decisions (12.5 of 15.0 min)
- 20261112_110421: 19 stocks won, 3 lost; kept 13426 of 15031 decisions (11.2 of 12.5 min)
- 20261112_113633: 20 stocks won, 4 lost; kept 12696 of 14968 decisions (10.6 of 12.5 min)
