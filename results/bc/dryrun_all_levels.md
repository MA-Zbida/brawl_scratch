# Behaviour cloning: dryrun_all_levels

2026-11-11 18:16. Levels: all; stretches kept: won; k = 3; network 256,256; seed 0. 58 stretches, 32061 decisions to train on, 5396 held out (whole stretches). Best epoch 52.

| Measure | Held out | Train |
|---|---|---|
| Loss (negative log likelihood) | 1.037 | 0.896 |
| Direction right | 0.815 | 0.827 |
| Button right | 0.860 | 0.880 |
| Both right | 0.704 | 0.733 |
| Baseline: repeat the previous action | 0.702 | 0.706 |
| Both right when the human changes action | 0.155 | 0.178 |
| New button presses predicted | 0.028 | 0.061 |
| Share of decisions with a button, human | 0.290 | 0.256 |
| Share of decisions with a button, policy's most likely action | 0.221 | 0.200 |

Recordings:

- 20261106_151844: 17 stocks won, 3 lost; kept 10757 of 11996 decisions (9.0 of 10.0 min)
- 20261106_155225: 6 stocks won, 10 lost; kept 2935 of 10542 decisions (2.4 of 8.8 min)
- 20261109_184230: 26 stocks won, 5 lost; kept 16913 of 19816 decisions (14.1 of 16.5 min)
- 20261109_211132: not prepared (scripts/prepare_demos.py)
- 20261110_194554: 3 stocks won, 6 lost; kept 2629 of 6915 decisions (2.2 of 5.8 min)
- 20261110_195213: 4 stocks won, 6 lost; kept 3382 of 7669 decisions (2.8 of 6.4 min)
- 20261110_195859: 2 stocks won, 9 lost; kept 841 of 7656 decisions (0.7 of 6.4 min)
