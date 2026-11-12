# Gate P1 (identity_yolo11s_1280): FAIL

2026-11-10 09:54. 149 test frames from C:\brawl_data\labelling\export. Thresholds are provisional (research/CHECKLIST.md). A fighter is found when the predicted box of that identity has its centre inside the labelled box; the error is the distance between centres (1920 x 1080 pixels).

| Check | Value | Pass at | |
|---|---|---|---|
| wu_shang found | 0.986 | >= 0.98 | ok |
| wu_shang identity swaps | 0.000 | <= 0.01 | ok |
| wu_shang false detections (per frame) | 0.000 | <= 0.01 | ok |
| wu_shang centre error p95 (px) | 24.485 | <= 12.0 | **fail** |
| mordex found | 0.944 | >= 0.98 | **fail** |
| mordex identity swaps | 0.028 | <= 0.01 | **fail** |
| mordex false detections (per frame) | 0.007 | <= 0.01 | ok |
| mordex centre error p95 (px) | 20.793 | <= 12.0 | **fail** |
| weapon_spawn recall | 0.972 | >= 0.95 | ok |
| weapon_spawn precision | 0.676 | >= 0.95 | **fail** |
| weapon_thrown recall | 0.667 | >= 0.8 | **fail** |
| weapon_thrown precision | 0.136 | >= 0.8 | **fail** |

| Fighter | Labelled | Found | Missed | Swapped | False | Error median / p95 / max (px) |
|---|---|---|---|---|---|---|
| wu_shang | 148 | 146 | 2 | 0 | 0 | 5.3 / 24.5 / 33.4 |
| mordex | 142 | 134 | 5 | 4 | 1 | 3.2 / 20.8 / 27.6 |

| Weapon class | Labelled | Predicted | Matched | Recall | Precision | Error median (px) |
|---|---|---|---|---|---|---|
| weapon_spawn | 71 | 102 | 69 | 0.972 | 0.676 | 5.6 |
| weapon_thrown | 9 | 44 | 6 | 0.667 | 0.136 | 1.8 |

This candidate does not read the held weapon; the held-weapon checks do not apply.
