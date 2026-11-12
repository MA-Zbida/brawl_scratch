# Gate P1 (old_yolo): FAIL

2026-11-10 12:18. 149 test frames from C:\brawl_data\labelling\export_kp. Thresholds are provisional (research/CHECKLIST.md). A fighter is found when the predicted box of that identity has its position inside the labelled box; the error is the distance from the labelled torso point to the predicted one, or to the box centre for a candidate without points (1920 x 1080 pixels).

| Check | Value | Pass at | |
|---|---|---|---|
| wu_shang found | 0.986 | >= 0.98 | ok |
| wu_shang identity swaps | 0.000 | <= 0.01 | ok |
| wu_shang false detections (per frame) | 0.000 | <= 0.01 | ok |
| wu_shang position error p95 (px) | 26.068 | <= 12.0 | **fail** |
| mordex found | 0.880 | >= 0.98 | **fail** |
| mordex identity swaps | 0.007 | <= 0.01 | ok |
| mordex false detections (per frame) | 0.013 | <= 0.01 | **fail** |
| mordex position error p95 (px) | 21.686 | <= 12.0 | **fail** |
| weapon_spawn recall | 0.878 | >= 0.95 | **fail** |
| weapon_spawn precision | 0.985 | >= 0.95 | ok |
| weapon_thrown recall | 0.000 | >= 0.8 | **fail** |
| weapon_thrown precision | 0.000 | >= 0.8 | **fail** |

| Fighter | Labelled | Found | Missed | Swapped | False | Error median / p95 / max (px) |
|---|---|---|---|---|---|---|
| wu_shang | 148 | 146 | 2 | 0 | 0 | 6.2 / 26.1 / 38.7 |
| mordex | 142 | 125 | 16 | 1 | 2 | 2.4 / 21.7 / 33.7 |

| Weapon class | Labelled | Predicted | Matched | Recall | Precision | Error median (px) |
|---|---|---|---|---|---|---|
| weapon_spawn | 74 | 66 | 65 | 0.878 | 0.985 | 1.6 |
| weapon_thrown | 15 | 0 | 0 | 0.000 | 0.000 | nan |

This candidate does not read the held weapon; the held-weapon checks do not apply.
