# Gate P1 (armed_yolo11s-pose_960): FAIL

2026-11-11 13:18. 149 test frames from C:\brawl_data\labelling\export_kp_final. Thresholds are provisional (research/CHECKLIST.md). A fighter is found when the predicted box of that identity has its position inside the labelled box; the error is the distance from the labelled torso point to the predicted one, or to the box centre for a candidate without points (1920 x 1080 pixels).

| Check | Value | Pass at | |
|---|---|---|---|
| wu_shang found | 0.993 | >= 0.98 | ok |
| wu_shang identity swaps | 0.000 | <= 0.01 | ok |
| wu_shang false detections (per frame) | 0.000 | <= 0.01 | ok |
| wu_shang position error p95 (px) | 13.617 | <= 12.0 | **fail** |
| wu_shang held weapon | 0.939 | >= 0.95 | **fail** |
| mordex found | 0.958 | >= 0.98 | **fail** |
| mordex identity swaps | 0.000 | <= 0.01 | ok |
| mordex false detections (per frame) | 0.000 | <= 0.01 | ok |
| mordex position error p95 (px) | 13.538 | <= 12.0 | **fail** |
| mordex held weapon | 0.926 | >= 0.95 | **fail** |
| weapon_spawn recall | 1.000 | >= 0.95 | ok |
| weapon_spawn precision | 0.925 | >= 0.95 | **fail** |
| weapon_thrown recall | 0.812 | >= 0.8 | ok |
| weapon_thrown precision | 0.812 | >= 0.8 | ok |

| Fighter | Labelled | Found | Missed | Swapped | False | Error median / p95 / max (px) |
|---|---|---|---|---|---|---|
| wu_shang | 148 | 147 | 1 | 0 | 0 | 5.0 / 13.6 / 22.8 |
| mordex | 142 | 136 | 6 | 0 | 0 | 3.5 / 13.5 / 25.3 |

| Weapon class | Labelled | Predicted | Matched | Recall | Precision | Error median (px) |
|---|---|---|---|---|---|---|
| weapon_spawn | 74 | 80 | 74 | 1.000 | 0.925 | 4.6 |
| weapon_thrown | 16 | 16 | 13 | 0.812 | 0.812 | 1.3 |

Held weapon (labelled -> predicted, found fighters):

- wu_shang_gauntlets -> wu_shang_gauntlets: 67
- wu_shang_gauntlets -> wu_shang_unarmed: 2
- wu_shang_spear -> wu_shang_spear: 52
- wu_shang_spear -> wu_shang_unarmed: 2
- wu_shang_unarmed -> wu_shang_gauntlets: 3
- wu_shang_unarmed -> wu_shang_spear: 2
- wu_shang_unarmed -> wu_shang_unarmed: 19
- mordex_gauntlets -> mordex_gauntlets: 43
- mordex_gauntlets -> mordex_unarmed: 3
- mordex_scythe -> mordex_scythe: 37
- mordex_scythe -> mordex_unarmed: 1
- mordex_unarmed -> mordex_gauntlets: 6
- mordex_unarmed -> mordex_unarmed: 46
