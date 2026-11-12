# Gate P1 (armed_yolo11s-pose_1280): FAIL

2026-11-11 13:17. 149 test frames from C:\brawl_data\labelling\export_kp_final. Thresholds are provisional (research/CHECKLIST.md). A fighter is found when the predicted box of that identity has its position inside the labelled box; the error is the distance from the labelled torso point to the predicted one, or to the box centre for a candidate without points (1920 x 1080 pixels).

| Check | Value | Pass at | |
|---|---|---|---|
| wu_shang found | 1.000 | >= 0.98 | ok |
| wu_shang identity swaps | 0.000 | <= 0.01 | ok |
| wu_shang false detections (per frame) | 0.000 | <= 0.01 | ok |
| wu_shang position error p95 (px) | 15.340 | <= 12.0 | **fail** |
| wu_shang held weapon | 0.953 | >= 0.95 | ok |
| mordex found | 0.965 | >= 0.98 | **fail** |
| mordex identity swaps | 0.000 | <= 0.01 | ok |
| mordex false detections (per frame) | 0.000 | <= 0.01 | ok |
| mordex position error p95 (px) | 15.146 | <= 12.0 | **fail** |
| mordex held weapon | 0.949 | >= 0.95 | **fail** |
| weapon_spawn recall | 0.986 | >= 0.95 | ok |
| weapon_spawn precision | 0.890 | >= 0.95 | **fail** |
| weapon_thrown recall | 0.875 | >= 0.8 | ok |
| weapon_thrown precision | 0.778 | >= 0.8 | **fail** |

| Fighter | Labelled | Found | Missed | Swapped | False | Error median / p95 / max (px) |
|---|---|---|---|---|---|---|
| wu_shang | 148 | 148 | 0 | 0 | 0 | 4.6 / 15.3 / 24.7 |
| mordex | 142 | 137 | 5 | 0 | 0 | 4.0 / 15.1 / 26.0 |

| Weapon class | Labelled | Predicted | Matched | Recall | Precision | Error median (px) |
|---|---|---|---|---|---|---|
| weapon_spawn | 74 | 82 | 73 | 0.986 | 0.890 | 4.8 |
| weapon_thrown | 16 | 18 | 14 | 0.875 | 0.778 | 1.7 |

Held weapon (labelled -> predicted, found fighters):

- wu_shang_gauntlets -> wu_shang_gauntlets: 68
- wu_shang_gauntlets -> wu_shang_unarmed: 2
- wu_shang_spear -> wu_shang_gauntlets: 1
- wu_shang_spear -> wu_shang_spear: 52
- wu_shang_spear -> wu_shang_unarmed: 1
- wu_shang_unarmed -> wu_shang_gauntlets: 1
- wu_shang_unarmed -> wu_shang_spear: 2
- wu_shang_unarmed -> wu_shang_unarmed: 21
- mordex_gauntlets -> mordex_gauntlets: 43
- mordex_gauntlets -> mordex_unarmed: 3
- mordex_scythe -> mordex_scythe: 38
- mordex_unarmed -> mordex_gauntlets: 3
- mordex_unarmed -> mordex_scythe: 1
- mordex_unarmed -> mordex_unarmed: 49
