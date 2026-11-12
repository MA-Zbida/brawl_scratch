# Gate P1 (armed_yolo11s-pose_960): FAIL

2026-11-10 18:25. 149 test frames from C:\brawl_data\labelling\export_kp. Thresholds are provisional (research/CHECKLIST.md). A fighter is found when the predicted box of that identity has its position inside the labelled box; the error is the distance from the labelled torso point to the predicted one, or to the box centre for a candidate without points (1920 x 1080 pixels).

| Check | Value | Pass at | |
|---|---|---|---|
| wu_shang found | 0.993 | >= 0.98 | ok |
| wu_shang identity swaps | 0.000 | <= 0.01 | ok |
| wu_shang false detections (per frame) | 0.000 | <= 0.01 | ok |
| wu_shang position error p95 (px) | 17.789 | <= 12.0 | **fail** |
| wu_shang held weapon | 0.912 | >= 0.95 | **fail** |
| mordex found | 0.944 | >= 0.98 | **fail** |
| mordex identity swaps | 0.000 | <= 0.01 | ok |
| mordex false detections (per frame) | 0.007 | <= 0.01 | ok |
| mordex position error p95 (px) | 16.063 | <= 12.0 | **fail** |
| mordex held weapon | 0.866 | >= 0.95 | **fail** |
| weapon_spawn recall | 1.000 | >= 0.95 | ok |
| weapon_spawn precision | 0.871 | >= 0.95 | **fail** |
| weapon_thrown recall | 0.733 | >= 0.8 | **fail** |
| weapon_thrown precision | 0.314 | >= 0.8 | **fail** |

| Fighter | Labelled | Found | Missed | Swapped | False | Error median / p95 / max (px) |
|---|---|---|---|---|---|---|
| wu_shang | 148 | 147 | 1 | 0 | 0 | 5.2 / 17.8 / 35.9 |
| mordex | 142 | 134 | 8 | 0 | 1 | 4.1 / 16.1 / 28.6 |

| Weapon class | Labelled | Predicted | Matched | Recall | Precision | Error median (px) |
|---|---|---|---|---|---|---|
| weapon_spawn | 74 | 85 | 74 | 1.000 | 0.871 | 4.8 |
| weapon_thrown | 15 | 35 | 11 | 0.733 | 0.314 | 1.9 |

Held weapon (labelled -> predicted, found fighters):

- wu_shang_gauntlets -> wu_shang_gauntlets: 67
- wu_shang_gauntlets -> wu_shang_unarmed: 3
- wu_shang_spear -> wu_shang_gauntlets: 2
- wu_shang_spear -> wu_shang_spear: 51
- wu_shang_unarmed -> wu_shang_gauntlets: 5
- wu_shang_unarmed -> wu_shang_spear: 3
- wu_shang_unarmed -> wu_shang_unarmed: 16
- mordex_gauntlets -> mordex_gauntlets: 40
- mordex_gauntlets -> mordex_unarmed: 4
- mordex_scythe -> mordex_gauntlets: 1
- mordex_scythe -> mordex_scythe: 36
- mordex_scythe -> mordex_unarmed: 1
- mordex_unarmed -> mordex_gauntlets: 12
- mordex_unarmed -> mordex_unarmed: 40
