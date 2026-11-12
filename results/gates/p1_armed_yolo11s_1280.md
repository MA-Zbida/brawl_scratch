# Gate P1 (armed_yolo11s_1280): FAIL

2026-11-10 09:54. 149 test frames from C:\brawl_data\labelling\export. Thresholds are provisional (research/CHECKLIST.md). A fighter is found when the predicted box of that identity has its centre inside the labelled box; the error is the distance between centres (1920 x 1080 pixels).

| Check | Value | Pass at | |
|---|---|---|---|
| wu_shang found | 0.959 | >= 0.98 | **fail** |
| wu_shang identity swaps | 0.000 | <= 0.01 | ok |
| wu_shang false detections (per frame) | 0.000 | <= 0.01 | ok |
| wu_shang centre error p95 (px) | 22.883 | <= 12.0 | **fail** |
| wu_shang held weapon | 0.648 | >= 0.95 | **fail** |
| mordex found | 0.958 | >= 0.98 | **fail** |
| mordex identity swaps | 0.014 | <= 0.01 | **fail** |
| mordex false detections (per frame) | 0.000 | <= 0.01 | ok |
| mordex centre error p95 (px) | 19.791 | <= 12.0 | **fail** |
| mordex held weapon | 0.603 | >= 0.95 | **fail** |
| weapon_spawn recall | 0.986 | >= 0.95 | ok |
| weapon_spawn precision | 0.654 | >= 0.95 | **fail** |
| weapon_thrown recall | 0.556 | >= 0.8 | **fail** |
| weapon_thrown precision | 0.152 | >= 0.8 | **fail** |

| Fighter | Labelled | Found | Missed | Swapped | False | Error median / p95 / max (px) |
|---|---|---|---|---|---|---|
| wu_shang | 148 | 142 | 6 | 0 | 0 | 5.3 / 22.9 / 32.4 |
| mordex | 142 | 136 | 4 | 2 | 0 | 3.2 / 19.8 / 28.7 |

| Weapon class | Labelled | Predicted | Matched | Recall | Precision | Error median (px) |
|---|---|---|---|---|---|---|
| weapon_spawn | 71 | 107 | 70 | 0.986 | 0.654 | 5.4 |
| weapon_thrown | 9 | 33 | 5 | 0.556 | 0.152 | 1.9 |

Held weapon (labelled -> predicted, found fighters):

- wu_shang_gauntlets -> wu_shang_gauntlets: 49
- wu_shang_spear -> wu_shang_spear: 31
- wu_shang_unarmed -> wu_shang_gauntlets: 28
- wu_shang_unarmed -> wu_shang_spear: 22
- wu_shang_unarmed -> wu_shang_unarmed: 12
- mordex_gauntlets -> mordex_gauntlets: 21
- mordex_gauntlets -> mordex_unarmed: 5
- mordex_scythe -> mordex_gauntlets: 1
- mordex_scythe -> mordex_scythe: 19
- mordex_scythe -> mordex_unarmed: 1
- mordex_unarmed -> mordex_gauntlets: 29
- mordex_unarmed -> mordex_scythe: 18
- mordex_unarmed -> mordex_unarmed: 42
