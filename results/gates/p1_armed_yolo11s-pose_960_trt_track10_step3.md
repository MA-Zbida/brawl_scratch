# Gate P1 (armed_yolo11s-pose_960_trt_track10_step3): FAIL

2026-11-11 15:32. 149 test frames from C:\brawl_data\labelling\export_kp_final. Thresholds are provisional (research/CHECKLIST.md). A fighter is found when the predicted box of that identity has its position inside the labelled box; the error is the distance from the labelled torso point to the predicted one, or to the box centre for a candidate without points (1920 x 1080 pixels). Fighters and weapons are the tracker's (brawl.sense.track), after the 10 recording frames before each test frame; a fighter or spawn marker it carries through a frame without a box counts as predicted. The tracker sees one frame in 3.

| Check | Value | Pass at | |
|---|---|---|---|
| wu_shang found | 1.000 | >= 0.98 | ok |
| wu_shang identity swaps | 0.000 | <= 0.01 | ok |
| wu_shang false detections (per frame) | 0.000 | <= 0.01 | ok |
| wu_shang position error p95 (px) | 13.594 | <= 20.0 | ok |
| wu_shang held weapon | 0.946 | >= 0.95 | **fail** |
| mordex found | 0.972 | >= 0.98 | **fail** |
| mordex identity swaps | 0.014 | <= 0.01 | **fail** |
| mordex false detections (per frame) | 0.007 | <= 0.01 | ok |
| mordex position error p95 (px) | 13.680 | <= 20.0 | ok |
| mordex held weapon | 0.964 | >= 0.95 | ok |
| weapon_spawn recall | 0.961 | >= 0.95 | ok |
| weapon_spawn precision | 0.948 | >= 0.95 | **fail** |
| weapon_thrown recall | 0.812 | >= 0.8 | ok |
| weapon_thrown precision | 0.812 | >= 0.8 | ok |

| Fighter | Labelled | Found | Missed | Swapped | False | Error median / p95 / max (px) |
|---|---|---|---|---|---|---|
| wu_shang | 148 | 148 | 0 | 0 | 0 | 4.9 / 13.6 / 22.9 |
| mordex | 142 | 138 | 2 | 2 | 1 | 3.5 / 13.7 / 25.4 |

| Weapon class | Labelled | Predicted | Matched | Recall | Precision | Error median (px) |
|---|---|---|---|---|---|---|
| weapon_spawn | 76 | 77 | 73 | 0.961 | 0.948 | 4.6 |
| weapon_thrown | 16 | 16 | 13 | 0.812 | 0.812 | 1.3 |

Held weapon (labelled -> predicted, found fighters):

- wu_shang_gauntlets -> wu_shang_gauntlets: 67
- wu_shang_gauntlets -> wu_shang_spear: 1
- wu_shang_gauntlets -> wu_shang_unarmed: 2
- wu_shang_spear -> wu_shang_spear: 53
- wu_shang_spear -> wu_shang_unarmed: 1
- wu_shang_unarmed -> wu_shang_gauntlets: 1
- wu_shang_unarmed -> wu_shang_spear: 3
- wu_shang_unarmed -> wu_shang_unarmed: 20
- mordex_gauntlets -> mordex_gauntlets: 45
- mordex_gauntlets -> mordex_unarmed: 1
- mordex_scythe -> mordex_scythe: 37
- mordex_scythe -> mordex_unarmed: 1
- mordex_unarmed -> mordex_gauntlets: 2
- mordex_unarmed -> mordex_scythe: 1
- mordex_unarmed -> mordex_unarmed: 51
