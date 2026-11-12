# Gate P1 (armed_yolo11s-pose_960_trt_track30): PASS

2026-11-11 18:15. 149 test frames from C:\brawl_data\labelling\export_kp_final. Thresholds are provisional (research/CHECKLIST.md). A fighter is found when the predicted box of that identity has its position inside the labelled box; the error is the distance from the labelled torso point to the predicted one, or to the box centre for a candidate without points (1920 x 1080 pixels). Fighters and weapons are the tracker's (brawl.sense.track), after the 30 recording frames before each test frame; a fighter or spawn marker it carries through a frame without a box counts as predicted. The tracker sees one frame in 1.

| Check | Value | Pass at | |
|---|---|---|---|
| wu_shang found | 1.000 | >= 0.98 | ok |
| wu_shang identity swaps | 0.000 | <= 0.01 | ok |
| wu_shang false detections (per frame) | 0.000 | <= 0.01 | ok |
| wu_shang position error p95 (px) | 13.522 | <= 20.0 | ok |
| wu_shang held weapon | 0.959 | >= 0.95 | ok |
| mordex found | 0.986 | >= 0.98 | ok |
| mordex identity swaps | 0.007 | <= 0.01 | ok |
| mordex false detections (per frame) | 0.007 | <= 0.01 | ok |
| mordex position error p95 (px) | 15.306 | <= 20.0 | ok |
| mordex held weapon | 0.957 | >= 0.95 | ok |
| weapon_spawn recall | 0.974 | >= 0.95 | ok |
| weapon_spawn precision | 0.974 | >= 0.95 | ok |
| weapon_thrown recall | 0.812 | >= 0.8 | ok |
| weapon_thrown precision | 0.812 | >= 0.8 | ok |

| Fighter | Labelled | Found | Missed | Swapped | False | Error median / p95 / max (px) |
|---|---|---|---|---|---|---|
| wu_shang | 148 | 148 | 0 | 0 | 0 | 4.9 / 13.5 / 22.9 |
| mordex | 142 | 140 | 1 | 1 | 1 | 3.5 / 15.3 / 37.7 |

| Weapon class | Labelled | Predicted | Matched | Recall | Precision | Error median (px) |
|---|---|---|---|---|---|---|
| weapon_spawn | 76 | 76 | 74 | 0.974 | 0.974 | 4.5 |
| weapon_thrown | 16 | 16 | 13 | 0.812 | 0.812 | 1.4 |

Held weapon (labelled -> predicted, found fighters):

- wu_shang_gauntlets -> wu_shang_gauntlets: 70
- wu_shang_spear -> wu_shang_spear: 51
- wu_shang_spear -> wu_shang_unarmed: 3
- wu_shang_unarmed -> wu_shang_gauntlets: 1
- wu_shang_unarmed -> wu_shang_spear: 2
- wu_shang_unarmed -> wu_shang_unarmed: 21
- mordex_gauntlets -> mordex_gauntlets: 44
- mordex_gauntlets -> mordex_unarmed: 3
- mordex_scythe -> mordex_scythe: 37
- mordex_scythe -> mordex_unarmed: 1
- mordex_unarmed -> mordex_gauntlets: 2
- mordex_unarmed -> mordex_unarmed: 53
