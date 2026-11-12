"""brawl.sense.truth: reading the labelled frames and scoring a detector against them (gate P1)."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from brawl.sense.truth import Box, canonical, read_export, score_frame, summarise

NAMES = ["mordex_guntlets", "mordex_scythe", "mordex_unarmed", "weapon_spawn", "weapon_thrown",
         "wu_shang_guntlets", "wu_shang_spear", "wu_shang_unarmed"]  # as Roboflow exported them


def test_roboflow_spellings_map_to_the_labelling_classes():
    assert canonical("mordex_guntlets") == "mordex_gauntlets"
    with pytest.raises(ValueError):
        canonical("character")


def write_export(root, demos):
    """A two-image export: one drafted name (recording__row) and one bare row that two recordings share."""
    for split in ("test", "train"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
    (root / "data.yaml").write_text(f"nc: {len(NAMES)}\nnames: {NAMES}\n", encoding="utf-8")
    for rec, value in (("recA", 40), ("recB", 200)):
        (demos / rec / "stills").mkdir(parents=True)
        cv2.imwrite(str(demos / rec / "stills" / "0000120.png"), np.full((1080, 1920, 3), value, np.uint8))
    cv2.imwrite(str(demos / "recA" / "stills" / "0000500.png"), np.zeros((1080, 1920, 3), np.uint8))
    stem = "recA__0000500_png.rf.0123abcd"
    cv2.imwrite(str(root / "test" / "images" / f"{stem}.jpg"), np.zeros((1080, 1920, 3), np.uint8))
    (root / "test" / "labels" / f"{stem}.txt").write_text("7 0.5 0.5 0.1 0.2\n0 0.25 0.5 0.05 0.1\n")
    stem = "0000120_png.rf.ffff0000"  # looks like recB's still
    cv2.imwrite(str(root / "train" / "images" / f"{stem}.jpg"), np.full((1080, 1920, 3), 195, np.uint8))
    (root / "train" / "labels" / f"{stem}.txt").write_text("3 0.1 0.9 0.02 0.04\n")


def test_the_export_is_read_back_to_its_lossless_stills(tmp_path):
    write_export(tmp_path / "export", tmp_path / "demos")
    test, train = read_export(tmp_path / "export", tmp_path / "demos", splits=("test", "train"))
    assert (test.recording, test.row, test.still.name) == ("recA", 500, "0000500.png")
    wu, mordex = test.boxes
    assert wu.cls == "wu_shang_unarmed" and mordex.cls == "mordex_gauntlets"
    assert (wu.x0, wu.y0, wu.x1, wu.y1) == pytest.approx((864, 432, 1056, 648))
    assert (train.recording, train.row) == ("recB", 120)  # the shared row is resolved by its pixels
    assert train.boxes[0].cls == "weapon_spawn"


def test_fighters_are_found_missed_swapped_or_false():
    wu, mordex = Box("wu_shang_spear", 100, 100, 160, 180), Box("mordex_unarmed", 400, 100, 460, 180)
    exact = score_frame([wu, mordex], [Box("wu_shang_spear", 104, 100, 164, 180, 0.9),
                                       Box("mordex_scythe", 400, 100, 460, 180, 0.8)], reports_held=True)
    assert exact["fighters"]["wu_shang"]["found"] and exact["fighters"]["wu_shang"]["error"] == pytest.approx(4)
    swapped = score_frame([wu, mordex], [Box("wu_shang_unarmed", 400, 100, 460, 180)], reports_held=False)
    assert swapped["fighters"]["wu_shang"]["swap"] and not swapped["fighters"]["mordex"]["predicted"]
    ko = score_frame([wu], [Box("mordex_unarmed", 700, 100, 760, 180)], reports_held=False)  # Mordex KO'd
    s = summarise([exact, swapped, ko], reports_held=True)
    w, m = s["fighters"]["wu_shang"], s["fighters"]["mordex"]
    assert (w["labelled"], w["found"], w["swaps"], w["missed"]) == (3, 1, 1, 1)
    assert (m["labelled"], m["found"], m["false"], m["missed"]) == (2, 1, 1, 1)
    assert w["held"] == 1.0 and m["held"] == 0.0  # Mordex was unarmed, predicted with a scythe


def test_weapons_are_matched_once_each():
    spawns = [Box("weapon_spawn", 100, 100, 120, 160), Box("weapon_spawn", 300, 100, 320, 160)]
    predicted = [Box("weapon_spawn", 101, 101, 121, 161, 0.9), Box("weapon_spawn", 102, 100, 122, 160, 0.5),
                 Box("weapon_thrown", 500, 500, 520, 520)]
    r = score_frame(spawns, predicted, reports_held=False)
    assert r["weapons"]["weapon_spawn"] == {"labelled": 2, "predicted": 2, "matched": 1,
                                            "errors": [pytest.approx(np.hypot(1, 1))]}
    s = summarise([r], reports_held=False)
    assert s["weapons"]["weapon_spawn"]["recall"] == 0.5 and s["weapons"]["weapon_thrown"]["precision"] == 0.0


def test_a_pose_label_gives_the_torso_as_the_position(tmp_path):
    root = tmp_path / "labels.txt"
    root.write_text("6 0.5 0.5 0.1 0.2 0.52 0.48 2\n3 0.1 0.9 0.02 0.04\n")
    from brawl.sense.truth import read_yolo

    spear, spawn = read_yolo(root, NAMES, 1920, 1080)
    assert spear.cls == "wu_shang_spear" and spear.position == pytest.approx((998.4, 518.4))
    assert spawn.point is None and spawn.position == spawn.centre


def test_the_position_error_is_between_points_not_box_centres():
    # The spear widens the box to one side: the centres differ by 30 px, the torsos by 3.
    truth = Box("wu_shang_spear", 100, 100, 220, 180, point=(130, 140))
    predicted = Box("wu_shang_spear", 70, 100, 190, 180, 0.9, point=(133, 140))
    r = score_frame([truth], [predicted], reports_held=True)["fighters"]["wu_shang"]
    assert r["found"] and r["error"] == pytest.approx(3)


def test_two_boxes_of_one_fighter_are_reported():
    from brawl.sense.truth import Frame, problems

    wu, mordex = Box("wu_shang_spear", 0, 0, 10, 10), Box("mordex_unarmed", 50, 0, 60, 10)
    good = Frame("train", "rec", 3, None, [wu, mordex, Box("weapon_thrown", 0, 0, 5, 5)])
    bad = Frame("test", "rec", 7, None, [wu, Box("wu_shang_gauntlets", 80, 0, 90, 10), mordex])
    assert problems([good, bad]) == ["test rec__0000007: 2 wu_shang boxes"]
