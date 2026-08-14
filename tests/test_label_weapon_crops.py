"""Tests for the weapon-crop labelling tool.

The parts worth guarding are the ones that would silently waste labelling effort:
reading the wrong class out of a YOLO label file, cropping so tightly that the held
weapon is outside the frame being judged, and failing to resume.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.label_weapon_crops import (
    CHARACTER_CLASS, Crop, crop_pixels, iter_crops, select_originals, source_key,
)


def _dataset(tmp_path, label_text: str, name: str = "frame_0001"):
    images = tmp_path / "train" / "images"
    labels = tmp_path / "train" / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (images / f"{name}.png").write_bytes(b"")          # iter_crops only reads labels
    (labels / f"{name}.txt").write_text(label_text, encoding="utf-8")
    return tmp_path


def test_only_character_boxes_are_offered(tmp_path) -> None:
    """Classes 1 and 2 are the self-indicator and ground weapons, not fighters."""
    root = _dataset(tmp_path, "\n".join([
        "0 0.50 0.50 0.10 0.20",
        "1 0.50 0.40 0.02 0.02",      # indicator_self
        "2 0.30 0.60 0.05 0.05",      # ground weapon
        "0 0.70 0.50 0.10 0.20",
    ]))
    crops = list(iter_crops(root, ["train"]))

    assert len(crops) == 2, "picked up non-character boxes"
    assert [c.index for c in crops] == [0, 1], "index must enumerate characters only"
    assert all(c.key.startswith("frame_0001__") for c in crops)


def test_missing_label_file_is_skipped(tmp_path) -> None:
    root = _dataset(tmp_path, "0 0.5 0.5 0.1 0.2")
    (root / "train" / "labels" / "frame_0001.txt").unlink()
    assert list(iter_crops(root, ["train"])) == []


def test_short_or_blank_lines_do_not_lose_the_rest_of_the_file(tmp_path) -> None:
    """Stray lines in an export are skipped; the real boxes after them survive."""
    root = _dataset(tmp_path, "\n".join(["", "garbage", "0 0.5 0.5 0.1 0.2"]))
    crops = list(iter_crops(root, ["train"]))
    assert len(crops) == 1
    assert crops[0].cx == 0.5


def test_padding_widens_the_crop_so_a_held_weapon_is_visible(tmp_path) -> None:
    """A tight box can exclude the weapon -- the thing being labelled."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    box = Crop(tmp_path / "x.png", 0, cx=0.5, cy=0.5, w=0.2, h=0.4)

    tight = crop_pixels(image, box, pad=0.0)
    padded = crop_pixels(image, box, pad=0.45)

    assert tight.shape[1] == 20 and tight.shape[0] == 40
    assert padded.shape[1] > tight.shape[1]
    assert padded.shape[0] > tight.shape[0]


def test_crop_is_clamped_at_the_frame_edge(tmp_path) -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    box = Crop(tmp_path / "x.png", 0, cx=0.02, cy=0.02, w=0.2, h=0.2)

    patch = crop_pixels(image, box, pad=0.45)

    assert patch is not None and patch.size > 0, "edge boxes must still yield a crop"
    assert patch.shape[0] <= 100 and patch.shape[1] <= 100


def test_degenerate_box_yields_nothing(tmp_path) -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    box = Crop(tmp_path / "x.png", 0, cx=-1.0, cy=-1.0, w=0.01, h=0.01)
    assert crop_pixels(image, box, pad=0.0) is None


def test_crop_keys_are_unique_per_character_in_a_frame(tmp_path) -> None:
    """Both fighters in one frame must not overwrite each other's saved crop."""
    root = _dataset(tmp_path, "0 0.3 0.5 0.1 0.2\n0 0.7 0.5 0.1 0.2")
    keys = [c.key for c in iter_crops(root, ["train"])]
    assert len(keys) == len(set(keys)) == 2


def test_augmented_copies_group_under_one_source_key() -> None:
    """Roboflow names augmented copies with the same prefix and a different hash."""
    base = "frame_20260114_180307_000200_png"
    paths = [Path(f"{base}.rf.513dcdfce1eece439d29e185b42f871e.jpg"),
             Path(f"{base}.rf.bd342035b75b4cd10addda189e8c4902.jpg"),
             Path(f"{base}.rf.fed75cea5cab8bc4b659e3a95635f7ce.jpg")]
    assert len({source_key(p) for p in paths}) == 1
    assert source_key(paths[0]) == base


def test_select_originals_keeps_the_least_noisy_of_each_group(tmp_path) -> None:
    """The un-augmented original scores 0.00; augmented copies score well above it.

    A crop from a heavily augmented frame is genuinely unlabelable -- noise over a
    ~70x180 sprite hides whether a weapon is held -- so offering one wastes the only
    expensive resource in this workflow, human attention.
    """
    import cv2

    clean = np.full((40, 40, 3), 128, dtype=np.uint8)
    rng = np.random.default_rng(0)
    noisy = np.clip(clean.astype(np.int16) + rng.integers(-60, 60, clean.shape), 0, 255).astype(np.uint8)

    paths = []
    for name, img in (("a.rf.aaa.jpg", noisy), ("a.rf.bbb.jpg", clean), ("a.rf.ccc.jpg", noisy)):
        p = tmp_path / name
        cv2.imwrite(str(p), img)
        paths.append(p)
    solo = tmp_path / "b.rf.ddd.jpg"
    cv2.imwrite(str(solo), clean)
    paths.append(solo)

    chosen = select_originals(cv2, paths, cache_path=tmp_path / "cache.json")

    assert len(chosen) == 2, "one image kept per source frame"
    assert (tmp_path / "a.rf.bbb.jpg") in chosen, "kept an augmented copy over the original"
    assert solo in chosen, "un-duplicated images must pass through untouched"


def test_iter_crops_honours_the_keep_filter(tmp_path) -> None:
    root = _dataset(tmp_path, "0 0.5 0.5 0.1 0.2", name="frame_a")
    (root / "train" / "images" / "frame_b.png").write_bytes(b"")
    (root / "train" / "labels" / "frame_b.txt").write_text("0 0.3 0.3 0.1 0.2", encoding="utf-8")

    keep = {root / "train" / "images" / "frame_a.png"}
    keys = [c.key for c in iter_crops(root, ["train"], keep=keep)]
    assert keys == ["frame_a__0"]
