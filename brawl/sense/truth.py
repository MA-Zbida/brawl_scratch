"""The labelled frames (docs/labelling.md) and how a detector is scored against them (gate P1).

Each label is a box, a class and one point (YOLO pose, ``cls cx cy w h px py v``): for a fighter the
point is the centre of the torso, its position; for anything else it is the box centre. Labels
without a point (plain detection) take the box centre.

Roboflow exports YOLO labels on renamed JPEG copies (``<name>_png.rf.<hash>.jpg``). Only the labels
are used: each is matched back to its lossless still in the recordings, so every box knows its
recording and row. Names drafted by scripts/prelabel.py carry the recording
(``<recording>__<row>``); a bare ``<row>`` is resolved to the recording that has that still, by
comparing pixels when several do.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CLASSES = ("wu_shang_unarmed", "wu_shang_spear", "wu_shang_gauntlets",
           "mordex_unarmed", "mordex_scythe", "mordex_gauntlets",
           "weapon_spawn", "weapon_thrown", "offscreen")
FIGHTERS = ("wu_shang", "mordex")
WEAPONS = ("weapon_spawn", "weapon_thrown")
SPELLINGS = {"guntlets": "gauntlets"}  # how classes were typed in Roboflow -> docs/labelling.md

# P1 pass conditions (reviewed 2026-11-11, research/CHECKLIST.md). The torso error was 12 px until
# then: hand-placed torso points agree to about 15 px in attack poses, and a fighter is 60-90 px tall.
PASS = {"found": 0.98, "swaps": 0.01, "false": 0.01, "error_p95_px": 20.0, "held": 0.95,
        "weapon_spawn_recall": 0.95, "weapon_spawn_precision": 0.95,
        "weapon_thrown_recall": 0.80, "weapon_thrown_precision": 0.80}


def canonical(name: str) -> str:
    for wrong, right in SPELLINGS.items():
        name = name.replace(wrong, right)
    if name not in CLASSES:
        raise ValueError(f"unknown class {name!r}")
    return name


def identity(cls: str) -> str | None:
    """``wu_shang`` or ``mordex`` for a fighter class, None otherwise."""
    return next((f for f in FIGHTERS if cls.startswith(f + "_")), None)


@dataclass(frozen=True)
class Box:
    cls: str
    x0: float
    y0: float
    x1: float
    y1: float
    score: float = 1.0
    point: tuple[float, float] | None = None  # the torso, for a fighter

    @property
    def centre(self) -> tuple[float, float]:
        return (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2

    @property
    def position(self) -> tuple[float, float]:
        """Where the thing is: the labelled or predicted point, else the box centre."""
        return self.point if self.point is not None else self.centre

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return self.x0 - margin <= x <= self.x1 + margin and self.y0 - margin <= y <= self.y1 + margin


@dataclass
class Frame:
    split: str
    recording: str
    row: int
    still: Path
    boxes: list[Box] = field(default_factory=list)


def read_yolo(path: Path, names: list[str], width: int, height: int) -> list[Box]:
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        c, (x, y, w, h) = parts[0], map(float, parts[1:5])
        x, y, w, h = x * width, y * height, w * width, h * height
        point = (float(parts[5]) * width, float(parts[6]) * height) if len(parts) >= 7 else None
        boxes.append(Box(canonical(names[int(c)]), x - w / 2, y - h / 2, x + w / 2, y + h / 2, point=point))
    return boxes


def _names(export: Path) -> list[str]:
    for line in (export / "data.yaml").read_text(encoding="utf-8").splitlines():
        if line.startswith("names:"):
            return re.findall(r"['\"]([^'\"]+)['\"]", line)
    raise ValueError(f"no names in {export / 'data.yaml'}")


def _resolve(stem: str, image: Path, demos: Path) -> tuple[str, int, Path]:
    if "__" in stem:
        recording, row = stem.split("__")
        return recording, int(row), demos / recording / "stills" / f"{row}.png"
    candidates = sorted(demos.glob(f"*/stills/{stem}.png"))
    if not candidates:
        raise FileNotFoundError(f"no recording has the still {stem}.png")
    if len(candidates) > 1:  # rows repeat across recordings: pick the still that looks like the image
        import cv2

        target = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        candidates.sort(key=lambda p: float(np.abs(
            cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) - target).mean()))
    still = candidates[0]
    return still.parent.parent.name, int(stem), still


def read_export(export: Path, demos: Path, splits=("train", "valid", "test"),
                size: tuple[int, int] = (1920, 1080)) -> list[Frame]:
    """Every labelled frame of a Roboflow YOLO export, pointing to its lossless still."""
    names = _names(export)
    frames = []
    for split in splits:
        for image in sorted((export / split / "images").glob("*")):
            stem = re.sub(r"_png\.rf\.[0-9a-f]+$", "", image.stem)
            recording, row, still = _resolve(stem, image, demos)
            boxes = read_yolo(export / split / "labels" / f"{image.stem}.txt", names, *size)
            frames.append(Frame(split, recording, row, still, boxes))
    return frames


def problems(frames: list[Frame]) -> list[str]:
    """Labels that cannot be right: there is one Wu Shang and one Mordex, so two boxes of one
    fighter in a frame mean a duplicate, a wrong identity, or a box on something else."""
    return [f"{f.split} {f.recording}__{f.row:07d}: {n} {who} boxes" for f in frames for who in FIGHTERS
            if (n := sum(identity(b.cls) == who for b in f.boxes)) > 1]


# -- scoring ------------------------------------------------------------------------------------
def _fighter(boxes: list[Box], who: str) -> Box | None:
    mine = [b for b in boxes if identity(b.cls) == who]
    return max(mine, key=lambda b: b.score) if mine else None


def score_frame(truth: list[Box], predicted: list[Box], reports_held: bool, weapon_margin: float = 10.0) -> dict:
    """Compare one frame's predictions to its labels. A fighter is found when the predicted box of
    that identity has its position inside the labelled box; the error is the distance between the
    positions (torso points, or box centres for a candidate without points)."""
    out: dict = {"fighters": {}, "weapons": {}}
    for who in FIGHTERS:
        t, p = _fighter(truth, who), _fighter(predicted, who)
        other = _fighter(truth, FIGHTERS[1 - FIGHTERS.index(who)])
        r = {"labelled": t is not None, "predicted": p is not None, "found": False, "swap": False,
             "error": None, "held_true": t.cls if t else None, "held_pred": p.cls if p and reports_held else None}
        if p is not None:
            if t is not None and t.contains(*p.position):
                r["found"] = True
                r["error"] = float(np.hypot(p.position[0] - t.position[0], p.position[1] - t.position[1]))
            elif other is not None and other.contains(*p.position):
                r["swap"] = True
        out["fighters"][who] = r
    for cls in WEAPONS:
        ts = [b for b in truth if b.cls == cls]
        ps = sorted((b for b in predicted if b.cls == cls), key=lambda b: -b.score)
        matched, errors = set(), []
        for p in ps:
            free = [i for i, t in enumerate(ts) if i not in matched and t.contains(*p.centre, weapon_margin)]
            if free:
                i = min(free, key=lambda i: np.hypot(*np.subtract(ts[i].centre, p.centre)))
                matched.add(i)
                errors.append(float(np.hypot(*np.subtract(ts[i].centre, p.centre))))
        out["weapons"][cls] = {"labelled": len(ts), "predicted": len(ps), "matched": len(matched), "errors": errors}
    return out


def summarise(results: list[dict], reports_held: bool) -> dict:
    """Totals over frames, and each P1 condition with its pass or fail."""
    s: dict = {"frames": len(results), "fighters": {}, "weapons": {}, "checks": []}
    for who in FIGHTERS:
        rs = [r["fighters"][who] for r in results]
        labelled = [r for r in rs if r["labelled"]]
        found = [r for r in labelled if r["found"]]
        errors = np.array([r["error"] for r in found]) if found else None
        f = {"labelled": len(labelled), "found": len(found), "swaps": sum(r["swap"] for r in rs),
             "false": sum(r["predicted"] and not r["labelled"] for r in rs),
             "missed": sum(not r["found"] and not r["swap"] for r in labelled),
             "error_median_px": float(np.median(errors)) if found else float("nan"),
             "error_p95_px": float(np.percentile(errors, 95)) if found else float("nan"),
             "error_max_px": float(errors.max()) if found else float("nan")}
        if reports_held:
            f["held"] = sum(r["held_true"] == r["held_pred"] for r in found) / max(len(found), 1)
            f["confusion"] = {}
            for r in found:
                key = (r["held_true"], r["held_pred"])
                f["confusion"][key] = f["confusion"].get(key, 0) + 1
        s["fighters"][who] = f
        n = max(len(labelled), 1)
        s["checks"] += [(f"{who} found", f["found"] / n, PASS["found"], f["found"] / n >= PASS["found"]),
                        (f"{who} identity swaps", f["swaps"] / n, PASS["swaps"], f["swaps"] / n <= PASS["swaps"]),
                        (f"{who} false detections (per frame)", f["false"] / len(rs), PASS["false"],
                         f["false"] / len(rs) <= PASS["false"]),
                        (f"{who} position error p95 (px)", f["error_p95_px"], PASS["error_p95_px"],
                         f["error_p95_px"] <= PASS["error_p95_px"])]
        if reports_held:
            s["checks"].append((f"{who} held weapon", f["held"], PASS["held"], f["held"] >= PASS["held"]))
    for cls in WEAPONS:
        ws = [r["weapons"][cls] for r in results]
        labelled, predicted, matched = (sum(w[k] for w in ws) for k in ("labelled", "predicted", "matched"))
        errors = [e for w in ws for e in w["errors"]]
        recall, precision = matched / max(labelled, 1), matched / max(predicted, 1)
        s["weapons"][cls] = {"labelled": labelled, "predicted": predicted, "matched": matched,
                             "recall": recall, "precision": precision,
                             "error_median_px": float(np.median(errors)) if errors else float("nan")}
        s["checks"] += [(f"{cls} recall", recall, PASS[f"{cls}_recall"], recall >= PASS[f"{cls}_recall"]),
                        (f"{cls} precision", precision, PASS[f"{cls}_precision"],
                         precision >= PASS[f"{cls}_precision"])]
    return s
