"""Label frames for the ground-truth set (research/DESIGN.md section 9, gates P1 and P2).

For each full-resolution frame you mark both fighters: a tight box around the body (not effects
or the weapon swing), the way it faces and the weapon it holds, or that it is not on screen.
Perception is then measured against these labels.

    python scripts/label.py data/demos/<recording>/stills
    python scripts/label.py screenshots            # any folder of 1920 x 1080 PNGs

Labels are saved to labels.json in that folder after every frame; running again resumes at the
first unfinished frame.

Mouse: drag a box for the selected fighter.
Keys:  1 / 2  select Wu Shang / Mordex     f  facing (right, left)     w  weapon (cycles)
       x      not on screen (toggle)       u  not gameplay: menu, loading (toggle, skips the frame)
       n or space  next     b  back     q or Esc  save and quit
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

FIGHTERS = ("wu_shang", "mordex")
WEAPONS = {"wu_shang": ("none", "spear", "gauntlets"), "mordex": ("none", "scythe", "gauntlets")}
FACING = ("right", "left")
DISPLAY_SCALE = 0.5


class Labels:
    """labels.json: {frame name: {fighter: {"box": [x0, y0, x1, y1], "facing", "weapon"} or
    {"absent": true}} or {"unusable": true}}. Boxes are in full-resolution pixels."""

    def __init__(self, folder: Path) -> None:
        self.path = Path(folder) / "labels.json"
        self.data: dict[str, dict[str, Any]] = (
            json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {})

    def frame(self, name: str) -> dict[str, Any]:
        return self.data.setdefault(name, {})

    def set_box(self, name: str, fighter: str, box: tuple[int, int, int, int]) -> None:
        x0, y0, x1, y1 = box
        entry = self.frame(name).setdefault(fighter, {})
        entry.pop("absent", None)
        entry["box"] = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
        entry.setdefault("facing", None)
        entry.setdefault("weapon", "none")

    def toggle_absent(self, name: str, fighter: str) -> None:
        frame = self.frame(name)
        frame[fighter] = {} if frame.get(fighter, {}).get("absent") else {"absent": True}

    def toggle_unusable(self, name: str) -> None:
        """Not a gameplay frame (a menu is open, the game is loading): excluded from the set."""
        self.data[name] = {} if self.data.get(name, {}).get("unusable") else {"unusable": True}

    def cycle(self, name: str, fighter: str, field: str) -> None:
        entry = self.frame(name).get(fighter)
        if not entry or "box" not in entry:
            return
        options = FACING if field == "facing" else WEAPONS[fighter]
        current = entry.get(field)
        entry[field] = options[(options.index(current) + 1) % len(options)] if current in options else options[0]

    def complete(self, name: str) -> bool:
        """Both fighters are either absent, or boxed with a facing and a weapon."""
        frame = self.data.get(name, {})
        if frame.get("unusable"):
            return True
        for fighter in FIGHTERS:
            entry = frame.get(fighter, {})
            if not entry.get("absent") and not ("box" in entry and entry.get("facing") and entry.get("weapon")):
                return False
        return True

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def draw(image, labels: Labels, name: str, selected: str, index: int, total: int,
         drag: Optional[tuple[int, int, int, int]]):
    import cv2

    view = cv2.resize(image, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_AREA)
    colours = {"wu_shang": (0, 200, 255), "mordex": (255, 120, 0)}
    frame = labels.data.get(name, {})
    for fighter in FIGHTERS:
        entry = frame.get(fighter, {})
        colour = colours[fighter]
        if "box" in entry:
            x0, y0, x1, y1 = (int(v * DISPLAY_SCALE) for v in entry["box"])
            cv2.rectangle(view, (x0, y0), (x1, y1), colour, 3 if fighter == selected else 1)
            text = f"{fighter} {entry.get('facing') or '? facing'} {entry.get('weapon')}"
            cv2.putText(view, text, (x0, max(12, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
    if drag is not None:
        x0, y0, x1, y1 = drag
        cv2.rectangle(view, (x0, y0), (x1, y1), colours[selected], 1)
    status = []
    for fighter in FIGHTERS:
        entry = frame.get(fighter, {})
        state = "absent" if entry.get("absent") else ("boxed" if "box" in entry else "TODO")
        status.append(f"{'>' if fighter == selected else ' '}{fighter}: {state}")
    done = "NOT GAMEPLAY" if frame.get("unusable") else ("complete" if labels.complete(name) else "incomplete")
    header = f"{index + 1}/{total} {name} [{done}]   " + "   ".join(status)
    cv2.rectangle(view, (0, view.shape[0] - 24), (view.shape[1], view.shape[0]), (0, 0, 0), -1)
    cv2.putText(view, header, (6, view.shape[0] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return view


def main(argv=None) -> int:
    import cv2

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", type=Path)
    args = p.parse_args(argv)
    files = sorted(args.folder.glob("*.png"))
    if not files:
        print(f"no PNG frames in {args.folder}")
        return 1
    labels = Labels(args.folder)
    index = next((i for i, f in enumerate(files) if not labels.complete(f.name)), len(files) - 1)
    state = {"selected": FIGHTERS[0], "start": None, "drag": None}

    def on_mouse(event, x, y, flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            state["start"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["start"] is not None:
            state["drag"] = (*state["start"], x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["start"] is not None:
            x0, y0 = state["start"]
            state["start"], state["drag"] = None, None
            if abs(x - x0) > 3 and abs(y - y0) > 3:
                box = tuple(int(round(v / DISPLAY_SCALE)) for v in (x0, y0, x, y))
                labels.set_box(files[index].name, state["selected"], box)

    window = "label: drag a box | 1/2 fighter | f facing | w weapon | x absent | u not gameplay | n next | b back | q quit"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    image, loaded = None, None
    while True:
        name = files[index].name
        if loaded != index:
            image, loaded = cv2.imread(str(files[index])), index
        cv2.imshow(window, draw(image, labels, name, state["selected"], index, len(files), state["drag"]))
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("1"):
            state["selected"] = FIGHTERS[0]
        elif key == ord("2"):
            state["selected"] = FIGHTERS[1]
        elif key == ord("f"):
            labels.cycle(name, state["selected"], "facing")
        elif key == ord("w"):
            labels.cycle(name, state["selected"], "weapon")
        elif key == ord("x"):
            labels.toggle_absent(name, state["selected"])
        elif key == ord("u"):
            labels.toggle_unusable(name)
        elif key in (ord("n"), ord(" ")):
            labels.save()
            index = min(index + 1, len(files) - 1)
            state["selected"] = FIGHTERS[0]
        elif key == ord("b"):
            labels.save()
            index = max(index - 1, 0)
    labels.save()
    cv2.destroyAllWindows()
    done = sum(labels.complete(f.name) for f in files)
    skipped = sum(bool(labels.data.get(f.name, {}).get("unusable")) for f in files)
    print(f"{done}/{len(files)} frames complete ({skipped} not gameplay) -> {labels.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
