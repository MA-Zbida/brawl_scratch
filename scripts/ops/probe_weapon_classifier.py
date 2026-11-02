#!/usr/bin/env python
"""Show what the weapon classifier is actually being fed, live, and what it says.

The classifier was trained on crops cut from a Roboflow export -- 960x960 images
resized from 1920x1080 -- and at inference it sees crops from the native capture.
That is a real distribution shift in both scale and aspect, and it is invisible
from the outside: a wrong prediction looks exactly like a right one.

This dumps the crops to disk next to their predictions, so the question "is the box
right, or is the model wrong?" is answered by looking rather than by reasoning. The
two failures need opposite fixes:

* **wrong crop** (off-centre, wrong fighter, clipped) -- fix `_weapon_classifier_boxes`
* **right crop, wrong label** -- retrain on crops cut the way inference cuts them

Writes `weapon_probe/<n>_<name>_<pred>_<conf>.png` plus a side-by-side sheet.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

_HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_HERE))

import brawl.capture_first  # noqa: E402,F401  (must precede torch)
from brawl.paths import DATA, LEGACY_RUNS


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=str, default=str(DATA / "models" / "weapon_classifier_r18.pt"))
    p.add_argument("--samples", type=int, default=12)
    p.add_argument("--interval", type=float, default=0.5, help="Seconds between samples.")
    p.add_argument("--out", type=Path, default=LEGACY_RUNS / "weapon_classifier" / "probe")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    import cv2
    import numpy as np

    from brawl.env.live import BrawlDeepEnv, EnvConfig, NullInputController
    from brawl.state.state_spec import StateSpec

    args.out.mkdir(parents=True, exist_ok=True)

    env = BrawlDeepEnv(
        config=EnvConfig(
            weapon_classifier_enabled=True,
            weapon_classifier_checkpoint=args.checkpoint,
            weapon_classifier_interval=1,        # every frame, this is a probe
            max_episode_steps=0,
        ),
        input_controller=NullInputController(),   # never touch the game
    )
    clf = env.weapon_classifier
    print(f"classifier: {args.checkpoint}")
    print(f"  contract {clf.width}x{clf.height}  pad={clf.pad}  cv_acc={clf.cv_accuracy:.3f}")
    print(f"  training source was 960x960 (Roboflow); live capture is 1920x1080\n")

    obs, _ = env.reset()
    rows = []
    for i in range(int(args.samples)):
        obs, _r, _t, _tr, info = env.step(0)
        frame = env._last_frame
        boxes = env._weapon_classifier_boxes()
        if frame is None or not boxes:
            print(f"[{i:02d}] no frame or no resolved fighter")
            time.sleep(float(args.interval))
            continue

        preds = clf.update(frame, boxes)
        line = f"[{i:02d}] "
        for name, box in boxes.items():
            patch = clf.crop(frame, *box)
            pred = preds.get(name)
            if patch is None or pred is None:
                line += f"{name}=<no crop>  "
                continue
            label = "ARMED" if pred.armed else "unarmed"
            line += (f"{name}={label}@{pred.confidence:.2f} "
                     f"box={box[2]*1920:.0f}x{box[3]*1080:.0f}px  ")
            path = args.out / f"{i:02d}_{name}_{label}_{pred.confidence:.2f}.png"
            cv2.imwrite(str(path), patch)
            rows.append((patch, f"{name} {label} {pred.confidence:.2f}"))
        line += f"| obs player_has_weapon={StateSpec.get(obs, 'player_has_weapon'):.0f}"
        print(line)
        time.sleep(float(args.interval))

    env.close()

    if rows:
        height = max(p.shape[0] for p, _ in rows) + 18
        width = sum(p.shape[1] for p, _ in rows) + 4 * len(rows)
        sheet = np.full((height, width, 3), 40, dtype=np.uint8)
        x = 0
        for patch, caption in rows:
            h, w = patch.shape[:2]
            sheet[18:18 + h, x:x + w] = patch
            cv2.putText(sheet, caption.split()[1][:7], (x, 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            x += w + 4
        cv2.imwrite(str(args.out / "sheet.png"), sheet)
        print(f"\nwrote {len(rows)} crops + sheet.png to {args.out}/")
        print("Open sheet.png: if the crops are well-centred on the fighters, the boxes")
        print("are fine and the model is wrong -- retrain on native-resolution crops.")
        print("If they are off-centre or clipped, the box conversion is wrong instead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
