#!/usr/bin/env python
"""Capture character crops from live gameplay, for relabelling the weapon classifier.

The first classifier reached 95.7% cross-validated and then called plainly unarmed
fighters ARMED at 1.00 confidence in live play. Measured mean P(armed):

    training `unarmed` crops (960x960 Roboflow export)   0.004   correct
    live crops (native 1920x1080 capture)                0.873   wrong
    live crops downscaled into 960-space                 0.632   still wrong

That is train/serve skew. The model learned from JPEG-compressed, downscaled export
frames and was served raw capture buffers; cross-validation could not see it because
every fold came from the same corpus. Rescaling at inference recovers only a quarter
of the gap, so the fix is training data drawn from the deployment distribution.

This captures crops exactly as `WeaponClassifier` will crop them at inference -- same
box source, same padding read from the checkpoint contract -- so the relabelled set
matches serving by construction. It never sends input to the game.

Afterwards, label with:

    python scripts\\ops\\label_weapon_crops.py --crops-dir data\\weapon_crops_live
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

_HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_HERE))

import brawl.capture_first  # noqa: E402,F401  (must precede torch)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data/weapon_crops_live/unlabelled"))
    p.add_argument("--samples", type=int, default=400, help="Crops to capture (both fighters count separately).")
    p.add_argument("--interval", type=float, default=0.35,
                   help="Seconds between captures. Consecutive frames are near-duplicates; "
                        "spacing them out buys variety per unit of labelling effort.")
    p.add_argument("--pad", type=float, default=0.45, help="Must match the labelling/inference contract.")
    p.add_argument("--min-side", type=int, default=24, help="Skip crops smaller than this; they are unlabelable.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    import cv2

    from brawl.env.live import BrawlDeepEnv, EnvConfig, NullInputController

    args.out.mkdir(parents=True, exist_ok=True)

    env = BrawlDeepEnv(
        config=EnvConfig(
            weapon_classifier_enabled=False,   # capturing pixels, not classifying
            max_episode_steps=0,
        ),
        input_controller=NullInputController(),   # never touches the game
    )

    print("=" * 70)
    print(f"Capturing up to {args.samples} live crops -> {args.out}")
    print("Play normally: pick weapons up, drop them, get disarmed, fight.")
    print("The more varied the weapon states, the better the retrained model.")
    print("Ctrl-C stops early; everything captured so far is kept.")
    print("=" * 70)

    env.reset()
    saved = 0
    index = 0
    manifest: list[dict] = []

    try:
        while saved < int(args.samples):
            _obs, _r, _t, _tr, _info = env.step(0)
            frame = env._last_frame
            boxes = env._weapon_classifier_boxes()
            if frame is None or not boxes:
                time.sleep(float(args.interval))
                continue

            height, width = frame.shape[:2]
            for name, (cx, cy, w, h) in boxes.items():
                half_w = (w * (1.0 + float(args.pad))) / 2.0
                half_h = (h * (1.0 + float(args.pad))) / 2.0
                x0 = max(0, int((cx - half_w) * width))
                x1 = min(width, int((cx + half_w) * width))
                y0 = max(0, int((cy - half_h) * height))
                y1 = min(height, int((cy + half_h) * height))
                if x1 - x0 < int(args.min_side) or y1 - y0 < int(args.min_side):
                    continue

                patch = frame[y0:y1, x0:x1]
                # PNG, not JPEG: the whole point is to avoid the compression the
                # export applied. Saving as JPEG would recreate the skew.
                path = args.out / f"live_{index:05d}__{0 if name == 'player' else 1}.png"
                cv2.imwrite(str(path), patch)
                manifest.append({"file": path.name, "fighter": name,
                                 "box": [cx, cy, w, h], "px": [x1 - x0, y1 - y0]})
                saved += 1
                index += 1

            if saved % 25 == 0 and saved:
                print(f"  {saved}/{args.samples} crops")
            time.sleep(float(args.interval))
    except KeyboardInterrupt:
        print("\nStopped early.")
    finally:
        env.close()

    (args.out / "capture_manifest.json").write_text(
        json.dumps({"pad": float(args.pad), "source": "live_dxcam_1920x1080",
                    "crops": manifest}, indent=2), encoding="utf-8")

    print(f"\n{saved} crops in {args.out}")
    print("Label them with:")
    print(f"  python scripts\\ops\\label_weapon_crops.py --crops-dir {args.out.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
