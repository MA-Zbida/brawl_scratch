#!/usr/bin/env python
"""Print exactly what the damage probe reads, live, so the reading can be checked.

`get_dmg` returns 0.0 both when a pixel is genuinely at zero damage and when the
pixel matches no colour range at all. Those are very different situations and the
return value cannot distinguish them, so this tool reports the raw pixel, whether
any branch matched, and the resulting damage separately.

It also samples a small patch around each configured point. The health readout is a
thin curved arc, so an exactly-centred pixel can still land on an anti-aliased edge
whose colour matches no branch while neighbours a pixel away read cleanly. If the
centre and the patch disagree, that is the whole story.

Run it with the game visible and hit the dummy a few times; the damage column should
move and stay moved.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

import capture_first  # noqa: E402,F401  (must precede torch)


def _matched_branch(rgb) -> str:
    """Name the branch `get_dmg` takes, or NO-MATCH when it falls through to 0.0."""
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    if g == 255 and r == 255 and b < 255:
        return "white->yellow"
    if r == 255 and b == 0 and g >= 153:
        return "yellow->orange"
    if r == 255 and b == 0 and g < 153:
        return "orange->red"
    if g == 0 and b == 0 and r >= 191:
        return "red"
    if g == 0 and b == 0 and 140 <= r < 191:
        return "dark red"
    if g == 0 and b == 0 and 74 < r <= 140:
        return "very dark red"
    if r <= 74 and g == 0 and b == 0:
        return "near black"
    return "NO-MATCH -> 0.0"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--hz", type=float, default=4.0)
    p.add_argument("--radius", type=int, default=2, help="Patch half-width sampled around each point.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    import numpy as np

    from capture import DxcamFrameProvider
    from config import UI_REGIONS
    from reward.rgb_to_dmg import get_dmg

    points = {k: v for k, v in UI_REGIONS.items() if k in ("op", "agent")}
    print(f"UI_REGIONS: {points}   patch radius {args.radius}")

    provider = DxcamFrameProvider()
    frame = None
    for _ in range(120):
        frame = provider.get_frame()
        if frame is not None:
            break
        time.sleep(0.02)
    if frame is None:
        print("No frame captured.", file=sys.stderr)
        return 1
    print(f"frame {frame.shape}\n")

    header = f"{'t':>6s}"
    for name in points:
        header += f" | {name:>6s} {'rgb':>16s} {'dmg':>7s} {'branch':>16s} {'patch_best':>10s}"
    print(header)
    print("-" * len(header))

    deadline = time.perf_counter() + float(args.seconds)
    period = 1.0 / max(0.5, float(args.hz))
    start = time.perf_counter()
    seen: dict[str, set] = {name: set() for name in points}

    while time.perf_counter() < deadline:
        frame = provider.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue
        row = f"{time.perf_counter() - start:6.1f}"
        for name, (x, y) in points.items():
            px = frame[y, x][:3]
            # DXCam surfaces BGR; get_dmg is written against RGB ordering.
            rgb = np.asarray([px[2], px[1], px[0]], dtype=int)
            dmg = get_dmg(rgb)

            r = int(args.radius)
            patch = frame[max(0, y - r): y + r + 1, max(0, x - r): x + r + 1]
            best = 0.0
            for row_px in patch.reshape(-1, patch.shape[-1]):
                cand = get_dmg(np.asarray([row_px[2], row_px[1], row_px[0]], dtype=int))
                best = max(best, cand)

            seen[name].add(round(float(dmg), 1))
            row += (f" | {name:>6s} {str(tuple(int(v) for v in rgb)):>16s} {dmg:7.1f} "
                    f"{_matched_branch(rgb):>16s} {best:10.1f}")
        print(row)
        time.sleep(period)

    provider.close()
    print("\ndistinct damage values observed per point:")
    for name, values in seen.items():
        ordered = sorted(values)
        print(f"  {name:>6s}: {ordered if len(ordered) <= 12 else ordered[:12] + ['...']}")
        if ordered == [0.0]:
            print(f"          {name} never left 0.0 -- either genuinely undamaged, or the pixel "
                  "matches no branch. Compare the branch column against patch_best.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
