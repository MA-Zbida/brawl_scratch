"""Cut fighter crops from prepared recordings, for the probes (scripts/probe_vision.py).

    python scripts/prepare_crops.py --difficulty medium

For every decision of decisions_k<k>.npz (scripts/prepare_demos.py), the crops of both fighters
(brawl/vision.py) at the tracked torso points, from the decision's frame. Written next to the
recording as crops_k<k>_<size>.npy, uint8 of shape [decisions, 2 (Wu Shang, Mordex), size, size],
memory-mapped so that nothing has to fit in RAM: 18 KB per decision at 96 px, 0.8 GB per hour of
play. A fighter that is not present is cropped where it was last seen (the vector says it is absent).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from brawl import observation, stage, vision  # noqa: E402
from brawl.paths import DATA  # noqa: E402
from brawl.record import Recording  # noqa: E402


def positions(obs: np.ndarray, who: str) -> np.ndarray:
    f = observation.FEATURES
    x = obs[:, f.index(f"{who}_x")] * observation.X + stage.CENTRE[0]
    y = obs[:, f.index(f"{who}_y")] * observation.Y + stage.CENTRE[1]
    return np.stack([x, y], 1)


def prepare(root: Path, k: int, size: int) -> dict:
    z = np.load(root / f"decisions_k{k}.npz")
    n = len(z["obs"])
    points = (positions(z["obs"], "agent"), positions(z["obs"], "opponent"))
    path = root / f"crops_k{k}_{size}.npy"
    out = np.lib.format.open_memmap(path.with_suffix(".part"), mode="w+", dtype=np.uint8, shape=(n, 2, size, size))
    missing, t0 = 0, time.perf_counter()
    for row, frame in enumerate(Recording(root).frames(rows={i * k for i in range(n)})):
        i, r = divmod(row, k)
        if r or i >= n:
            continue
        if frame is None:
            missing += 1
        else:
            for who in range(2):
                out[i, who] = vision.crop(frame, points[who][i], size)
        if i % 3000 == 0:
            print(f"\r{root.name}: {i}/{n}", end="", flush=True)
    out.flush()
    del out
    path.with_suffix(".part").replace(path)
    return {"decisions": n, "missing": missing, "seconds": time.perf_counter() - t0, "mb": path.stat().st_size / 1e6}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recordings", type=Path, nargs="*", help="default: every prepared recording at --difficulty")
    p.add_argument("--difficulty", default="medium", help="comma separated bot levels (default medium)")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    levels = set(args.difficulty.split(","))
    roots = args.recordings or [q for q in sorted((DATA / "demos").iterdir())
                                if (q / f"decisions_k{args.k}.npz").exists()
                                and Recording(q).meta["setting"].get("difficulty") in levels]
    for root in roots:
        path = root / f"crops_k{args.k}_{args.size}.npy"
        if path.exists() and not args.force and np.load(path, mmap_mode="r").ndim == 4:
            print(f"{root.name}: already cut")
            continue
        s = prepare(root, args.k, args.size)
        print(f"\r{root.name}: {s['decisions']} decisions in {s['seconds']:.0f} s, {s['mb']:.0f} MB; "
              f"{s['missing']} without a picture")
    return 0


if __name__ == "__main__":
    sys.exit(main())
