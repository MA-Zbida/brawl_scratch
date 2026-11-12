"""Turn recordings of human play into decisions: observation, action and events, one row per decision.

    python scripts/prepare_demos.py                      # every recording not yet prepared
    python scripts/prepare_demos.py <recording> --force  # one, again

Each recording is read as the environment would have seen it: every ``k``-th frame goes through
brawl.sense.perception (detector, tracker, HUD, ledger) and brawl.observation, and the keys of the
``k`` frames that follow become the action (brawl.actions.from_demo). The result is written next to
the recording as decisions_k<k>.npz:

    obs [N, 112]  action [N, 2]   dealt, taken, ko_scored, ko_lost [N]: what happened during the decision
    lost_presses [N]: button presses the action could not express

Recordings made with overlays are skipped (docs/labelling.md: frames without overlays only).
A file prepared before the timers existed gets them added in place, without the detector.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from brawl import actions  # noqa: E402
from brawl.observation import BASE, FEATURES, Observation, add_timers  # noqa: E402
from brawl.paths import DATA, RUNS  # noqa: E402
from brawl.record import Recording  # noqa: E402

DETECTOR = RUNS / "detector" / "armed_yolo11s-pose_960" / "weights" / "best.engine"


def prepare(root: Path, perception, k: int) -> dict:
    rec = Recording(root)
    columns = rec.columns()
    keys, pressed = columns["keys"].astype(int), columns["pressed"].astype(int)
    n = len(keys) // k
    perception.reset()
    builder = Observation()
    obs = np.zeros((n, len(FEATURES)), np.float32)
    action = np.zeros((n, 2), np.int64)
    events = np.zeros((n, 4), np.float32)   # dealt, taken, ko_scored, ko_lost during the decision
    lost = np.zeros(n, np.int64)
    missing, previous, t0 = 0, 0, time.perf_counter()
    for row, frame in enumerate(rec.frames(rows=set(range(0, n * k, k)))):
        i, first = divmod(row, k)
        if first or i >= n:
            continue
        if frame is None:  # a row without a picture: nothing new is known
            missing += 1
            obs[i] = obs[i - 1] if i else 0
        else:
            sense = perception(frame, row)
            e = sense.events
            if i:
                events[i - 1] = (e.damage_dealt, e.damage_taken, e.ko_scored, e.ko_lost)
            obs[i] = builder.build(sense)
        a, lost[i] = actions.from_demo(list(keys[row:row + k]), list(pressed[row:row + k]), previous)
        action[i], previous = a, a[1]
        builder.acted(a)
        if i % 2000 == 0:
            print(f"\r{root.name}: {i}/{n} ({(time.perf_counter() - t0) / max(i, 1) * 1000:.1f} ms each)", end="", flush=True)
    np.savez_compressed(root / f"decisions_k{k}.npz", obs=obs, action=action, dealt=events[:, 0], taken=events[:, 1],
                        ko_scored=events[:, 2], ko_lost=events[:, 3], lost_presses=lost,
                        features=np.array(FEATURES), k=k, detector=str(perception.detector.name))
    return {"decisions": n, "missing": missing, "ko_scored": int(events[:, 2].sum()), "ko_lost": int(events[:, 3].sum()),
            "dealt": float(events[:, 0].sum()), "taken": float(events[:, 1].sum()), "lost_presses": int(lost.sum()),
            "difficulty": rec.meta["setting"].get("difficulty", "?"), "seconds": time.perf_counter() - t0}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recordings", type=Path, nargs="*", help="default: every recording under <BRAWL_DATA>/demos")
    p.add_argument("--k", type=int, default=3, help="game frames per decision")
    p.add_argument("--detector", type=Path, default=DETECTOR)
    p.add_argument("--force", action="store_true", help="prepare again what is already prepared")
    args = p.parse_args(argv)

    from brawl.sense.detectors import Yolo
    from brawl.sense.perception import Perception

    roots = args.recordings or sorted(q for q in (DATA / "demos").iterdir() if (q / "meta.json").exists())
    perception = None
    for root in roots:
        meta = Recording(root).meta
        if meta["setting"].get("overlays"):
            print(f"{root.name}: skipped, recorded with overlays {meta['setting']['overlays']}")
            continue
        path = root / f"decisions_k{args.k}.npz"
        if path.exists() and not args.force:
            z = dict(np.load(path))
            if tuple(z["features"]) == BASE:
                z["obs"], z["features"] = add_timers(z["obs"], z["action"], z["taken"], z["dealt"]), np.array(FEATURES)
                np.savez_compressed(path, **z)
                print(f"{root.name}: timers added")
            elif tuple(z["features"]) != FEATURES:
                print(f"{root.name}: prepared with another feature list; prepare it again with --force")
            else:
                print(f"{root.name}: already prepared")
            continue
        perception = perception or Perception(Yolo(args.detector, conf=0.05), step=args.k)
        s = prepare(root, perception, args.k)
        print(f"\r{root.name} ({s['difficulty']}): {s['decisions']} decisions in {s['seconds']:.0f} s; KOs scored "
              f"{s['ko_scored']}, lost {s['ko_lost']}; damage dealt {s['dealt']:.0f}, taken {s['taken']:.0f}; "
              f"{s['lost_presses']} presses not expressed; {s['missing']} rows without a picture")
    return 0


if __name__ == "__main__":
    sys.exit(main())
