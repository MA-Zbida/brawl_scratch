"""Label what Mordex is doing, from the fighter crops saved in the night traces (the fighter-state network).

    python scripts/label_states.py              # label (the clips are cut from the traces the first time)
    python scripts/label_states.py --prepare    # only cut the clips (about 10 minutes)
    python scripts/label_states.py --stats      # how many of each label so far

A clip is 8 decisions of Mordex's 96 px crop (0.4 s), from 5 decisions before the labelled one to 2
after it. The window shows the 8 frames in a row (the labelled one framed in green) and, larger, the clip
playing in a loop. Label the **framed** frame; the frames after it are there to help (an attack that is
starting), the network will only see the past.

    1  neutral     standing, walking, running, jumping, falling: no attack, no dodge, not hurt
    2  attacking   any attack or weapon throw, from its wind-up to the end of its swing
    3  dodging     a dodge on the ground or in the air (spot, side or down dodge)
    4  recovering  the up + heavy recovery move, or a ground pound (down + heavy in the air)
    5  hurt        hit or flying from a hit (hitstun)
    0 / space      cannot tell (hidden, blurred, two fighters overlap): skipped, not used
    b / backspace  back one clip (to change a label)
    q / Esc        quit; everything is saved after every key, and the next run resumes

The clips are drawn half from night 4 (Medium) and half from night 5 (Hard), shuffled, and over-sample the
moments that matter: 35% in the half second before Wu Shang takes damage, 15% just after Mordex took
damage, 15% with Mordex off the stage, 35% anywhere. Labels: <BRAWL_DATA>/labels/fighter_state/labels.csv.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brawl.paths import DATA  # noqa: E402

RUNS = [ROOT / "runs/advised_night4/default/seed0", ROOT / "runs/advised_night5/default/seed0"]
OUT = DATA / "labels" / "fighter_state"
BEFORE, AFTER = 5, 2  # decisions shown around the labelled one
LENGTH = BEFORE + 1 + AFTER
WHO = 1  # crops[:, 1] is Mordex (brawl.observation.OPPONENT)
LABELS = {ord("1"): "neutral", ord("2"): "attacking", ord("3"): "dodging", ord("4"): "recovering", ord("5"): "hurt",
          ord("0"): "unclear", ord(" "): "unclear"}
STRATA = {"before_hit": 0.35, "after_dealt": 0.15, "offstage": 0.15, "anywhere": 0.35}


def candidates(obs: np.ndarray, events: np.ndarray, features: list[str]) -> dict[str, np.ndarray]:
    """Decision indices of one shard by stratum; only where Mordex is seen and the clip stays in the shard."""
    f = {n: i for i, n in enumerate(features)}
    n = len(obs)
    ok = np.zeros(n, bool)
    ok[BEFORE:n - AFTER] = True
    seen = obs[:, f["opponent_present"]] > 0.5
    for d in range(-BEFORE, AFTER + 1):  # seen over the whole clip
        ok &= np.roll(seen, -d)
    taken, dealt = events[:, 1] > 0, events[:, 0] > 0
    soon = np.zeros(n, bool)
    for d in range(0, 10):  # Wu Shang takes damage within 0.5 s
        soon |= np.roll(taken, -d)
    after = np.zeros(n, bool)
    for d in range(1, 6):  # Mordex took damage in the last 0.25 s
        after |= np.roll(dealt, d)
    off = (obs[:, f["opponent_offstage"]] > 0.5) | (obs[:, f["opponent_below"]] > 0.5)
    return {"before_hit": np.flatnonzero(ok & soon), "after_dealt": np.flatnonzero(ok & after),
            "offstage": np.flatnonzero(ok & off), "anywhere": np.flatnonzero(ok)}


def prepare(runs: list[Path], total: int, out: Path, seed: int = 0, log=print) -> Path:
    """Draw ``total`` clips over ``runs`` (equal shares) and save their crops to out/clips.npz."""
    rng = np.random.default_rng(seed)
    crops, source, stratum = [], [], []
    for run in runs:
        shards = sorted((run / "trace").glob("shard_*.npz"))
        assert shards, f"no trace in {run}"
        per_shard = {k: round(total / len(runs) * share / len(shards) + 0.5) for k, share in STRATA.items()}
        for s_i, path in enumerate(shards):
            z = np.load(path)
            if "crops" not in z.files:
                continue
            obs, events, steps = z["obs"].astype(np.float32), z["events"], z["step"]
            picked: dict[int, str] = {}
            for name, idx in candidates(obs, events, list(z["features"])).items():
                idx = np.setdiff1d(idx, list(picked))
                for i in rng.choice(idx, size=min(per_shard[name], len(idx)), replace=False):
                    picked[int(i)] = name
            if picked:
                c = z["crops"]  # decompressed once per shard, released after it
                for i, name in picked.items():
                    crops.append(c[i - BEFORE:i + AFTER + 1, WHO].copy())  # a copy: a view would keep the whole shard (110 MB) alive
                    source.append((run.parts[-3], int(steps[i])))
                    stratum.append(name)
            if s_i % 10 == 0:
                log(f"  {run.parts[-3]}: {s_i + 1}/{len(shards)} shards, {len(crops)} clips")
    order = rng.permutation(len(crops))
    out.mkdir(parents=True, exist_ok=True)
    path = out / "clips.npz"
    np.savez(path, crops=np.stack(crops)[order], run=np.array([s[0] for s in source])[order],
             step=np.array([s[1] for s in source])[order], stratum=np.array(stratum)[order],
             before=BEFORE, after=AFTER)
    log(f"-> {path}: {len(order)} clips")
    return path


def read_labels(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    out: dict[int, str] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["label"] == "undo":
                out.pop(int(row["clip"]), None)
            else:
                out[int(row["clip"])] = row["label"]
    return out


def render(clip: np.ndarray, tick: int, header: str) -> np.ndarray:
    """The strip of frames (the labelled one framed) above the clip playing, as a BGR image."""
    import cv2

    small, big = 150, 384
    strip = []
    for i, frame in enumerate(clip):
        im = cv2.cvtColor(cv2.resize(frame, (small, small), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
        if i == BEFORE:
            cv2.rectangle(im, (0, 0), (small - 1, small - 1), (0, 220, 0), 4)
        strip.append(im)
    strip = np.hstack(strip)
    frame_i = tick % (LENGTH + 4)  # a short hold on the last frame before the loop restarts
    frame_i = min(frame_i, LENGTH - 1)
    play = cv2.cvtColor(cv2.resize(clip[frame_i], (big, big), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    if frame_i == BEFORE:
        cv2.rectangle(play, (0, 0), (big - 1, big - 1), (0, 220, 0), 5)
    canvas = np.full((40 + small + 10 + big + 40, strip.shape[1], 3), 30, np.uint8)
    canvas[40:40 + small] = strip
    x0 = (strip.shape[1] - big) // 2
    canvas[50 + small:50 + small + big, x0:x0 + big] = play
    cv2.putText(canvas, header, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
    legend = "1 neutral  2 attacking  3 dodging  4 recovering  5 hurt  0 unclear  b back  q quit"
    cv2.putText(canvas, legend, (10, canvas.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    return canvas


def label(clips_path: Path, labels_path: Path) -> None:
    import cv2

    z = np.load(clips_path)
    crops = z["crops"]
    done = read_labels(labels_path)
    new = not labels_path.exists()
    f = open(labels_path, "a", newline="", encoding="utf-8")
    rows = csv.writer(f)
    if new:
        rows.writerow(["clip", "label", "run", "step", "time"])
    history: list[int] = []
    i = next((k for k in range(len(crops)) if k not in done), len(crops))
    window = "Mordex: label the framed frame"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    tick = 0
    try:
        while i < len(crops):
            counts = {v: list(done.values()).count(v) for v in ("neutral", "attacking", "dodging", "recovering", "hurt")}
            header = f"clip {i + 1}/{len(crops)}  labelled {len(done)}  " + "  ".join(f"{k[:4]} {v}" for k, v in counts.items())
            cv2.imshow(window, render(crops[i], tick, header))
            key = cv2.waitKey(50) & 0xFF  # 20 frames a second: the game's decision rate
            tick += 1
            if key in (ord("q"), 27):
                break
            if key in (ord("b"), 8) and history:
                i = history.pop()
                rows.writerow([i, "undo", z["run"][i], int(z["step"][i]), round(time.time())])
                f.flush()
                done.pop(i, None)
                tick = 0
            elif key in LABELS:
                rows.writerow([i, LABELS[key], z["run"][i], int(z["step"][i]), round(time.time())])
                f.flush()
                done[i] = LABELS[key]
                history.append(i)
                i = next((k for k in range(i + 1, len(crops)) if k not in done), len(crops))
                tick = 0
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        f.close()
        cv2.destroyAllWindows()
    print(f"{len(done)} clips labelled; saved in {labels_path}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prepare", action="store_true", help="only cut the clips")
    p.add_argument("--clips", type=int, default=4000, help="clips to cut (default 4000)")
    p.add_argument("--stats", action="store_true", help="count the labels so far")
    args = p.parse_args(argv)
    clips_path, labels_path = OUT / "clips.npz", OUT / "labels.csv"
    if args.stats:
        done = read_labels(labels_path)
        z = np.load(clips_path)
        for name in ("neutral", "attacking", "dodging", "recovering", "hurt", "unclear"):
            ids = [k for k, v in done.items() if v == name]
            by_run = {r: int(sum(z["run"][k] == r for k in ids)) for r in np.unique(z["run"])}
            print(f"{name:11s} {len(ids):5d}  {by_run}")
        return 0
    if not clips_path.exists() or args.prepare:
        print(f"cutting {args.clips} clips from {', '.join(r.parts[-3] for r in RUNS)} ...")
        prepare(RUNS, args.clips, OUT)
    if not args.prepare:
        label(clips_path, labels_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
