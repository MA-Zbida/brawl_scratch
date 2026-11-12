"""Gate P1: score a detector candidate on the labelled test frames (docs/labelling.md).

    python scripts/gate_p1.py old_yolo
    python scripts/gate_p1.py runs/detector/<name>/weights/best.engine              # single frames
    python scripts/gate_p1.py runs/detector/<name>/weights/best.engine --track 30   # with the tracker

Reads the Roboflow export (default <BRAWL_DATA>/labelling/export_kp_final), runs the candidate on each test
frame's lossless still, and writes results/gates/p1_<candidate>.md. Per-frame rows and pictures of
the worst frames (missed or swapped fighters, largest position errors) go to
runs/gates/p1/<candidate>/ for inspection.
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brawl.paths import DATA, RESULTS, RUNS  # noqa: E402
from brawl.sense.truth import FIGHTERS, WEAPONS, problems, read_export, score_frame, summarise  # noqa: E402

COLOURS = {"wu_shang": (0, 200, 0), "mordex": (0, 0, 230), "weapon_spawn": (0, 165, 255),
           "weapon_thrown": (255, 0, 255), "offscreen": (255, 255, 0)}


def candidate(name: str, conf: float):
    from brawl.sense import detectors

    if name == "old_yolo":
        return detectors.OldYolo(conf=conf)
    if name.endswith((".pt", ".engine")) and Path(name).exists():
        return detectors.Yolo(Path(name), conf=conf)
    raise SystemExit(f"unknown candidate {name!r}: old_yolo, a trained .pt (scripts/train_detector.py) "
                     "or its .engine (scripts/export_detector.py)")


def tracked(frames, detector, demos: Path, before: int, step: int = 1) -> dict:
    """Each test frame's fighters and weapons from brawl.sense.track, after the tracker has followed
    the ``before`` recording frames that lead to it, one every ``step`` game frames (half-size JPEG,
    scaled up; the test frame itself is its lossless still)."""
    import cv2

    from brawl.record import Recording
    from brawl.sense import hud
    from brawl.sense.track import Tracker
    out, carried = {}, 0
    for recording in sorted({f.recording for f in frames}):
        labelled = {f.row: f for f in frames if f.recording == recording}
        lead = {row - step * j for row in labelled for j in range(1, before + 1)} - set(labelled)
        tracker = None
        for row, frame in enumerate(Recording(demos / recording).frames(rows=lead)):
            if row > max(labelled):
                break
            if row in labelled:
                image = cv2.imread(str(labelled[row].still))
            elif row in lead:
                if frame is None:
                    continue
                image = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_CUBIC)
            else:
                if not any(row < r <= row + step * before for r in labelled):
                    tracker = None  # between test frames: the next one starts from nothing
                continue
            tracker = tracker or Tracker(step)
            boxes = detector(image)
            lost = {who for who, slot in (hud.read(image) or {}).items() if slot.badge == "lost"}
            fighters = tracker.update(boxes, lost)
            if row in labelled:
                carried += sum(not f.seen for f in fighters.values())
                out[recording, row] = [f.box for f in fighters.values()] + tracker.weapons
                print(f"\rtracking {len(out)}/{len(frames)}", end="", flush=True)
    print(f"\n{carried} fighters on test frames were carried, not seen")
    return out


def colour(cls: str):
    return next((c for k, c in COLOURS.items() if cls.startswith(k)), (255, 255, 255))


def picture(image, truth, predicted, path: Path, title: str) -> None:
    import cv2

    out = image.copy()
    for b in truth:      # labels: thin, point as a ring
        cv2.rectangle(out, (int(b.x0), int(b.y0)), (int(b.x1), int(b.y1)), colour(b.cls), 1)
        cv2.circle(out, tuple(map(int, b.position)), 6, colour(b.cls), 1)
    for b in predicted:  # predictions: thick, point as a dot, with the class
        cv2.rectangle(out, (int(b.x0), int(b.y0)), (int(b.x1), int(b.y1)), colour(b.cls), 3)
        cv2.circle(out, tuple(map(int, b.position)), 4, colour(b.cls), -1)
        cv2.putText(out, f"{b.cls} {b.score:.2f}", (int(b.x0), int(b.y0) - 6), 0, 0.6, colour(b.cls), 2)
    cv2.putText(out, title, (20, 1060), 0, 1.0, (0, 0, 0), 3)
    cv2.imwrite(str(path), out)


def report(name: str, detector, s: dict, frames: int, export: Path) -> list[str]:
    ok = all(c[3] for c in s["checks"])
    lines = [f"# Gate P1 ({name}): {'PASS' if ok else 'FAIL'}", "",
             f"{datetime.now():%Y-%m-%d %H:%M}. {frames} test frames from {export}. Thresholds are "
             "provisional (research/CHECKLIST.md). A fighter is found when the predicted box of that "
             "identity has its position inside the labelled box; the error is the distance from the labelled "
             "torso point to the predicted one, or to the box centre for a candidate without points "
             "(1920 x 1080 pixels).", "",
             "| Check | Value | Pass at | |", "|---|---|---|---|"]
    for label, value, threshold, passed in s["checks"]:
        better = ">=" if label.endswith(("found", "recall", "precision", "weapon")) else "<="
        lines.append(f"| {label} | {value:.3f} | {better} {threshold} | {'ok' if passed else '**fail**'} |")
    lines += ["", "| Fighter | Labelled | Found | Missed | Swapped | False | Error median / p95 / max (px) |",
              "|---|---|---|---|---|---|---|"]
    for who in FIGHTERS:
        f = s["fighters"][who]
        lines.append(f"| {who} | {f['labelled']} | {f['found']} | {f['missed']} | {f['swaps']} | {f['false']} | "
                     f"{f['error_median_px']:.1f} / {f['error_p95_px']:.1f} / {f['error_max_px']:.1f} |")
    lines += ["", "| Weapon class | Labelled | Predicted | Matched | Recall | Precision | Error median (px) |",
              "|---|---|---|---|---|---|---|"]
    for cls in WEAPONS:
        w = s["weapons"][cls]
        lines.append(f"| {cls} | {w['labelled']} | {w['predicted']} | {w['matched']} | {w['recall']:.3f} | "
                     f"{w['precision']:.3f} | {w['error_median_px']:.1f} |")
    if detector.reports_held:
        lines += ["", "Held weapon (labelled -> predicted, found fighters):", ""]
        for who in FIGHTERS:
            for (t, p), n in sorted(s["fighters"][who]["confusion"].items()):
                lines.append(f"- {t} -> {p}: {n}")
    else:
        lines += ["", "This candidate does not read the held weapon; the held-weapon checks do not apply."]
    return lines


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("candidate", help="old_yolo, or the weights of a trained detector (.pt)")
    p.add_argument("--export", type=Path, default=DATA / "labelling" / "export_kp_final")
    p.add_argument("--demos", type=Path, default=DATA / "demos")
    p.add_argument("--split", default="test")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--worst", type=int, default=12, help="pictures of the worst frames to keep")
    p.add_argument("--track", type=int, default=0, metavar="FRAMES",
                   help="score the tracker's fighters (brawl.sense.track) after it has followed this many "
                        "recording frames before each test frame; 0 scores the detector on single frames")
    p.add_argument("--step", type=int, default=1, help="with --track: game frames between the frames the tracker sees "
                                                       "(3 = what the environment gives it at k = 3)")
    args = p.parse_args(argv)

    import cv2

    frames = read_export(args.export, args.demos, splits=(args.split,))
    for problem in problems(frames):
        print(f"label problem, fix in Roboflow: {problem}")
    detector = candidate(args.candidate, 0.05 if args.track else args.conf)  # the tracker weighs doubtful boxes
    name = getattr(detector, "name", args.candidate) + (f"_track{args.track}" if args.track else "") + (
        f"_step{args.step}" if args.track and args.step != 1 else "")
    tracks = tracked(frames, detector, args.demos, args.track, args.step) if args.track else None
    out = RUNS / "gates" / "p1" / name
    out.mkdir(parents=True, exist_ok=True)
    results, rows = [], []
    for i, frame in enumerate(frames):
        image = cv2.imread(str(frame.still))
        predicted = tracks[frame.recording, frame.row] if tracks else detector(image)
        r = score_frame(frame.boxes, predicted, detector.reports_held)
        results.append(r)
        bad = sum(100 * (f["labelled"] and not f["found"]) + 100 * f["swap"] + (f["error"] or 0)
                  for f in r["fighters"].values())
        rows.append({"recording": frame.recording, "row": frame.row, "badness": bad,
                     **{f"{who}_{k}": r["fighters"][who][k] for who in FIGHTERS
                        for k in ("labelled", "found", "swap", "error")},
                     **{f"{c}_{k}": r["weapons"][c][k] for c in WEAPONS for k in ("labelled", "predicted", "matched")}})
        rows[-1]["_frame"], rows[-1]["_predicted"] = frame, predicted
        print(f"\r{i + 1}/{len(frames)}", end="", flush=True)
    print()
    s = summarise(results, detector.reports_held)
    for old in out.glob("worst_*.png"):
        old.unlink()
    for rank, row in enumerate(sorted(rows, key=lambda r: -r["badness"])[:args.worst]):
        if row["badness"] <= 0:
            break
        frame = row["_frame"]
        picture(cv2.imread(str(frame.still)), frame.boxes, row["_predicted"],
                out / f"worst_{rank:02d}_{frame.recording}__{frame.row:07d}.png",
                f"{frame.recording} row {frame.row}: thin = label, thick = {name}")
    with open(out / "frames.csv", "w", newline="", encoding="utf-8") as f:
        fields = [k for k in rows[0] if not k.startswith("_")]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    lines = report(name, detector, s, len(frames), args.export)
    if args.track:
        lines[2] += (f" Fighters and weapons are the tracker's (brawl.sense.track), after the {args.track} recording "
                     "frames before each test frame; a fighter or spawn marker it carries through a frame without a "
                     f"box counts as predicted. The tracker sees one frame in {args.step}.")
    path = RESULTS / "gates" / f"p1_{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n-> {path}\n-> {out}")
    return 0 if all(c[3] for c in s["checks"]) else 1


if __name__ == "__main__":
    sys.exit(main())
