"""Record your play as demonstrations (research/DESIGN.md section 8).

Start Brawlhalla in training mode (Wu Shang vs a Mordex bot, Brawlhaven), **not paused**, with the
bot at the level you name below. Run this, click into the game during the countdown, and play.
Press **F10** to stop (or wait for --minutes).

    python scripts/record_demos.py --difficulty easy --minutes 10

Overlays are part of the frames, so every session uses the same ones: by default none (no
hitboxes, hurtboxes, stun, damage numbers, combo counter, force or Power Viewer). Name any other
setting with --overlays (e.g. ``--overlays damage`` for the HUD check); it is kept in meta.json.

Every 1/60 s it records the screen (half resolution, JPEG), the fight keys you hold, and the fight
keys you pressed since the previous frame (a jump or an attack starts on the press). Every 2 s it
also keeps a full-resolution lossless still, for labelling (scripts/label.py). The keyboard is
read 1,000 times a second, so short taps are not missed.

Output: data/demos/<date_time>/ in the recording format of brawl/record.py. Recordings are large
(about 15 GB per hour, 2.5 GB per 10 minutes): set BRAWL_DATA to a folder outside OneDrive.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # sys and pathlib do not load torch

import brawl.game.capture as capture  # noqa: E402  first: the screen duplicator must exist before torch loads

import argparse  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402
from queue import Full, Queue  # noqa: E402

from brawl.game.keys import KEY_ORDER, KeyReader, to_mask  # noqa: E402
from brawl.paths import DATA, REPO_ROOT  # noqa: E402
from brawl.record import Writer  # noqa: E402

LEVELS = ("easy", "medium", "hard", "extreme", "chosen")


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def record(screen, reader: KeyReader, writer: Writer, *, rate_hz: float, seconds: float,
           still_every_s: float, now=time.perf_counter, sleep=time.sleep) -> dict:
    """The capture loop: one row every 1 / rate_hz s until F10 or ``seconds``. Encoding and disk
    writes happen on a second thread so the schedule holds."""
    queue: Queue = Queue(maxsize=int(rate_hz * 4))

    def write() -> None:
        while (item := queue.get()) is not None:
            image, row, keep_still = item
            index = writer.append(image, **row)
            if keep_still:
                writer.still(index, image)

    thread = threading.Thread(target=write, name="writer")
    thread.start()
    period = 1.0 / rate_hz
    stats = {"rows": 0, "late": 0, "skipped": 0, "dropped": 0, "stale": 0, "stills": 0, "rows_with_keys": 0}
    screen.grab(fresh=True)  # the first frame
    t0 = now()
    k, next_still = 0, 0.0
    try:
        while True:
            wait = t0 + k * period - now()
            if wait > 0:
                sleep(wait)
            elif wait < -3 * period:  # far behind (the system stalled): skip ahead, do not burst
                behind = int(-wait / period)
                stats["skipped"] += behind
                k += behind
            frame = screen.grab(fresh=False)
            held, pressed = reader.take()
            t = now() - t0
            stats["late"] += int(t - k * period > period)
            stats["stale"] += int(not frame.fresh)
            keep_still = t >= next_still
            if keep_still:
                next_still += still_every_s
                stats["stills"] += 1
            row = {"t": t, "keys": to_mask(held), "pressed": to_mask(pressed), "fresh": frame.fresh}
            try:
                queue.put_nowait((frame.image, row, keep_still))
                stats["rows"] += 1
                stats["rows_with_keys"] += int(bool(held))
            except Full:
                stats["dropped"] += 1
            k += 1
            if reader.stop_requested() or t >= seconds:
                break
    finally:
        queue.put(None)
        thread.join()
    stats["seconds"] = now() - t0
    return stats


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--difficulty", required=True, choices=LEVELS, help="the bot's level, as set in the game")
    p.add_argument("--minutes", type=float, default=10.0)
    p.add_argument("--rate", type=float, default=60.0, help="rows per second")
    p.add_argument("--still-every", type=float, default=2.0, help="seconds between full-resolution stills")
    p.add_argument("--overlays", default="none",
                   help="training-mode overlays shown while recording, comma separated (none for no overlays); "
                        "keep the same in every session: they are part of the frames")
    p.add_argument("--note", default="", help="free text kept in meta.json")
    p.add_argument("--countdown", type=int, default=5)
    args = p.parse_args(argv)
    print(capture.status())
    if "onedrive" in str(DATA).lower():
        print(f"note: recording into {DATA}, inside OneDrive; set BRAWL_DATA to a folder outside it "
              f"(and open a new terminal after setx)")
    screen = capture.Screen(timeout_s=0.05)
    out = DATA / "demos" / datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {"source": "human", "regime": "realtime", "started": datetime.now().isoformat(timespec="seconds"),
            "setting": {"map": "Brawlhaven (small)", "agent": "Wu Shang", "opponent": "Mordex",
                        "difficulty": args.difficulty, "mode": "training",
                        "overlays": sorted(o for o in args.overlays.split(",") if o and o != "none")},
            "rate_hz": args.rate, "still_every_s": args.still_every, "key_order": list(KEY_ORDER),
            "columns_doc": {"t": "seconds since the first row", "keys": "fight keys held (bitmask over key_order)",
                            "pressed": "fight keys pressed since the previous row (bitmask)",
                            "fresh": "the game presented a new frame since the previous row"},
            "git": git_commit(), "note": args.note}
    reader = KeyReader()
    reader.start()
    for i in range(args.countdown, 0, -1):
        print(f"click into the game window and play... {i}", flush=True)
        time.sleep(1)
    print("recording: press F10 to stop", flush=True)
    writer = Writer(out, meta)
    try:
        stats = record(screen, reader, writer, rate_hz=args.rate, seconds=args.minutes * 60,
                       still_every_s=args.still_every)
    finally:
        reader.stop()
    writer.meta["stats"] = stats
    writer.close()
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    rate = stats["rows"] / max(stats["seconds"], 1e-9)
    print(f"\nrecorded {stats['rows']} rows in {stats['seconds']:.0f} s ({rate:.1f}/s), {stats['stills']} stills, "
          f"{size:.0f} MB -> {out}")
    print(f"late rows {stats['late']}, skipped {stats['skipped']}, dropped {stats['dropped']}, "
          f"repeated frames {stats['stale']}, rows with keys held {stats['rows_with_keys']}")
    if writer.still_failures:
        print(f"WARNING: {writer.still_failures} stills could not be written (is {out} in a synced folder?)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
