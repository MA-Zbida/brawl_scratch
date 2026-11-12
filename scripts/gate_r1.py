"""Gate R1: the HUD reader and the event ledger, replayed over recordings and checked by hand counts.

    python scripts/gate_r1.py

Every frame of each recording goes through brawl.sense.hud and brawl.ledger, exactly as the
environment will call them. Three checks, written to results/gates/r1.md:

1. **KOs** against the score numbers on the HUD, read by a person on the first and last frame of
   each recording (HAND_SCORES). The game shows a red badge on the fighter who lost a stock (-1,
   or -3 when nobody is credited: a self-destruct) and a cyan "+2" on the fighter credited. The
   badges counted over the recording must turn the first score into the last, and the ledger's
   KOs must equal the red badges.
2. **Damage** against the damage-number overlay, read by a person for single hits (HAND_HITS).
3. **Consistency:** frames the reader could not read, and the reading on each lossless still
   against the half-size JPEG frame of the same row. The live game is read from lossless frames,
   where a ring is one exact colour. In a recording's JPEG frames the dark reds wobble by up to 8
   levels, which is 5 damage above 150 (64 levels of red span 50 damage); readings that fall
   without reaching zero are that wobble, counted and ignored by the ledger.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from brawl.ledger import Ledger  # noqa: E402
from brawl.paths import DATA, RESULTS  # noqa: E402
from brawl.record import Recording  # noqa: E402
from brawl.sense import hud  # noqa: E402

# Score numbers read on the HUD: recording -> ((mordex, wu_shang) on the first frame, on the last).
HAND_SCORES = {"20261106_151844": ((0, 0), (-15, 27)), "20261106_155225": ((-17, 31), (-5, 31)),
               "20261109_184230": ((0, 0), (-16, 47)), "20261109_211132": ((0, -12), (-4, -1))}
# Damage numbers read from the overlay: recording -> [(row the ring changed, fighter hit, damage)].
HAND_HITS = {"20261109_211132": [(84, "mordex", 13.0), (127, "mordex", 8.0), (141, "mordex", 7.0),
                                 (186, "mordex", 8.0), (201, "mordex", 7.0), (484, "mordex", 17.0),
                                 (618, "mordex", 13.0), (770, "mordex", 11.34), (956, "mordex", 17.0),
                                 (1342, "mordex", 17.0)]}
PASS = {"unreadable": 0.02, "hit_error": 2.0, "still_error": 2.0}  # damage, on JPEG frames


def replay(root: Path) -> dict:
    rec, ledger = Recording(root), Ledger()
    stills = {int(p.stem): p for p in rec.stills()}
    badge = {"mordex": None, "wu_shang": None}
    out = {"frames": 0, "unreadable": 0, "misread": 0, "dealt": 0.0, "taken": 0.0, "kos": [],
           "lost": {"mordex": 0, "wu_shang": 0}, "scored": {"mordex": 0, "wu_shang": 0},
           "gains": [], "still_errors": [], "difficulty": rec.meta["setting"].get("difficulty", "?")}
    for row, frame in enumerate(rec.frames()):
        if frame is None:
            continue
        reading = hud.read(frame)
        e = ledger.update(reading, row)
        out["frames"] += 1
        out["unreadable"] += reading is None
        out["misread"] += e.misread
        out["dealt"] += e.damage_dealt
        out["taken"] += e.damage_taken
        out["kos"] += [(row, "mordex")] * e.ko_scored + [(row, "wu_shang")] * e.ko_lost
        if e.damage_dealt or e.damage_taken:
            out["gains"].append((row, e.damage_dealt, e.damage_taken))
        for who, slot in (reading or {}).items():  # badges counted apart from the ledger
            if slot.badge and slot.badge != badge[who]:
                out[slot.badge][who] += 1
            badge[who] = slot.badge
        if row in stills and reading is not None:
            import cv2

            full = hud.read(cv2.imread(str(stills[row])))
            if full is not None:
                out["still_errors"] += [abs(full[w].damage - reading[w].damage) for w in full]
    return out


def main() -> int:
    lines = ["# Gate R1 (HUD reader and ledger)", "",
             f"{datetime.now():%Y-%m-%d %H:%M}. Every frame of each recording, read by brawl.sense.hud and "
             "turned into events by brawl.ledger.", "",
             "| Recording | Bot | Frames | Unreadable | Falls not to zero (JPEG) | Mordex KOs | Wu Shang KOs | "
             "Damage dealt / taken | Score check |", "|---|---|---|---|---|---|---|---|---|"]
    ok, hits, still_errors = True, [], []
    for root in sorted(p for p in (DATA / "demos").iterdir() if (p / "meta.json").exists()):
        r = replay(root)
        m, w = (sum(who == x for _, who in r["kos"]) for x in ("mordex", "wu_shang"))
        check = "not read"
        if root.name in HAND_SCORES:
            (m0, w0), (m1, w1) = HAND_SCORES[root.name]
            lost, scored = r["lost"], r["scored"]  # a loss the other was not credited with costs 3
            predicted = (m0 - lost["mordex"] - 2 * (lost["mordex"] - scored["wu_shang"]) + 2 * scored["mordex"],
                         w0 - lost["wu_shang"] - 2 * (lost["wu_shang"] - scored["mordex"]) + 2 * scored["wu_shang"])
            good = predicted == (m1, w1) and (m, w) == (r["lost"]["mordex"], r["lost"]["wu_shang"])
            check = "ok" if good else (f"**fail**: badges {r['lost']} lost, {r['scored']} scored predict "
                                       f"{predicted}, HUD {(m1, w1)}")
            ok = ok and good
        share = r["unreadable"] / max(r["frames"], 1)
        ok = ok and share <= PASS["unreadable"]
        lines.append(f"| {root.name} | {r['difficulty']} | {r['frames']} | {r['unreadable']} ({share:.1%}) | "
                     f"{r['misread']} | {m} | {w} | {r['dealt']:.0f} / {r['taken']:.0f} | {check} |")
        still_errors += r["still_errors"]
        for row, who, truth in HAND_HITS.get(root.name, []):
            got = sum(g[1 if who == "mordex" else 2] for g in r["gains"] if row - 2 <= g[0] < row + 12)
            hits.append((root.name, row, who, truth, got))
        print(lines[-1], flush=True)
    worst_hit = max(abs(t - g) for *_, t, g in hits)
    worst_still = float(np.percentile(still_errors, 90)) if still_errors else float("nan")
    ok = ok and worst_hit <= PASS["hit_error"] and worst_still <= PASS["still_error"]
    lines += ["", f"Lossless stills against the half-size JPEG frame of the same row: {len(still_errors)} "
              f"readings, 90% within {worst_still:.2f} damage (pass at <= {PASS['still_error']}), 99% within "
              f"{np.percentile(still_errors, 99):.2f}, largest {max(still_errors):.2f} (dark reds, JPEG).", "",
              f"Single hits against the damage overlay (largest error {worst_hit:.2f}, pass at <= "
              f"{PASS['hit_error']}):", "", "| Recording | Row | Hit on | Overlay | Ledger |", "|---|---|---|---|---|"]
    lines += [f"| {n} | {row} | {who} | {t:.2f} | {g:.2f} |" for n, row, who, t, g in hits]
    lines[0] += f": {'PASS' if ok else 'FAIL'}"
    path = RESULTS / "gates" / "r1.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-len(hits) - 6:]), f"\n-> {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
