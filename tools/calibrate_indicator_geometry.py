#!/usr/bin/env python
"""Measure the indicator-to-character association constants from labelled data.

`feature_extractor/memory/detection_schema.py` decides which `character` carries the
`indicator_self` triangle with

    score = H_WEIGHT * |dx| + V_WEIGHT * max(0, dy)

rejecting a match when `score > INDICATOR_MAX_MATCH_SCORE`, or when the character sits
more than `INDICATOR_ABOVE_TOLERANCE` above the indicator. Those constants started life
as guesses. This measures them.

Why not just take min/max of the observed scores
------------------------------------------------
Because deciding which character is the *true* partner is itself the problem being
calibrated. A naive "nearest character below the indicator" rule guesses wrong in exactly
the crowded frames that matter, and a single wrong assignment poisons both tails: the true
set gets an enormous outlier, the false set gets a near-zero one, and any threshold
derived from extremes collapses.

So this script does two things differently:

1. **Ambiguity rejection.** A frame contributes to the calibration set only when the best
   candidate is clearly better than the runner-up (`--ratio`). Ambiguous frames are
   counted and reported rather than guessed at.
2. **Robust statistics.** Thresholds come from percentiles, never from min/max.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CHARACTER_ID = 0
INDICATOR_ID = 1
WEAPON_ID = 2

# Weights are a design choice, not a measurement: horizontal alignment is the strong cue
# because the triangle sits directly above its character.
H_WEIGHT = 2.0
V_WEIGHT = 1.0

# How many top-end true scores the threshold percentile must stay clear of, so a
# handful of mis-pairings cannot drag it.
OUTLIER_HEADROOM = 5

# Single source of truth for the margin, so the CLI cannot drift from the function.
_DEFAULT_MARGIN = 1.15


@dataclass
class Box:
    cls: int
    cx: float
    cy: float
    w: float
    h: float


@dataclass
class Calibration:
    n_frames: int = 0
    n_usable: int = 0
    n_ambiguous: int = 0
    n_no_candidate: int = 0
    n_multi_indicator: int = 0
    n_gate_rejected: int = 0
    dx: np.ndarray = field(default_factory=lambda: np.array([]))
    dy: np.ndarray = field(default_factory=lambda: np.array([]))
    true_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    false_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    char_h: np.ndarray = field(default_factory=lambda: np.array([]))
    char_w: np.ndarray = field(default_factory=lambda: np.array([]))


def read_label(path: Path) -> list[Box]:
    if not path.exists():
        return []
    out: list[Box] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").strip().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            out.append(Box(int(float(parts[0])), *(float(v) for v in parts[1:5])))
    return out


def score(dx: float, dy: float) -> float:
    return H_WEIGHT * abs(dx) + V_WEIGHT * max(0.0, dy)


def calibrate(
    label_files: Iterable[Path],
    *,
    ratio: float = 1.6,
    max_dy: float = 0.25,
) -> Calibration:
    """Pair each indicator with its character, skipping frames that are ambiguous.

    ``ratio`` is how much better the best candidate must be than the runner-up for the
    frame to be trusted. ``max_dy`` bounds how far below the indicator its character can
    plausibly sit.
    """
    cal = Calibration()
    dx_list: list[float] = []
    dy_list: list[float] = []
    true_list: list[float] = []
    false_list: list[float] = []
    h_list: list[float] = []
    w_list: list[float] = []

    for path in label_files:
        boxes = read_label(path)
        if not boxes:
            continue
        cal.n_frames += 1

        indicators = [b for b in boxes if b.cls == INDICATOR_ID]
        characters = [b for b in boxes if b.cls == CHARACTER_ID]
        if not indicators or not characters:
            continue
        if len(indicators) > 1:
            cal.n_multi_indicator += 1
            continue

        ind = indicators[0]
        # A character can only be the partner if it sits below the indicator and within a
        # plausible vertical reach.
        cands = [c for c in characters if 0.0 <= (c.cy - ind.cy) <= max_dy]
        if not cands:
            cal.n_no_candidate += 1
            continue

        scored = sorted(((score(c.cx - ind.cx, c.cy - ind.cy), c) for c in cands), key=lambda t: t[0])
        best_score, best = scored[0]

        # Ambiguous when the runner-up is nearly as good: guessing here is what corrupts
        # both tails, so the frame is excluded and counted instead.
        if len(scored) > 1 and scored[1][0] < best_score * ratio:
            cal.n_ambiguous += 1
            continue

        cal.n_usable += 1
        dx_list.append(best.cx - ind.cx)
        dy_list.append(best.cy - ind.cy)
        true_list.append(best_score)
        h_list.append(best.h)
        w_list.append(best.w)

        # Only other CANDIDATES are competitors. A character above the indicator is
        # rejected by ABOVE_TOLERANCE, not by the score threshold -- and because the
        # score zeroes its vertical term (max(0, dy) with dy < 0), scoring it here
        # would fabricate a near-zero "false" pair and drag the threshold to nothing.
        for c in cands:
            if c is not best:
                false_list.append(score(c.cx - ind.cx, c.cy - ind.cy))
        cal.n_gate_rejected += len(characters) - len(cands)

    cal.dx = np.asarray(dx_list)
    cal.dy = np.asarray(dy_list)
    cal.true_scores = np.asarray(true_list)
    cal.false_scores = np.asarray(false_list)
    cal.char_h = np.asarray(h_list)
    cal.char_w = np.asarray(w_list)
    return cal


def derive_constants(cal: Calibration, *, margin: float = _DEFAULT_MARGIN) -> dict[str, float]:
    """Derive a threshold that accepts real pairs, then report what it lets in.

    The two failure directions are not symmetric:

    * **Threshold too low** -> a correct match is rejected, identity falls back to
      carry-forward, and ``identity_observed`` reports 0. The policy is told its
      senses are stale. Safe degradation.
    * **Threshold too high** -> a *wrong* character is adopted as the agent, silently,
      and every relational feature inverts. Nothing reports it.

    That asymmetry says: never set the threshold below the true distribution (which
    would cripple identity for no benefit), and never derive it by clamping against
    the false distribution (which is what produced a threshold under the true median).
    Anchor on the true tail, then *measure* the wrong-match rate that implies and
    surface it, rather than silently trading one failure for the other.
    """
    if cal.true_scores.size == 0:
        raise ValueError("no usable pairs; cannot derive constants")

    # Pick a percentile that always leaves at least OUTLIER_HEADROOM samples above it.
    # A fixed p99 is fine at n in the thousands but sits within one rank of the maximum
    # at n in the dozens, where a single mis-pairing still drags it.
    n = int(cal.true_scores.size)
    q = min(99.0, 100.0 * (1.0 - OUTLIER_HEADROOM / max(n, OUTLIER_HEADROOM + 1)))
    max_score = float(np.percentile(cal.true_scores, q)) * margin

    # Characters do occasionally sit a hair above the indicator centre.
    dy_p001 = float(np.percentile(cal.dy, 0.1)) if cal.dy.size else 0.0
    above_tol = float(max(0.01, abs(min(0.0, dy_p001)) * 1.5))

    return {
        "INDICATOR_HORIZONTAL_WEIGHT": H_WEIGHT,
        "INDICATOR_VERTICAL_WEIGHT": V_WEIGHT,
        "INDICATOR_MAX_MATCH_SCORE": round(max_score, 4),
        "INDICATOR_ABOVE_TOLERANCE": round(above_tol, 4),
    }


def threshold_quality(cal: Calibration, max_score: float) -> dict[str, float]:
    """How the derived threshold behaves on both distributions."""
    true_accept = (
        float(np.mean(cal.true_scores <= max_score)) if cal.true_scores.size else float("nan")
    )
    false_admit = (
        float(np.mean(cal.false_scores <= max_score)) if cal.false_scores.size else 0.0
    )
    return {
        "true_accept_rate": true_accept,
        "false_admit_rate": false_admit,
        "n_false_admitted": int(round(false_admit * cal.false_scores.size)),
    }


def _pct(arr: np.ndarray, q: float) -> float:
    return float(np.percentile(arr, q)) if arr.size else float("nan")


def report(cal: Calibration, consts: dict[str, float]) -> str:
    lines: list[str] = []
    add = lines.append

    add("=" * 72)
    add("INDICATOR GEOMETRY CALIBRATION")
    add("=" * 72)
    add(f"frames with labels     : {cal.n_frames}")
    add(f"usable (unambiguous)   : {cal.n_usable}")
    add(f"skipped, ambiguous     : {cal.n_ambiguous}")
    add(f"skipped, no candidate  : {cal.n_no_candidate}")
    add(f"skipped, >1 indicator  : {cal.n_multi_indicator}")
    add(f"chars above indicator  : {cal.n_gate_rejected}  (rejected by ABOVE_TOLERANCE, not by score)")

    if cal.n_usable == 0:
        add("\nNo usable pairs. Check the class ids and the label directory.")
        return "\n".join(lines)

    add("")
    add("offset from indicator centre to its character centre (normalised)")
    add(f"{'':>6}{'p1':>10}{'p25':>10}{'median':>10}{'p75':>10}{'p99':>10}")
    for name, arr in (("dx", cal.dx), ("dy", cal.dy)):
        add(f"{name:>6}{_pct(arr,1):>10.4f}{_pct(arr,25):>10.4f}"
            f"{_pct(arr,50):>10.4f}{_pct(arr,75):>10.4f}{_pct(arr,99):>10.4f}")

    add("")
    add("match score  (competing candidates only)")
    add(f"  true  n={cal.true_scores.size:<6} p50={_pct(cal.true_scores,50):.4f} "
        f"p95={_pct(cal.true_scores,95):.4f} p99={_pct(cal.true_scores,99):.4f}")
    if cal.false_scores.size:
        add(f"  false n={cal.false_scores.size:<6} p01={_pct(cal.false_scores,1):.4f} "
            f"p05={_pct(cal.false_scores,5):.4f} p50={_pct(cal.false_scores,50):.4f}")
    else:
        add("  false n=0  -- no frame had two characters both below the indicator.")
        add("  Separability is UNVERIFIED. Label frames with fighters overlapping;")
        add("  that is the case this whole mechanism exists to handle.")

    q = threshold_quality(cal, consts["INDICATOR_MAX_MATCH_SCORE"])
    add("")
    add(f"at INDICATOR_MAX_MATCH_SCORE = {consts['INDICATOR_MAX_MATCH_SCORE']:.4f}")
    add(f"  correct pairs accepted : {q['true_accept_rate']:6.2%}")
    add(f"  wrong pairs admitted   : {q['false_admit_rate']:6.2%}  ({q['n_false_admitted']} of {cal.false_scores.size})")
    add("")
    add("  Reading these: the matcher takes the LOWEST-scoring candidate, so a wrong pair")
    add("  under the threshold is harmless while the correct character is also detected --")
    add("  it loses the comparison anyway. The threshold only decides the case where the")
    add("  correct character was MISSED, leaving only wrong candidates. So the risk is")
    add("  roughly  P(character missed) x (wrong pairs admitted).")
    if q["true_accept_rate"] < 0.98:
        add("  -> too tight: identity would fall back more often than necessary, for nothing")
    elif q["false_admit_rate"] > 0.15:
        add("  -> too loose. When the real character is missed, a wrong one is adopted almost")
        add("     half the time -- silently, with identity_observed still reporting 1.")
        add("     Lower --margin; past ~1.2 it buys no true acceptance worth this cost.")
    elif q["false_admit_rate"] > 0.02:
        add("  -> acceptable, given it only applies on frames where the character was missed.")
    else:
        add("  -> clean separation; the indicator disambiguates reliably")

    if cal.char_h.size:
        add("")
        add("agent character box (normalised)")
        add(f"  height p25={_pct(cal.char_h,25):.4f}  median={_pct(cal.char_h,50):.4f}  p75={_pct(cal.char_h,75):.4f}")
        add(f"  width  p25={_pct(cal.char_w,25):.4f}  median={_pct(cal.char_w,50):.4f}  p75={_pct(cal.char_w,75):.4f}")
        suggested = _pct(cal.char_h, 50) / 2.0
        add("")
        add(f"  FighterState.height should be about {suggested:.4f}  (median box height / 2)")
        add("  It shifts the box centre down to the feet, and grounded / on_edge /")
        add("  offstage / ledge distance are ALL derived from that shifted point. A stale")
        add("  value silently biases every one of them.")

    add("")
    add("-" * 72)
    add("paste into feature_extractor/memory/detection_schema.py")
    add("-" * 72)
    for k, v in consts.items():
        add(f"{k} = {v}")
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", type=Path, required=True,
                   help="Directory of YOLO .txt label files (searched recursively)")
    p.add_argument("--ratio", type=float, default=1.6,
                   help="Best candidate must beat the runner-up by this factor (default 1.6)")
    p.add_argument("--max-dy", type=float, default=0.25,
                   help="Largest plausible vertical offset from indicator to character")
    p.add_argument("--margin", type=float, default=_DEFAULT_MARGIN,
                   help="Safety multiplier on the true-score tail. Above ~1.2 this buys "
                        "almost no extra true acceptance and admits far more wrong pairs.")
    p.add_argument("--json-out", type=Path, default=None, help="Write the full report as JSON")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.labels.exists():
        print(f"No such directory: {args.labels}", file=sys.stderr)
        return 1

    files = sorted(args.labels.rglob("*.txt"))
    if not files:
        print(f"No .txt label files under {args.labels}", file=sys.stderr)
        return 1
    print(f"reading {len(files)} label files from {args.labels}\n")

    cal = calibrate(files, ratio=args.ratio, max_dy=args.max_dy)
    try:
        consts = derive_constants(cal, margin=args.margin)
    except ValueError as exc:
        print(f"{exc}", file=sys.stderr)
        return 1

    print(report(cal, consts))

    if args.json_out:
        payload = {
            "frames": cal.n_frames,
            "usable": cal.n_usable,
            "ambiguous": cal.n_ambiguous,
            "no_candidate": cal.n_no_candidate,
            "dx": {q: _pct(cal.dx, q) for q in (1, 25, 50, 75, 99)},
            "dy": {q: _pct(cal.dy, q) for q in (1, 25, 50, 75, 99)},
            "true_score_p95": _pct(cal.true_scores, 95),
            "true_score_p99": _pct(cal.true_scores, 99),
            "false_score_p01": _pct(cal.false_scores, 1) if cal.false_scores.size else None,
            "gate_rejected": cal.n_gate_rejected,
            "character_box": {
                "height_median": _pct(cal.char_h, 50),
                "width_median": _pct(cal.char_w, 50),
                "suggested_fighter_height": _pct(cal.char_h, 50) / 2.0 if cal.char_h.size else None,
            },
            "threshold_quality": threshold_quality(cal, consts["INDICATOR_MAX_MATCH_SCORE"]),
            "constants": consts,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
