#!/usr/bin/env python
"""Does the observation contain a signal that precedes a hit, and how many frames early?

This bounds the useful imagination horizon of any world model built on this state.
The detector emits boxes, not animation state, so the agent cannot see startup
frames directly. The one available proxy is that an attack *extends the hitbox*
before it connects, which should show up in `opponent_dw`/`opponent_dh` a few frames
ahead of `self_delta_damage`. If that signal is at chance, no dynamics model can
predict incoming attacks regardless of capacity, and the usable horizon is however
long ballistic motion stays predictable -- not longer.

Method: for each horizon k, treat "a hit lands within the next k frames" as a binary
label and score each candidate feature by AUC (rank-based, so no fitting and no
distributional assumptions). AUC 0.5 is chance; ~0.6 is a weak but real signal.

Distance is the obvious confound -- you get hit when you are close, which we already
know and which no model needs bbox features to learn. So every feature is scored
twice: marginally, and restricted to frames already `in_strike_range`. The second
number is the one that says whether bbox dynamics add anything beyond proximity.

Both directions are measured. Incoming hits (`self_delta_damage`) are the question
that matters for defence; outgoing hits (`op_delta_damage`) act as a positive
control, since the agent's own attacks are caused by actions we recorded and *must*
be predictable if the method works at all.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_extractor.memory.state_spec import StateSpec
from world_model import read_session

FEATURES: tuple[str, ...] = (
    "opponent_dw",
    "opponent_dh",
    "opponent_w",
    "opponent_h",
    "rel_distance",
    "rel_vx",
    "opponent_vx",
    "in_strike_range",
)


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney). Returns 0.5 when either class is empty."""
    pos = int(labels.sum())
    neg = int(labels.size - pos)
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    # Average ranks within ties, or discrete features get spurious separation.
    sorted_scores = scores[order]
    i = 0
    while i < sorted_scores.size:
        j = i
        while j + 1 < sorted_scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = np.mean(ranks[order[i : j + 1]])
        i = j + 1
    return float((ranks[labels.astype(bool)].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def label_within_k(events: np.ndarray, episode: np.ndarray, k: int) -> np.ndarray:
    """True at t when an event occurs in t+1..t+k inside the same episode."""
    n = events.size
    out = np.zeros(n, dtype=bool)
    for offset in range(1, k + 1):
        shifted = np.zeros(n, dtype=bool)
        shifted[: n - offset] = events[offset:]
        same = np.zeros(n, dtype=bool)
        same[: n - offset] = episode[offset:] == episode[: n - offset]
        out |= shifted & same
    return out


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("data/world_replay"))
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3, 5, 8, 10, 15])
    p.add_argument("--damage-threshold", type=float, default=1e-6)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    sessions = sorted(d for d in args.root.iterdir() if d.is_dir()) if args.root.exists() else []
    if not sessions:
        print(f"No sessions under {args.root}", file=sys.stderr)
        return 1

    obs_parts, ep_parts, self_dmg, op_dmg, in_range = [], [], [], [], []
    base = 0
    for session in sessions:
        data = read_session(session)
        keep = data["action"] != -1        # drop terminal state rows
        obs_parts.append(data["obs"][keep])
        ep_parts.append(data["episode_id"][keep].astype(np.int64) + base)
        base += int(data["episode_id"].max()) + 1
        self_dmg.append(data["self_delta_damage"][keep])
        op_dmg.append(data["op_delta_damage"][keep])

    obs = np.concatenate(obs_parts)
    episode = np.concatenate(ep_parts)
    self_dmg = np.concatenate(self_dmg)
    op_dmg = np.concatenate(op_dmg)

    columns = {name: np.asarray([StateSpec.get(row, name) for row in obs], dtype=np.float64) for name in FEATURES}
    exists = np.asarray([StateSpec.get(row, "opponent_exists") for row in obs], dtype=np.float64) > 0.5
    in_range = columns["in_strike_range"] > 0.5

    print("=" * 78)
    print(f"{len(sessions)} sessions, {obs.shape[0]} transitions, {len(np.unique(episode))} episodes")
    print(f"opponent visible: {exists.mean():.1%}   in_strike_range: {in_range.mean():.1%}")

    for direction, damage in (("INCOMING (agent is hit)", self_dmg), ("OUTGOING (agent lands a hit)", op_dmg)):
        events = damage > float(args.damage_threshold)
        print("\n" + "=" * 78)
        print(f"{direction}:  {int(events.sum())} events  ({events.mean():.3%} of frames), "
              f"total damage {damage.sum():.1f}")
        if int(events.sum()) < 30:
            print("  TOO FEW EVENTS to measure. Any AUC here is noise.")
            continue

        print(f"\n  {'feature':<18}" + "".join(f"  k={k:<5d}" for k in args.horizons))
        print("  " + "-" * (18 + 8 * len(args.horizons)))
        for name in FEATURES:
            row = f"  {name:<18}"
            for k in args.horizons:
                labels = label_within_k(events, episode, k)
                row += f"  {auc(columns[name][exists], labels[exists]):.3f} "
            print(row)

        print(f"\n  restricted to in_strike_range frames ({int((in_range & exists).sum())} frames) "
              "-- removes the proximity confound")
        subset = in_range & exists
        print(f"  {'feature':<18}" + "".join(f"  k={k:<5d}" for k in args.horizons))
        print("  " + "-" * (18 + 8 * len(args.horizons)))
        for name in FEATURES:
            if name == "in_strike_range":
                continue
            row = f"  {name:<18}"
            for k in args.horizons:
                labels = label_within_k(events, episode, k)
                if int(labels[subset].sum()) < 10:
                    row += "     -   "
                    continue
                row += f"  {auc(columns[name][subset], labels[subset]):.3f} "
            print(row)

    print("\n" + "=" * 78)
    print("AUC 0.50 = chance. >0.60 = usable. The largest k that stays above ~0.60 is")
    print("roughly the furthest ahead this observation can see an attack coming, and")
    print("therefore a ceiling on a world model's useful imagination horizon.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
