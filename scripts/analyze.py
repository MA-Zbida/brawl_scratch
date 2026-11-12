"""Summarise the runs of an experiment from their progress files.

    python scripts/analyze.py s1_ppo
    python scripts/analyze.py s1_ppo --block 10      # rollouts per block (10 one-minute rollouts by default)

For every run under runs/<experiment>/<variant>/seed<k>/ (attempts that stopped early are listed
apart): the reward per minute of game time in blocks of rollouts with its standard error, the
change from the first block to the last, stocks, damage, and what the updates cost (stall time,
damage taken meanwhile). Writes results/<experiment>/summary.md.

A change is called real when it is beyond 2 standard errors of the difference between the two
blocks; the spread between one-minute rollouts is the run's own noise (gate N1 measured 0.58 per
5 minutes for a fixed policy).
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from brawl.paths import RESULTS, RUNS  # noqa: E402


def read(path: Path) -> dict[str, np.ndarray]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {key: np.array([float(r[key]) for r in rows]) for key in rows[0]} if rows else {}


def summarise(p: dict[str, np.ndarray], block: int) -> dict:
    minutes = p["game_frames"] / 3600
    rate = p["reward"] / minutes
    blocks = [rate[a:a + block] for a in range(0, len(rate) - block + 1, block)]
    mean = [float(b.mean()) for b in blocks]
    se = [float(b.std(ddof=1) / np.sqrt(len(b))) for b in blocks]
    change = se_change = float("nan")
    if len(blocks) >= 2:
        change, se_change = mean[-1] - mean[0], float(np.hypot(se[0], se[-1]))
    return {"rollouts": len(rate), "minutes": float(minutes.sum()), "mean": mean, "se": se, "change": change,
            "se_change": se_change, "stocks": (int(p["stocks_taken"].sum()), int(p["stocks_lost"].sum())),
            "damage": (float(p["damage_dealt"].sum()), float(p["damage_taken"].sum())),
            "stall_s": float(np.median(p["stall_s"][1:])) if len(rate) > 1 else 0.0,
            "stall_share": float(p["stall_s"].sum() / max(p["wall_s"][-1], 1e-9)),
            "stall_damage": float(p["stall_damage_taken"].sum()), "stall_stocks": int(p["stall_stocks_lost"].sum()),
            "over_budget": float(p["over_budget"].mean()), "pauses": int(p["pauses"].sum()),
            "net_stocks": [float((p["stocks_taken"][a:a + block] - p["stocks_lost"][a:a + block]).sum() / minutes[a:a + block].sum())
                           for a in range(0, len(rate) - block + 1, block)]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment")
    p.add_argument("--block", type=int, default=10, help="rollouts per block")
    args = p.parse_args(argv)
    root = RUNS / args.experiment
    files = sorted(root.glob("*/seed*/progress.csv"))
    if not files:
        raise SystemExit(f"no runs under {root}")
    lines = [f"# {args.experiment}: summary", "",
             f"{datetime.now():%Y-%m-%d %H:%M}. Reward per minute of game time in blocks of {args.block} rollouts (mean ± standard "
             "error over the block's rollouts). scripts/analyze.py.", ""]
    for path in files:
        run, progress = path.parent, read(path)
        if not progress or len(progress["reward"]) < args.block:
            lines += [f"## {run.parent.name} / {run.name}", "", f"Fewer than {args.block} rollouts: not summarised.", ""]
            continue
        s = summarise(progress, args.block)
        finished = (run / "policy.zip").exists()
        verdict = "no blocks to compare"
        if s["change"] == s["change"]:
            beyond = abs(s["change"]) > 2 * s["se_change"]
            verdict = (f"{s['change']:+.2f} ± {s['se_change']:.2f} from the first block to the last: "
                       + (("an improvement" if s["change"] > 0 else "a decline") + " beyond 2 standard errors" if beyond
                          else "within 2 standard errors, so not distinguishable from no change"))
        lines += [f"## {run.parent.name} / {run.name}" + ("" if finished else " (stopped before its end)"), "",
                  f"{s['rollouts']} rollouts, {s['minutes']:.0f} minutes of game time. Stocks {s['stocks'][0]} - {s['stocks'][1]}; "
                  f"damage {s['damage'][0]:.0f} / {s['damage'][1]:.0f}.", "",
                  "| Block (minutes) | Reward per minute | Net stocks per minute |", "|---|---|---|",
                  *(f"| {i * args.block}-{(i + 1) * args.block} | {m:+.2f} ± {e:.2f} | {n:+.2f} |"
                    for i, (m, e, n) in enumerate(zip(s["mean"], s["se"], s["net_stocks"]))), "",
                  f"- Change: {verdict}.",
                  f"- Updates: {s['stall_s']:.2f} s each (median), {s['stall_share']:.2%} of the run; during them "
                  f"{s['stall_damage']:.0f} damage taken and {s['stall_stocks']} stocks lost.",
                  f"- Decisions over the time budget: {s['over_budget']:.2%}; pauses for lost focus: {s['pauses']}.", ""]
    out = RESULTS / args.experiment / "summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
