"""Draw the figures of the results so far, from the run files, into results/figures/.

    python scripts/plot_results.py

Every figure is rebuilt from its sources each time, so a new run only needs the script run again:

- rl_runs.png: reward per minute of game time through each live RL run, against the cloned policy;
- rl_stocks.png: stocks lost and taken per minute through the same runs (falls are what went wrong);
- value_estimates.png: what the Q-network believes moments are worth against what followed;
- probes_inputs.png: what each candidate input adds to the observation vector (held-out probes);
- value_bench.png: the offline test of value-learning variants (when results/rdqn/value_bench.md exists).

Colours: the dataviz reference palette (slots 1-3, validated light; aqua is below 3:1 on the surface,
so every series is also labelled directly).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from brawl.paths import RESULTS, RUNS  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
OUT = RESULTS / "figures"


def style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                         "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
                         "legend.frameon": False, "lines.linewidth": 2})
    return plt


def progress(path: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def blocks(values: np.ndarray, minutes: np.ndarray, size: int = 10):
    """Per-minute rate in blocks of ``size`` rollouts: (block middle in minutes, mean, standard error)."""
    rate = values / minutes
    n = len(rate) // size
    r = rate[:n * size].reshape(n, size)
    t = np.cumsum(minutes)[:n * size].reshape(n, size)
    return t.mean(1), r.mean(1), r.std(1, ddof=1) / np.sqrt(size)


def clone_floor() -> tuple[float, float]:
    z = json.load(open(RUNS / "play" / "medium_v4_medium_realtime_x6.json", encoding="utf-8"))
    r = np.array([b["reward"] / (b["frames"] / 3600) for b in z["blocks"]])
    return float(r.mean()), float(r.std(ddof=1) / np.sqrt(len(r)))


RUNS_SHOWN = [("PPO from the clone (1 h)", RUNS / "s1_ppo" / "realtime-bc" / "seed0" / "progress.csv", ORANGE),
              ("Recurrent Q-learning, first settings (7 h)", RUNS / "rdqn_medium" / "default" / "seed0" / "progress.csv", BLUE),
              ("The clone, advised by a Q-network (7 h)", RUNS / "rdqn_medium_v3" / "default" / "seed0" / "progress.csv", AQUA),
              ("Advised, second night (6 h)", RUNS / "advised_medium_night2" / "default" / "seed0" / "progress.csv", "#eda100")]


def rl_runs(plt) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.6))
    mean, se = clone_floor()
    ax.axhspan(mean - 1.96 * se, mean + 1.96 * se, color=GRAY, alpha=0.18, lw=0)
    ax.axhline(mean, color=GRAY, lw=1.5, ls="--")
    ax.text(5, mean + 1.96 * se + 0.08, "cloned policy, no RL (30 min, 95% band)", color=INK2, fontsize=9)
    ax.axhline(0, color=INK2, lw=0.8)
    for label, path, colour in RUNS_SHOWN:
        if not path.exists():
            continue
        p = progress(path)
        t, m, e = blocks(p["reward"], p["game_frames"] / 3600)
        ax.fill_between(t, m - 1.96 * e, m + 1.96 * e, color=colour, alpha=0.15, lw=0)
        ax.plot(t, m, color=colour, marker="o", ms=4, label=label)  # named in the legend: end labels collide
    ax.set_xlabel("minutes of game time")
    ax.set_ylabel("reward per minute (10-minute blocks)")
    ax.set_title("Advising the clone with learned values beats it; replacing the clone did not")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_xlim(0, None)
    fig.tight_layout()
    path = OUT / "rl_runs.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def rl_stocks(plt) -> Path:
    shown = [(label, path, colour) for label, path, colour in RUNS_SHOWN if path.exists()]
    fig, axes = plt.subplots(1, len(shown), figsize=(5 * len(shown), 4), sharey=True, squeeze=False)
    for ax, (label, path, _) in zip(axes[0], shown):
        p = progress(path)
        minutes = p["game_frames"] / 3600
        for key, colour, name in (("stocks_lost", ORANGE, "stocks lost"), ("stocks_taken", BLUE, "stocks taken")):
            t, m, e = blocks(p[key], minutes)
            ax.fill_between(t, m - 1.96 * e, m + 1.96 * e, color=colour, alpha=0.15, lw=0)
            ax.plot(t, m, color=colour, marker="o", ms=4, label=name)
            ax.annotate(name, (t[-1], m[-1]), xytext=(6, 0), textcoords="offset points", color=INK, fontsize=9, va="center")
        ax.axhline(16 / 30, color=GRAY, lw=1.2, ls="--")
        ax.axhline(17 / 30, color=GRAY, lw=1.2, ls="--")
        ax.text(2, 0.7, "the clone: about 0.55 a minute each way", color=INK2, fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("minutes of game time")
        ax.set_xlim(0, None)
    axes[0][0].set_ylabel("per minute (10-minute blocks)")
    axes[0][0].legend(loc="upper left", fontsize=9)
    fig.suptitle("The Q-learner lost stocks faster and faster: mostly falls (damage per stock lost fell from 73 to 19)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    path = OUT / "rl_stocks.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def value_estimates(plt) -> Path | None:
    """Mean Q of the human's action on held-out human sessions, and the true discounted return there."""
    import torch

    from brawl import rdqn
    from value_bench import mc_return
    from train import demo_sessions

    nets = [("network at the start", RUNS / "rdqn" / "pretrain_all" / "qnet.pt", BLUE),
            ("after the night", RUNS / "rdqn_medium" / "default" / "seed0" / "qnet.pt", ORANGE)]
    if not all(p.exists() for _, p, _ in nets):
        return None
    cases = [("your play vs Medium (you win)", demo_sessions(["medium"], 3)[:2]),
             ("your play vs Hard (you lose)", demo_sessions(["hard"], 3)[:2]),
             ("your play vs Chosen (you lose)", demo_sessions(["chosen"], 3)[:1])]
    truth, est = [], {name: [] for name, _, _ in nets}
    for _, roots in cases:
        t, q = [], {name: [] for name, _, _ in nets}
        for root in roots:
            z = np.load(root / "decisions_k3.npz")
            r = 0.01 * (z["dealt"] - z["taken"]) + (z["ko_scored"] - z["ko_lost"])
            a = z["action"][:, 0] * 6 + z["action"][:, 1]
            t.append(mc_return(r, 0.995)[:-600])
            for name, path, _ in nets:
                net, out, h = rdqn.load(path)[0].eval(), [], None
                with torch.no_grad():
                    for s in range(0, len(r), 2048):
                        v, h = net(torch.as_tensor(z["obs"][s:s + 2048], dtype=torch.float32)[None], h)
                        out.append(v[0].numpy())
                q[name].append(np.take_along_axis(np.concatenate(out), a[:, None], 1)[:-600, 0])
        truth.append(np.concatenate(t).mean())
        for name in q:
            est[name].append(np.concatenate(q[name]).mean())
    fig, ax = plt.subplots(figsize=(9, 4.2))
    y = np.arange(len(cases))[::-1]
    for i in range(len(cases)):
        xs = [truth[i]] + [est[name][i] for name, _, _ in nets]
        ax.plot([min(xs), max(xs)], [y[i]] * 2, color=GRID, lw=2, zorder=1)
    ax.scatter(truth, y, s=80, color=INK, zorder=3, label="what actually followed")
    for name, _, colour in nets:
        ax.scatter(est[name], y, s=80, color=colour, edgecolor=SURFACE, linewidth=2, zorder=3, label=f"network's value, {name}")
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, [c for c, _ in cases])
    ax.set_xlabel("value of a moment (discounted reward, gamma 0.995)")
    ax.set_title("It rated losing play like winning play, then inflated overnight")
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=3, fontsize=8.5)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    path = OUT / "value_estimates.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def probes(plt) -> Path | None:
    done: dict = {}
    for path in sorted((RUNS / "probes" / "vision").glob("folds*.json")):
        done.update(json.loads(path.read_text()))
    if not done:
        return None
    names = {"vector, no timers": "no timers", "vector, GRU": "+ memory (GRU)", "both, 1 frame": "+ crops of both fighters",
             "both, 4 decisions (0.15 s)": "+ crops, 4 decisions", "both, 1 frame, GRU": "+ crops + memory",
             "vector + reader states": "+ readers' 5 states", "vector + reader states, GRU": "+ readers' states + memory",
             "vector + reader memory": "+ readers' hidden state"}
    measures = [("act_nll_changed", "imitating your changes of action\n(change in loss; lower is better)", -1),
                ("hit_auc", "seeing a hit coming within 0.5 s\n(change in AUC; higher is better)", 1),
                ("move_rmse_10", "Mordex's position 0.5 s ahead\n(change in error, px; lower is better)", -1)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
    labels = list(names.values())
    y = np.arange(len(labels))[::-1]
    for ax, (key, title, better) in zip(axes, measures):
        for i, (name, label) in enumerate(names.items()):
            runs = [k.split(" | ", 1)[1] for k in done if k.startswith(name + " | ")]
            d = np.array([done[f"{name} | {r}"][key] - done[f"vector | {r}"][key] for r in runs if f"vector | {r}" in done])
            if not len(d):
                continue
            good = better * d.mean() > 2 * d.std() / max(np.sqrt(len(d)), 1) and better * d.mean() > 0
            colour = BLUE if good else GRAY
            ax.errorbar(d.mean(), y[i], xerr=d.std(), fmt="o", color=colour, ms=7, elinewidth=2, capsize=0)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(title, fontsize=9.5, fontweight="normal")
        ax.grid(axis="y", visible=False)
        from matplotlib.ticker import MaxNLocator

        ax.xaxis.set_major_locator(MaxNLocator(5))
    axes[0].set_yticks(y, labels)
    fig.suptitle("What each input adds to the vector with timers (held-out sessions; dot = mean, bar = spread over 6 runs; "
                 "blue = a clear gain)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    path = OUT / "probes_inputs.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def value_bench(plt) -> Path | None:
    source = RESULTS / "rdqn" / "value_bench.md"
    if not source.exists():
        return None
    rows = [line.split("|")[1:-1] for line in source.read_text(encoding="utf-8").splitlines()
            if line.startswith("| ") and not line.startswith("| Variant") and not line.startswith("|---")]
    if not rows:
        return None
    names = [r[0].strip() for r in rows]
    corr = [float(r[1]) for r in rows]
    bias = [float(r[2]) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.45 * len(rows) + 1.6), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, values, title in ((axes[0], corr, "correlation with what actually followed (higher is better)"),
                              (axes[1], bias, "bias: value minus what followed (0 is right)")):
        colours = [GRAY if n.startswith("mc") else BLUE for n in names]
        ax.barh(y, values, color=colours, height=0.6)
        for yi, v in zip(y, values):
            ax.text(v, yi, f" {v:+.2f}" if ax is axes[1] else f" {v:.2f}", va="center", color=INK, fontsize=8.5,
                    ha="left" if v >= 0 else "right")
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(title, fontsize=9.5, fontweight="normal")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, names, fontsize=8.5)
    fig.suptitle("Offline test of value learning (gray: regressing the true return directly, the best the inputs allow)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    path = OUT / "value_bench.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> int:
    plt = style()
    OUT.mkdir(parents=True, exist_ok=True)
    for make in (rl_runs, rl_stocks, value_estimates, probes, value_bench):
        path = make(plt)
        print(f"{make.__name__}: {path if path else 'no data yet'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
