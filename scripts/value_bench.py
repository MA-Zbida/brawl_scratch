"""Offline test of value learning, on the human recordings, before any live run.

    python scripts/value_bench.py                  # every variant
    python scripts/value_bench.py --only "a;b" --updates 10000

On recordings the true outcome of every moment is known: the discounted sum of the rewards that
followed (the Monte Carlo return). Each variant trains the recurrent Q-network (brawl/rdqn.py) on two
thirds of the human sessions and is scored on the held-out third, on the value it gives the action the
human took, Q(s, a_human):

- **correlation** with the true return (does it tell good moments from bad?);
- **bias**: mean Q minus mean true return (is it inflated?);
- **lose AUC**: does a low value come before the human loses a stock within 3 s? **win AUC**: does a
  high value come before a stock taken within 3 s? (0.5 is chance);
- **sessions**: correlation, across held-out sessions, of their mean Q with their mean true return
  (a losing Chosen session must be rated below a winning Medium one);
- **headroom**: mean of max Q minus Q(s, a_human): how much better than the human it believes it can do.

Variants: ``mc`` regresses the true return directly (no bootstrapping: what the observation can tell
at best); ``sarsa`` bootstraps on the action actually taken next (evaluates the human's play: Q should
match the true return); ``double_q`` is what the live learner does (values the best action, so some
headroom is expected; inflation and lost correlation are not). Writes results/rdqn/value_bench.md.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from brawl import actions, observation, rdqn  # noqa: E402
from brawl.config import resolve_config  # noqa: E402
from brawl.paths import CONFIGS, RESULTS  # noqa: E402
from brawl.record import Recording  # noqa: E402
from train import demo_sessions  # noqa: E402

AHEAD = 60  # decisions (3 s) for the lose and win AUCs
VARIANTS = {
    "mc regression, g 0.995": dict(mode="mc", gamma=0.995),
    "mc regression, g 0.99": dict(mode="mc", gamma=0.99),
    "sarsa, g 0.995, n 5": dict(mode="sarsa", gamma=0.995, n_step=5),
    "double_q, g 0.995, n 5, target 2000 (the night's)": dict(mode="double_q", gamma=0.995, n_step=5),
    "double_q, same, no margin loss": dict(mode="double_q", gamma=0.995, n_step=5, margin_weight=0.0),
    "double_q, g 0.99, n 5": dict(mode="double_q", gamma=0.99, n_step=5),
    "double_q, g 0.99, n 20": dict(mode="double_q", gamma=0.99, n_step=20),
    "double_q, g 0.995, n 20, target 500": dict(mode="double_q", gamma=0.995, n_step=20, target_every=500),
    "double_q, g 0.99, n 20, target 500, lr 3e-5": dict(mode="double_q", gamma=0.99, n_step=20, target_every=500,
                                                        learning_rate=3e-5),
}


def mc_return(reward: np.ndarray, gamma: float) -> np.ndarray:
    out, acc = np.zeros(len(reward), np.float32), 0.0
    for i in range(len(reward) - 1, -1, -1):
        acc = reward[i] + gamma * acc
        out[i] = acc
    return out


def auc(p: np.ndarray, y: np.ndarray) -> float:
    ranks = np.empty(len(p))
    ranks[np.argsort(p)] = np.arange(1, len(p) + 1)
    return float((ranks[y > 0].sum() - y.sum() * (y.sum() + 1) / 2) / max(y.sum() * (len(y) - y.sum()), 1))


class Session:
    def __init__(self, root: Path, k: int, weights: tuple[float, float]):
        z = np.load(root / f"decisions_k{k}.npz")
        self.name, self.level = root.name, Recording(root).meta["setting"].get("difficulty", "?")
        self.obs = z["obs"].astype(np.float32)
        self.action = (z["action"][:, 0] * actions.SHAPE[1] + z["action"][:, 1]).astype(np.int64)
        self.reward = (weights[0] * (z["dealt"] - z["taken"]) + weights[1] * (z["ko_scored"] - z["ko_lost"])).astype(np.float32)
        self.lose = np.array([z["ko_lost"][i:i + AHEAD].sum() > 0 for i in range(len(self.obs))])
        self.win = np.array([z["ko_scored"][i:i + AHEAD].sum() > 0 for i in range(len(self.obs))])
        from brawl.learn import stretches

        self.margin = np.zeros(len(self.obs), bool)
        for a, b, outcome in stretches(z["ko_scored"], z["ko_lost"]):
            self.margin[a:b] = outcome == "won"


def replay_of(sessions: list[Session], gamma: float) -> tuple[rdqn.Replay, np.ndarray]:
    replay = rdqn.Replay(observation.SIZE, sum(len(s.obs) for s in sessions))
    rets = []
    for s in sessions:
        cut = np.zeros(len(s.obs), bool)
        cut[0] = True
        replay.add(s.obs, s.action, s.reward, cut, s.margin)
        rets.append(mc_return(s.reward, gamma))
    return replay, np.concatenate(rets)


def train(variant: dict, base: dict, sessions: list[Session], updates: int, device: str, log) -> object:
    import torch
    import torch.nn.functional as fn

    cfg = {**base, **{k: v for k, v in variant.items() if k not in ("mode", "gamma")}, "bootstrap": variant["mode"]}
    gamma = float(variant["gamma"])
    learner = rdqn.Learner(observation.SIZE, cfg, gamma, device, seed=0)
    replay, ret = replay_of(sessions, gamma)
    if variant["mode"] != "mc":
        t0 = time.perf_counter()
        for i in range(1, updates + 1):
            learner.step([(replay, int(cfg["batch"]), True)], float(cfg["margin_weight"]))
            if i % 5000 == 0:
                log(f"    {i}/{updates} ({(time.perf_counter() - t0) / i * 1000:.0f} ms per update)")
        return learner.net
    net, opt, rng = learner.net, learner.opt, np.random.default_rng(0)  # direct regression on the true return
    b, t = int(cfg["burn_in"]), int(cfg["train_length"])
    for i in range(1, updates + 1):
        idx = replay.sample(int(cfg["batch"]), b + t, rng)
        obs = torch.as_tensor(replay.obs[idx].astype(np.float32), device=device)
        act = torch.as_tensor(replay.action[idx], device=device)
        target = torch.as_tensor(ret[idx], device=device)
        q, _ = net(obs)
        q_a = q[:, b:].gather(-1, act[:, b:, None]).squeeze(-1)
        loss = fn.smooth_l1_loss(q_a, target[:, b:])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()
        if i % 5000 == 0:
            log(f"    {i}/{updates}")
    return net


def evaluate(net, sessions: list[Session], gamma: float, device: str) -> dict:
    import torch

    net.eval()
    qa, best, ret, lose, win, per = [], [], [], [], [], []
    for s in sessions:
        out, h = [], None
        with torch.no_grad():
            for a in range(0, len(s.obs), 2048):
                q, h = net(torch.as_tensor(s.obs[a:a + 2048], device=device)[None], h)
                out.append(q[0].float().cpu().numpy())
        q = np.concatenate(out)
        keep = slice(0, len(s.obs) - int(3 / (1 - gamma)))  # the true return is cut short at the end of a session
        r = mc_return(s.reward, gamma)[keep]
        v = np.take_along_axis(q, s.action[:, None], 1)[:, 0][keep]
        qa.append(v), best.append(q.max(1)[keep]), ret.append(r), lose.append(s.lose[keep]), win.append(s.win[keep])
        per.append((s.name, s.level, float(v.mean()), float(r.mean())))
    qa, best, ret, lose, win = (np.concatenate(x) for x in (qa, best, ret, lose, win))
    means = np.array([(m, t) for *_, m, t in per])
    return {"correlation": float(np.corrcoef(qa, ret)[0, 1]), "bias": float(qa.mean() - ret.mean()),
            "lose_auc": auc(-qa, lose), "win_auc": auc(qa, win), "headroom": float((best - qa).mean()),
            "sessions": float(np.corrcoef(means[:, 0], means[:, 1])[0, 1]), "per": per}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", default="", help="variant names separated by ;")
    p.add_argument("--updates", type=int, default=20000)
    args = p.parse_args(argv)

    import torch

    base_cfg = resolve_config(CONFIGS / "base.yaml")
    base, env = base_cfg["train"]["rdqn"], base_cfg["env"]
    k, weights = int(env["k"]), (float(env["damage_weight"]), float(env["ko_weight"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    roots = demo_sessions(base["demo_levels"], k)
    sessions = [Session(r, k, weights) for r in roots]
    test = sessions[::3]
    train_set = [s for s in sessions if s not in test]
    log = lambda m: print(m, flush=True)  # noqa: E731
    log(f"{len(train_set)} sessions to train on, {len(test)} held out: "
        + ", ".join(f"{s.name} ({s.level})" for s in test))
    names = [n for n in VARIANTS if not args.only or n in args.only.split(";")]
    lines = ["# Offline test of value learning", "",
             f"{datetime.now():%Y-%m-%d %H:%M}. scripts/value_bench.py. Trained on {len(train_set)} human sessions, "
             f"{args.updates} updates each; scored on {len(test)} held-out sessions on Q(s, a_human) against the true "
             f"discounted return. Lose / win AUC: the value before a stock lost / taken within {AHEAD} decisions.", "",
             "| Variant | Correlation | Bias | Lose AUC | Win AUC | Sessions ranked | Headroom |",
             "|---|---|---|---|---|---|---|"]
    detail = []
    for name in names:
        log(f"{name}")
        net = train(VARIANTS[name], base, train_set, args.updates, device, log)
        r = evaluate(net, test, float(VARIANTS[name]["gamma"]), device)
        lines.append(f"| {name} | {r['correlation']:.3f} | {r['bias']:+.2f} | {r['lose_auc']:.3f} | {r['win_auc']:.3f} | "
                     f"{r['sessions']:.2f} | {r['headroom']:.2f} |")
        detail.append(f"- {name}: " + "; ".join(f"{n} ({lv}) Q {q:+.2f} vs true {t:+.2f}" for n, lv, q, t in r["per"]))
        log(lines[-1])
    path = RESULTS / "rdqn" / "value_bench.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines + ["", "Per held-out session (mean Q of the human's action against the mean true return):", ""]
                              + detail) + "\n", encoding="utf-8")
    print("\n".join(lines), f"\n-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
