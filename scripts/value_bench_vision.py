"""Does the advisor value outcomes better with pictures of the fighters? An offline test on human play.

    python scripts/value_bench_vision.py
    python scripts/value_bench_vision.py --updates 3000 --only "vector"      # a quick look

The advisor of the live runs (configs/experiments/rdqn_medium_v3.yaml: a recurrent Q-network evaluating
the play it sees, sarsa, gamma 0.99, 20-step returns, target every 500 updates) is trained on the human
Medium sessions that have fighter crops (scripts/prepare_crops.py), with and without the crops, and
scored on held-out sessions as scripts/value_bench.py scores it: correlation of Q(s, a_human) with the
true discounted return, bias, and how well a low value comes before a stock lost within 3 s (lose AUC)
and a high value before a stock taken (win AUC). The lose AUC is the one vision should move: seeing an
attack start is seeing a lost stock coming.

The picture network (brawl/rdqn.py build_vision): a CNN shared by both fighters' 96 px crops, its two
outputs concatenated with the vector's, into the same GRU and dueling head. Crops are read from disk
per sequence (the laptop's GPU cannot hold them all next to the network). Writes
results/rdqn/value_bench_vision.md.
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
from brawl.config import load_experiment  # noqa: E402
from brawl.paths import CONFIGS, DATA, REPO_ROOT, RESULTS  # noqa: E402
from brawl.record import Recording, is_demo  # noqa: E402
from value_bench import AHEAD, auc, mc_return  # noqa: E402

VARIANTS = {"vector": dict(crops=False, learning_rate=3e-5),
            "vector + crops": dict(crops=True, learning_rate=3e-5),
            "vector + crops, lr 1e-4": dict(crops=True, learning_rate=1e-4)}


class Session:
    def __init__(self, name, obs, action, reward, ko_scored, ko_lost, crops):
        self.name, self.obs, self.action, self.reward, self.crops = name, obs, action, reward, crops  # crops [n, 2, 96, 96]
        self.n = len(obs)
        self.lose = np.array([ko_lost[i:i + AHEAD].sum() > 0 for i in range(self.n)])
        self.win = np.array([ko_scored[i:i + AHEAD].sum() > 0 for i in range(self.n)])

    @classmethod
    def demo(cls, root: Path, k: int, weights):
        z = np.load(root / f"decisions_k{k}.npz")
        reward = (weights[0] * (z["dealt"] - z["taken"]) + weights[1] * (z["ko_scored"] - z["ko_lost"])).astype(np.float32)
        return cls(root.name, z["obs"].astype(np.float32), (z["action"][:, 0] * actions.SHAPE[1] + z["action"][:, 1]).astype(np.int64),
                   reward, z["ko_scored"], z["ko_lost"], np.load(root / f"crops_k{k}_96.npy", mmap_mode="r"))


def night(run: Path, segments: int, log) -> list[Session]:
    """A live run's trace (train.rdqn.trace_crops) cut into equal segments of time. The crops are unpacked
    once into DATA/traces/<experiment>/crops.npy (13 GB for 10 hours: the laptop's RAM cannot hold them)."""
    cache = DATA / "traces" / run.parts[-3]
    cache.mkdir(parents=True, exist_ok=True)
    shards = sorted((run / "trace").glob("shard_*.npz"))
    if not (cache / "done").exists():
        n = sum(len(np.load(f)["step"]) for f in shards)
        crops = np.lib.format.open_memmap(cache / "crops.npy", mode="w+", dtype=np.uint8, shape=(n, 2, 96, 96))
        cols: dict[str, list] = {"obs": [], "action": [], "reward": [], "events": []}
        a = 0
        for i, f in enumerate(shards):
            z = np.load(f)
            m = len(z["step"])
            crops[a:a + m] = z["crops"]
            for key in cols:
                cols[key].append(z[key])
            a += m
            if i % 20 == 0:
                log(f"  unpacking {i + 1}/{len(shards)} shards")
        crops.flush()
        del crops
        np.savez(cache / "decisions.npz", **{key: np.concatenate(v) for key, v in cols.items()})
        (cache / "done").write_text("ok")
    z = np.load(cache / "decisions.npz")
    crops = np.load(cache / "crops.npy", mmap_mode="r")
    obs, act, rew, ev = z["obs"].astype(np.float32), z["action"].astype(np.int64), z["reward"].astype(np.float32), z["events"]
    edges = np.linspace(0, len(obs), segments + 1).astype(int)
    return [Session(f"{run.parts[-3]}[{a}:{b}]", obs[a:b], act[a:b], rew[a:b], ev[a:b, 2], ev[a:b, 3], crops[a:b])
            for a, b in zip(edges[:-1], edges[1:])]


def shift(img, torch):
    """One random shift of up to 4 px per sequence, the same for every picture in it."""
    import torch.nn.functional as fn

    b, t = img.shape[:2]
    x = img.reshape(b * t, *img.shape[2:])
    theta = torch.zeros(b, 2, 3, device=img.device)
    theta[:, 0, 0] = theta[:, 1, 1] = 1
    theta[:, :, 2] = (torch.rand(b, 2, device=img.device) * 2 - 1) * (8 / 96)
    grid = fn.affine_grid(theta.repeat_interleave(t, 0), x.shape, align_corners=False)
    return fn.grid_sample(x, grid, padding_mode="zeros", align_corners=False).reshape(img.shape)


def train(variant: dict, cfg: dict, gamma: float, sessions: list[Session], updates: int, device: str, log):
    import copy

    import torch
    import torch.nn.functional as fn

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    vision = variant["crops"]
    net = (rdqn.build_vision if vision else rdqn.build)(observation.SIZE, int(cfg["hidden"]), device, rdqn.HISTORY)
    target = copy.deepcopy(net).eval()
    opt = torch.optim.AdamW(net.parameters(), lr=float(variant["learning_rate"]), weight_decay=1e-5)
    b, t, n = int(cfg["burn_in"]), int(cfg["train_length"]), int(cfg["n_step"])
    length, batch = b + t + n, int(cfg["batch"])
    weights = np.array([s.n - length for s in sessions], float)
    obs = [torch.as_tensor(s.obs, device=device) for s in sessions]
    act = [torch.as_tensor(s.action, device=device) for s in sessions]
    rew = [torch.as_tensor(s.reward, device=device) for s in sessions]
    t0 = time.perf_counter()
    for u in range(1, updates + 1):
        picks = rng.choice(len(sessions), size=batch, p=weights / weights.sum())
        starts = [int(rng.integers(0, sessions[i].n - length)) for i in picks]
        x = torch.stack([obs[i][a:a + length] for i, a in zip(picks, starts)])
        a_ = torch.stack([act[i][a:a + length] for i, a in zip(picks, starts)])
        r_ = torch.stack([rew[i][a:a + length] for i, a in zip(picks, starts)])
        img = None
        if vision:
            img = torch.as_tensor(np.stack([np.asarray(sessions[i].crops[a:a + length]) for i, a in zip(picks, starts)]),
                                  device=device).float().div_(255)
            img = shift(img, torch)
        q, _ = net(x, img) if vision else net(x)
        with torch.no_grad():
            qt, _ = target(x, img) if vision else target(x)
            steps = torch.arange(b, b + t, device=device)
            ret = sum(gamma ** j * r_[:, steps + j] for j in range(n))
            y = ret + gamma ** n * qt[:, steps + n].gather(-1, a_[:, steps + n, None]).squeeze(-1)
        q_a = q[:, steps].gather(-1, a_[:, steps, None]).squeeze(-1)
        loss = fn.smooth_l1_loss(q_a, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()
        if u % int(cfg["target_every"]) == 0:
            target.load_state_dict(net.state_dict())
        if u % 2000 == 0 or u == 50:
            log(f"    {u}/{updates}: loss {float(loss):.5f} ({(time.perf_counter() - t0) / u * 1000:.0f} ms per update)")
    return net


def evaluate(net, vision: bool, sessions: list[Session], gamma: float, device: str) -> dict:
    import torch

    net.eval()
    qa, ret, lose, win = [], [], [], []
    for s in sessions:
        out, h = [], None
        with torch.no_grad():
            for a in range(0, s.n, 512):
                x = torch.as_tensor(s.obs[a:a + 512], device=device)[None]
                if vision:
                    img = torch.as_tensor(np.asarray(s.crops[a:a + 512]), device=device).float().div_(255)[None]
                    q, h = net(x, img, h)
                else:
                    q, h = net(x, h)
                out.append(q[0].float().cpu().numpy())
        q = np.concatenate(out)
        keep = slice(0, s.n - int(3 / (1 - gamma)))
        qa.append(np.take_along_axis(q, s.action[:, None], 1)[:, 0][keep])
        ret.append(mc_return(s.reward, gamma)[keep])
        lose.append(s.lose[keep])
        win.append(s.win[keep])
    qa, ret, lose, win = (np.concatenate(v) for v in (qa, ret, lose, win))
    return {"correlation": float(np.corrcoef(qa, ret)[0, 1]), "bias": float(qa.mean() - ret.mean()),
            "lose_auc": auc(-qa, lose), "win_auc": auc(qa, win)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", default="", help="variant names separated by ;")
    p.add_argument("--updates", type=int, default=20000)
    p.add_argument("--folds", type=int, default=3, help="sessions held out in turn (at least 2)")
    p.add_argument("--trace", type=Path, default=None, help="a live run (runs/<exp>/<variant>/seed<k>) whose trace has crops, "
                   "instead of the human sessions")
    p.add_argument("--segments", type=int, default=8, help="with --trace: the run cut into this many segments of time; "
                   "each fold holds out two, half a run apart")
    args = p.parse_args(argv)

    import torch

    config = load_experiment(CONFIGS / "experiments" / "rdqn_medium_v3.yaml").resolve("default", 0)
    cfg, env = config["train"]["rdqn"], config["env"]
    gamma, k = float(config["train"]["gamma"]), int(env["k"])
    weights = (float(env["damage_weight"]), float(env["ko_weight"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log = lambda m: print(m, flush=True)  # noqa: E731
    if args.trace:
        run = args.trace if args.trace.is_absolute() else REPO_ROOT / args.trace
        sessions = night(run, args.segments, log)
        half = args.segments // 2
        step = max(1, half // args.folds)
        held = [{f * step + 1, f * step + 1 + half} for f in range(args.folds)]
        source = f"the agent's own play in {'/'.join(run.parts[-4:])} ({args.segments} segments of time; each fold holds out two, half the run apart)"
        out = RESULTS / "rdqn" / f"value_bench_vision_{run.parts[-3]}.md"
    else:
        roots = [q for q in sorted((DATA / "demos").iterdir()) if (q / f"crops_k{k}_96.npy").exists()
                 and (q / f"decisions_k{k}.npz").exists() and is_demo(Recording(q).meta)
                 and Recording(q).meta["setting"].get("difficulty") == "medium"]
        sessions = [Session.demo(r, k, weights) for r in roots]
        folds = max(2, args.folds)
        held = [{i for i in range(len(sessions)) if i % folds == f} for f in range(folds)]
        source = f"{len(sessions)} human Medium sessions with crops ({folds} folds, a share of the sessions held out in turn)"
        out = RESULTS / "rdqn" / "value_bench_vision.md"
    log(f"{len(sessions)} sessions with crops, {sum(s.n for s in sessions)} decisions; {len(held)} folds")
    names = [n for n in VARIANTS if not args.only or n in args.only.split(";")]
    results: dict[str, list[dict]] = {}
    for fold, ids in enumerate(held):
        test = [s for i, s in enumerate(sessions) if i in ids]
        train_set = [s for s in sessions if s not in test]
        for name in names:
            log(f"{name}, fold {fold} (held out: {', '.join(s.name for s in test)})")
            v = VARIANTS[name]
            net = train(v, cfg, gamma, train_set, args.updates, device, log)
            r = evaluate(net, v["crops"], test, gamma, device)
            results.setdefault(name, []).append(r)
            log(f"  -> correlation {r['correlation']:.3f}, bias {r['bias']:+.2f}, lose AUC {r['lose_auc']:.3f}, "
                f"win AUC {r['win_auc']:.3f}")
            del net
            torch.cuda.empty_cache()
    base = results.get("vector")
    lines = ["# Does the advisor value outcomes better with pictures?", "",
             f"{datetime.now():%Y-%m-%d %H:%M}. scripts/value_bench_vision.py. The advisor of rdqn_medium_v3 (sarsa, gamma {gamma}, "
             f"n {cfg['n_step']}, target every {cfg['target_every']}), {args.updates} updates, on {source}, "
             f"{sum(s.n for s in sessions)} decisions. Mean over folds; in brackets the mean "
             f"difference from `vector` in the same fold, and its spread.", "",
             "| Input | Correlation | Bias | Lose AUC | Win AUC |", "|---|---|---|---|---|"]
    for name, rs in results.items():
        cells = []
        for key, d in (("correlation", 3), ("bias", 2), ("lose_auc", 3), ("win_auc", 3)):
            v = np.array([r[key] for r in rs])
            cell = f"{v.mean():.{d}f}"
            if base and name != "vector":
                diff = v - np.array([r[key] for r in base])
                cell += f" ({diff.mean():+.{d}f} ± {diff.std():.{d}f})"
            cells.append(cell)
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    path = out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), f"\n-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
