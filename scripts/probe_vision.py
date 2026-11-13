"""Gates S1 and S2: what do memory and a picture of the fighters add to the observation vector?

    python scripts/prepare_crops.py --difficulty medium     # first
    python scripts/probe_vision.py                          # every candidate, 3 folds, 2 seeds
    python scripts/probe_vision.py --only "vector;both, 1 frame" --seeds 1
    python scripts/probe_vision.py --report                 # the table again, from the runs so far

Each candidate input (CANDIDATES) trains the same network on recorded play to predict, at every
decision:

1. **Defence:** does Wu Shang take damage within the next 10 decisions (0.5 s)?
2. **Anticipation:** where Mordex is 5 and 10 decisions later (0.25 and 0.5 s), as a correction
   to the guess that he keeps his current velocity. Only where he is seen at both times and no
   stock is lost in between.
3. **Imitation:** the human's action, scored apart where the human changes action.

All three targets are free: damage from the HUD, positions from the tracker, actions from the keys.
Sessions are held out whole (three folds). The network: an MLP on the vector; for crops a small
CNN shared by both fighters (random shifts while training, one shift per window); a GRU over the
session for the recurrent candidates; four heads. Trained on windows of 128 decisions until the
last 12% of the training sessions stops improving, so every candidate is compared at its best.

Writes results/probes/vision.md and runs/probes/vision/folds*.json (every run as it finishes; a
stopped run resumes). Runs go seed by seed and fold by fold, so that if ``--hours`` runs out every
candidate has the same runs; a run the clock interrupts keeps its best epoch and is marked.
``--part 0/2`` and ``--part 1/2`` split the runs evenly over two processes (one per GPU). A candidate earns its place when it beats ``vector`` in the same fold and
seed by more than the spread of that difference.

On Kaggle (two GPUs): scripts/pack_probes.py makes the pack and prints the notebook cell.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from brawl import observation  # noqa: E402
from brawl.paths import DATA, RESULTS, RUNS  # noqa: E402
from brawl.record import Recording, is_demo  # noqa: E402

CANDIDATES = {
    "vector, no timers": dict(fighters=(), stack=1, timers=False, recurrent=False),
    "vector": dict(fighters=(), stack=1, timers=True, recurrent=False),
    "vector, GRU": dict(fighters=(), stack=1, timers=True, recurrent=True),
    "both, 1 frame": dict(fighters=(0, 1), stack=1, timers=True, recurrent=False),
    "both, 4 decisions (0.15 s)": dict(fighters=(0, 1), stack=4, timers=True, recurrent=False),
    "both, 1 frame, GRU": dict(fighters=(0, 1), stack=1, timers=True, recurrent=True),
    # the recurrent readers' view of both fighters (scripts/train_reader.py), added to the vector: their five
    # outputs each (attacking, dodging, jumping, since the last dodge, jumps), or their hidden states
    "vector + reader states": dict(fighters=(), stack=1, timers=True, recurrent=False, extra="states"),
    "vector + reader states, GRU": dict(fighters=(), stack=1, timers=True, recurrent=True, extra="states"),
    "vector + reader memory": dict(fighters=(), stack=1, timers=True, recurrent=False, extra="hidden"),
    "vector + reader memory, GRU": dict(fighters=(), stack=1, timers=True, recurrent=True, extra="hidden"),
}
EXTRA = {None: 0, "states": 10, "hidden": 256}


def vector_size(c: dict) -> int:
    return observation.SIZE + EXTRA[c.get("extra")]
HIT_AHEAD, AHEAD = 10, (5, 10)
LENGTH, PER_BATCH, PATIENCE, TAIL = 128, 8, 12, 0.12  # window (decisions), windows per batch, epochs, stop share
F = observation.FEATURES


class Session:
    def __init__(self, root: Path, k: int, size: int):
        z = np.load(root / f"decisions_k{k}.npz")
        if tuple(z["features"]) != F:
            raise SystemExit(f"{root.name}: prepared with another feature list (scripts/prepare_demos.py)")
        self.name = root.name
        self.obs = z["obs"].astype(np.float32)
        self.action = z["action"].astype(np.int64)
        self.crops = np.load(root / f"crops_k{k}_{size}.npy", mmap_mode="r")  # [n, 2, S, S]
        if self.crops.ndim != 4:
            raise SystemExit(f"{root.name}: crops in the old format; cut them again (scripts/prepare_crops.py)")
        n = len(self.obs)
        self.previous = np.vstack([np.zeros((1, 2), np.int64), self.action[:-1]])
        taken = z["taken"]
        self.hit = np.array([taken[i:i + HIT_AHEAD].sum() > 0 for i in range(n)], np.float32)
        self.hit_ok = np.arange(n) < n - HIT_AHEAD
        ko = (z["ko_scored"] + z["ko_lost"]) > 0
        x = self.obs[:, F.index("opponent_x")] * observation.X
        y = self.obs[:, F.index("opponent_y")] * observation.Y
        vx = self.obs[:, F.index("opponent_vx")] * observation.V
        vy = self.obs[:, F.index("opponent_vy")] * observation.V
        present, seen = self.obs[:, F.index("opponent_present")] > 0, self.obs[:, F.index("opponent_seen")] > 0
        guess = np.zeros((n, 4), np.float32)  # where he went, minus where constant velocity said (px)
        self.move_ok = np.zeros(n, bool)
        ko_before = np.concatenate([[0], np.cumsum(ko)])
        for i in range(n - max(AHEAD)):
            ok = present[i]
            for j, h in enumerate(AHEAD):
                ok &= seen[i + h] and ko_before[i + h] == ko_before[i]
                guess[i, 2 * j:2 * j + 2] = (x[i + h] - x[i] - vx[i] * h, y[i + h] - y[i] - vy[i] * h)
            self.move_ok[i] = ok
        self.move = guess / 100.0
        self.gpu = None  # the crops on the GPU, when they fit
        path = root / f"reader_k{k}.npz"
        self.reader = dict(np.load(path)) if path.exists() else None

    def images(self, idx: np.ndarray, c: dict, device: str):
        """uint8 tensor [N, fighters x stack, S, S]: for each fighter, its last ``stack`` decisions, newest first."""
        import torch

        back = np.maximum(idx[:, None] - np.arange(c["stack"]), 0)
        if self.gpu is not None:
            a = self.gpu[torch.as_tensor(back, device=self.gpu.device)]
        else:
            a = torch.as_tensor(np.ascontiguousarray(self.crops[back.reshape(-1)]), device=device)
            a = a.reshape(len(idx), c["stack"], *self.crops.shape[1:])
        a = a[:, :, list(c["fighters"])].permute(0, 2, 1, 3, 4)
        return a.reshape(len(idx), -1, *a.shape[-2:])


def build(c: dict, device: str):
    import torch
    from torch import nn

    class Probe(nn.Module):
        def __init__(self):
            super().__init__()
            self.fighters = len(c["fighters"])
            self.cnn = nn.Sequential(nn.Conv2d(c["stack"], 32, 8, 4), nn.ReLU(), nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
                                     nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten(),
                                     nn.LazyLinear(256), nn.ReLU()) if self.fighters else None
            self.mlp = nn.Sequential(nn.Linear(vector_size(c), 256), nn.ReLU())
            width = 256 * (1 + self.fighters)
            self.gru = nn.GRU(width, 128, batch_first=True) if c["recurrent"] else None
            self.trunk = nn.Sequential(nn.Linear(width + (128 if c["recurrent"] else 0), 256), nn.ReLU(),
                                       nn.Linear(256, 256), nn.ReLU())
            self.heads = nn.Linear(256, 1 + 4 + 9 + 6)

        def forward(self, vec, img, h=None):  # vec [B, T, n]; img [B, T, fighters x stack, S, S] or None
            b, t = vec.shape[:2]
            parts = [self.mlp(vec)]
            if self.cnn is not None:  # one CNN for both fighters
                z = self.cnn(img.reshape(b * t * self.fighters, -1, *img.shape[-2:]))
                parts.append(z.reshape(b, t, -1))
            z = torch.cat(parts, -1)
            if self.gru is not None:
                g, h = self.gru(z, h)
                z = torch.cat([z, g], -1)
            return self.heads(self.trunk(z)), h

    model = Probe().to(device)
    with torch.no_grad():  # materialise the lazy layer
        model(torch.zeros(1, 1, vector_size(c), device=device), blank(c, device))
    return model


def blank(c: dict, device: str, size: int = 96):
    import torch

    channels = len(c["fighters"]) * c["stack"]
    return torch.zeros(1, 1, channels, size, size, device=device) if channels else None


def to_input(img, train: bool):
    """uint8 [B, T, C, S, S] -> float; while training, one random shift of up to 4 px per window."""
    import torch
    import torch.nn.functional as fn

    b, t = img.shape[:2]
    x = img.reshape(b * t, *img.shape[2:]).float().div_(255)
    if train:
        theta = torch.zeros(b, 2, 3, device=x.device)
        theta[:, 0, 0] = theta[:, 1, 1] = 1
        theta[:, :, 2] = (torch.rand(b, 2, device=x.device) * 2 - 1) * (8 / 96)
        grid = fn.affine_grid(theta.repeat_interleave(t, 0), x.shape, align_corners=False)
        x = fn.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
    return x.reshape(b, t, *x.shape[1:])


def losses(out, t):
    import torch.nn.functional as fn

    hit = fn.binary_cross_entropy_with_logits(out[:, 0], t["hit"], reduction="none")
    move = ((out[:, 1:5] - t["move"]) ** 2).mean(1)
    act = fn.cross_entropy(out[:, 5:14], t["action"][:, 0], reduction="none") + \
        fn.cross_entropy(out[:, 14:20], t["action"][:, 1], reduction="none")
    return hit, move, act


def metrics(out: np.ndarray, sessions: list[Session], idx: list[np.ndarray]) -> dict:
    """Scores of outputs [N, 20] for decisions ``idx`` of ``sessions``, concatenated in that order."""
    import torch

    cat = lambda name: np.concatenate([getattr(s, name)[i] for s, i in zip(sessions, idx)])  # noqa: E731
    hit, hit_ok, move, move_ok, action, previous = (cat(n) for n in ("hit", "hit_ok", "move", "move_ok", "action", "previous"))
    t = {"hit": torch.as_tensor(hit), "move": torch.as_tensor(move), "action": torch.as_tensor(action)}
    act = losses(torch.as_tensor(out), t)[2].numpy()
    p, y = 1 / (1 + np.exp(-out[:, 0][hit_ok].astype(np.float64))), hit[hit_ok]
    rate = float(y.mean())
    logloss = float(-np.mean(y * np.log(p + 1e-6) + (1 - y) * np.log(1 - p + 1e-6)))
    ranks = np.empty(len(p))
    ranks[np.argsort(p)] = np.arange(1, len(p) + 1)
    auc = float((ranks[y > 0].sum() - y.sum() * (y.sum() + 1) / 2) / max(y.sum() * (len(y) - y.sum()), 1))
    rmse = lambda e, j: float(np.sqrt((e[move_ok][:, 2 * j:2 * j + 2] ** 2).sum(1).mean()))  # noqa: E731
    err, changed = (move - out[:, 1:5]) * 100.0, (action != previous).any(1)
    return {"hit_rate": rate, "hit_logloss": logloss, "hit_auc": auc,
            "hit_logloss_base": float(-(rate * np.log(rate) + (1 - rate) * np.log(1 - rate))),
            "move_rmse_5": rmse(err, 0), "move_rmse_10": rmse(err, 1),
            "velocity_rmse_5": rmse(move * 100.0, 0), "velocity_rmse_10": rmse(move * 100.0, 1),
            "act_nll": float(act.mean()), "act_nll_changed": float(act[changed].mean())}


def run(sessions, c: dict, train_ids, test_ids, seed: int, device: str, epochs: int, log, deadline=None) -> dict:
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build(c, device)
    optim = torch.optim.Adam(model.parameters(), lr=5e-4)
    x, tgt = {}, {}
    for i, s in enumerate(sessions):
        x[i] = torch.as_tensor(s.obs, device=device).clone()
        if not c["timers"]:
            x[i][:, len(observation.BASE):] = 0
        if c.get("extra"):
            if s.reader is None:
                raise SystemExit(f"{s.name}: no reader outputs (scripts/train_reader.py)")
            x[i] = torch.cat([x[i], torch.as_tensor(s.reader[c["extra"]].astype(np.float32), device=device)], 1)
        tgt[i] = {k: torch.as_tensor(getattr(s, k), device=device) for k in ("hit", "move", "action", "hit_ok", "move_ok")}
    cut = {i: int(len(sessions[i].obs) * (1 - TAIL)) for i in train_ids}

    def predict(i: int, a: int, b: int) -> np.ndarray:
        """Outputs for decisions a..b of session i, memory carried from a."""
        model.eval()
        out, h = [], None
        with torch.no_grad():
            for start in range(a, b, 512):
                idx = np.arange(start, min(b, start + 512))
                img = to_input(sessions[i].images(idx, c, device)[None], False) if c["fighters"] else None
                o, h = model(x[i][start:start + len(idx)][None], img, h)
                out.append(o[0].float().cpu().numpy())
        return np.concatenate(out)

    best, state, since, t0, clock = None, None, 0, time.perf_counter(), False
    for epoch in range(1, epochs + 1):
        model.train()
        windows = [(i, a) for i in train_ids for a in range(int(rng.integers(LENGTH)), cut[i] - LENGTH, LENGTH)]
        order = rng.permutation(len(windows))
        for b in range(0, len(order), PER_BATCH):
            batch = [windows[j] for j in order[b:b + PER_BATCH]]
            img = None
            if c["fighters"]:
                img = to_input(torch.stack([sessions[i].images(np.arange(a, a + LENGTH), c, device) for i, a in batch]), True)
            out = model(torch.stack([x[i][a:a + LENGTH] for i, a in batch]), img)[0].reshape(-1, 20)
            t = {k: torch.cat([tgt[i][k][a:a + LENGTH] for i, a in batch]) for k in ("hit", "move", "action", "hit_ok", "move_ok")}
            hit, move, act = losses(out, t)
            loss = (hit * t["hit_ok"]).mean() + (move * t["move_ok"]).mean() + act.mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
        tails = [np.arange(cut[i], len(sessions[i].obs)) for i in train_ids]
        v = metrics(np.concatenate([predict(i, cut[i], len(sessions[i].obs)) for i in train_ids]),
                    [sessions[i] for i in train_ids], tails)
        score = v["hit_logloss"] + v["act_nll"] + (v["move_rmse_10"] / 100) ** 2
        if best is None or score < best:
            best, since = score, 0
            state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            since += 1
        if epoch == 1 or epoch % 10 == 0 or since >= PATIENCE:
            log(f"    epoch {epoch}: stop score {score:.4f} (best {best:.4f}), {(time.perf_counter() - t0) / epoch:.1f} s per epoch")
        if since >= PATIENCE:
            break
        if deadline is not None and time.time() > deadline:
            clock = True
            log(f"    out of time at epoch {epoch}: the best epoch so far is kept")
            break
    model.load_state_dict(state)
    r = metrics(np.concatenate([predict(i, 0, len(sessions[i].obs)) for i in test_ids]), [sessions[i] for i in test_ids],
                [np.arange(len(sessions[i].obs)) for i in test_ids])
    r.update(epochs=epoch, best_epoch=epoch - since, stopped_by_clock=clock, cost_ms=cost_ms(model, c))
    return r


def cost_ms(model, c: dict) -> dict:
    """One decision's forward pass on this machine: one CPU thread, and the GPU."""
    import torch

    out, threads = {}, torch.get_num_threads()
    for device in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
        m = model.to(device).eval()
        vec, img = torch.zeros(1, 1, vector_size(c), device=device), blank(c, device)
        torch.set_num_threads(1)
        with torch.no_grad():
            ts = []
            for i in range(220):
                t = time.perf_counter()
                m(vec, img)
                if device == "cuda":
                    torch.cuda.synchronize()
                if i >= 20:
                    ts.append((time.perf_counter() - t) * 1000)
        torch.set_num_threads(threads)
        out[device] = float(np.median(ts))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--difficulty", default="medium")
    p.add_argument("--only", default="", help="candidate names separated by ; (default: all)")
    p.add_argument("--epochs", type=int, default=300, help="at most; training stops when the stop score does")
    p.add_argument("--seeds", type=int, default=2, help="seeds per fold")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--gpu", default=None, help="which GPU this process uses (one process per GPU)")
    p.add_argument("--tag", default="", help="names this process's record file: folds_<tag>.json")
    p.add_argument("--part", default="0/1", help="i/n: this process takes every n-th run, starting at the i-th")
    p.add_argument("--hours", type=float, default=None, help="stop cleanly after this long (Kaggle ends a notebook at 12 h)")
    p.add_argument("--report", action="store_true", help="only write the report from the runs recorded so far")
    args = p.parse_args(argv)
    deadline = None if args.hours is None else time.time() + args.hours * 3600
    out_dir = RUNS / "probes" / "vision"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.report:
        write_report(out_dir, args)
        return 0
    if args.gpu is not None:  # before anything starts CUDA
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    roots = [q for q in sorted((DATA / "demos").iterdir()) if (q / f"crops_k{args.k}_{args.size}.npy").exists()
             and is_demo(Recording(q).meta) and Recording(q).meta["setting"].get("difficulty") in set(args.difficulty.split(","))]
    if len(roots) < 3:
        raise SystemExit(f"need at least 3 sessions with crops; found {len(roots)} (scripts/prepare_crops.py)")
    sessions = [Session(r, args.k, args.size) for r in roots]
    folds = [[i for i in range(len(roots)) if i % 3 == f] for f in range(3)]
    names = [n for n in CANDIDATES if not args.only or n in args.only.split(";")]
    record = out_dir / f"folds{'_' + args.tag if args.tag else ''}.json"
    done = json.loads(record.read_text()) if record.exists() else {}
    print(f"{len(roots)} sessions, {sum(len(s.obs) for s in sessions)} decisions; folds hold out {folds}; {device}", flush=True)
    todo = [(seed, f, name) for seed in range(args.seeds) for f in range(len(folds)) for name in names]
    part, parts = (int(v) for v in args.part.split("/"))
    light = [r for r in todo if not CANDIDATES[r[2]]["fighters"]][part::parts]  # each kind shared out evenly
    heavy = [r for r in todo if CANDIDATES[r[2]]["fighters"]][part::parts]
    todo = [r for r in todo if r in light or r in heavy]
    need = sum(s.crops.size for s in sessions)
    if heavy and device == "cuda":
        if need <= torch.cuda.get_device_properties(0).total_memory / 2:  # beyond it Windows spills into RAM
            for s in sessions:
                s.gpu = torch.empty(s.crops.shape, dtype=torch.uint8, device=device)
                for a in range(0, len(s.obs), 4000):
                    s.gpu[a:a + 4000] = torch.from_numpy(np.array(s.crops[a:a + 4000]))
            print(f"crops on the GPU ({need / 1e9:.1f} GB)", flush=True)
        else:
            print(f"crops take {need / 1e9:.1f} GB, over half the GPU: read from disk (slower)", flush=True)
    for n, (seed, f, name) in enumerate(todo):
        key = f"{name} | fold {f} | seed {seed}"
        if key in done:
            continue
        if deadline is not None and time.time() > deadline - 300:
            print(f"out of time: {sum(f'{c} | fold {b} | seed {a}' not in done for a, b, c in todo[n:])} runs not started", flush=True)
            break
        test_ids = folds[f]
        print(f"{key} (held out: {', '.join(roots[i].name for i in test_ids)})", flush=True)
        train_ids = [i for i in range(len(roots)) if i not in test_ids]
        r = run(sessions, CANDIDATES[name], train_ids, test_ids, 10 * f + seed, device, args.epochs,
                lambda m: print(m, flush=True), deadline)
        done[key] = r
        record.write_text(json.dumps(done, indent=1))
        print(f"  -> hit AUC {r['hit_auc']:.3f}; Mordex in 0.5 s {r['move_rmse_10']:.1f} px (velocity guess "
              f"{r['velocity_rmse_10']:.1f}); action NLL when changed {r['act_nll_changed']:.3f} "
              f"(best epoch {r['best_epoch']} of {r['epochs']})", flush=True)
    write_report(out_dir, args)
    return 0


def write_report(out_dir: Path, args) -> None:
    done: dict = {}
    for path in sorted(out_dir.glob("folds*.json")):
        done.update(json.loads(path.read_text()))
    if not done:
        raise SystemExit(f"no runs recorded in {out_dir}")
    runs = {name: {k.split(" | ", 1)[1]: r for k, r in done.items() if k.split(" | ", 1)[0] == name} for name in CANDIDATES}
    base = runs["vector"]
    keys = (("hit_auc", "Hit within 0.5 s: AUC", 3), ("move_rmse_5", "Mordex in 0.25 s (px)", 1),
            ("move_rmse_10", "Mordex in 0.5 s (px)", 1), ("act_nll_changed", "Action NLL when the human changes", 3),
            ("act_nll", "Action NLL", 3))
    rows = []
    for name, rs in runs.items():
        if not rs:
            continue
        cells = []
        for key, _, d in keys:
            v = np.array([r[key] for r in rs.values()])
            shared = [k for k in rs if k in base]
            diff = np.array([rs[k][key] - base[k][key] for k in shared])
            cells.append(f"{v.mean():.{d}f} ± {v.std():.{d}f}" + (
                "" if name == "vector" or not shared else f" ({diff.mean():+.{d}f} ± {diff.std():.{d}f})"))
        cost = [r["cost_ms"] for r in rs.values()]
        ms = " / ".join(f"{np.mean([c[d] for c in cost if d in c]):.2f}" for d in ("cpu", "cuda") if any(d in c for c in cost))
        rows.append(f"| {name} | {len(rs)} | " + " | ".join(cells) + f" | {np.mean([r['best_epoch'] for r in rs.values()]):.0f} | {ms} |")
    every = list(done.values())
    mean = lambda k: float(np.mean([r[k] for r in every]))  # noqa: E731
    lines = ["# Gates S1 and S2 probes: what memory and a picture of the fighters add", "",
             f"{datetime.now():%Y-%m-%d %H:%M}. {args.difficulty} sessions held out whole (3 folds), trained until the stop "
             "score stops improving. Each cell: mean ± spread over the runs, then (for inputs other than `vector`) the "
             "mean difference from `vector` in the same fold and seed ± its spread. Lower is better except AUC. "
             "scripts/probe_vision.py.", "",
             f"- Wu Shang is hit within 0.5 s after {mean('hit_rate'):.1%} of decisions.",
             f"- Guessing that Mordex keeps his velocity misses by {mean('velocity_rmse_5'):.1f} px at 0.25 s and "
             f"{mean('velocity_rmse_10'):.1f} px at 0.5 s.",
             "- Choosing among the 54 actions at random gives an action NLL of 3.99.",
             f"- Runs interrupted by the time limit (their best epoch so far is used): "
             f"{sum(bool(r.get('stopped_by_clock')) for r in every)} of {len(every)}.", "",
             "| Input | Runs | " + " | ".join(label for _, label, _ in keys) + " | Best epoch | Cost per decision here, CPU / GPU (ms) |",
             "|---|---|" + "---|" * (len(keys) + 2), *rows, "",
             "A candidate earns its place when its difference from `vector` is larger than that difference's spread."]
    path = RESULTS / "probes" / "vision.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), f"\n-> {path}")


if __name__ == "__main__":
    sys.exit(main())
