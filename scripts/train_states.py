"""The fighter-state network: what Mordex is doing, from his last few crops (runs on Kaggle; numpy and torch only).

    python scripts/train_states.py --labels <clips_labelled.npz> --pretrain <dir of shard_*.npz> [--encoder pretrained.pt]

Each 96 px grey crop becomes one token (a small CNN). A 2-layer transformer over the tokens of the newest
frame and the 5 before it reads the context, and the newest frame's token gives two sets of probabilities
over neutral, attacking, dodging, recovering, hurt: Mordex's state **now**, and **100 ms ahead** (2
decisions: trained on the frames that end 2 before the labelled one). Only past frames go in, as in the
game; the future is used only for the targets. The agent will get both distributions, not the argmax.

**Pretraining, labelled for free by the night traces.** For each of Wu Shang's crops: how many decisions ago
each button was pressed (jump, light, heavy, dodge, throw) and he was last hit, in 6 bins each. For each of
Mordex's crops: whether Wu Shang is hit within the next 0.5 s (Mordex's attack is coming), and how long ago
Mordex was hit. This teaches what attacks, dodges and hits look like and when a hit is coming; the hand
labels of Mordex then only teach the last step.

Four variants, scored on the hand labels by time: 4 folds, each holding out two eighths of each night, half
the night apart (as scripts/value_bench_vision.py). In brackets the mean difference from `frame` over the
same folds, and its standard error.

    frame              the newest frame alone, from scratch
    context            + the 5 frames before it, from scratch
    frame, pretrained  the newest frame alone, from the pretrained network
    context, pretrained

Writes results/states/fighter_state.md, and states/{pretrained.pt, state_net.pt} (the last: the last
variant, trained on every label) under --out.
"""
from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path

import numpy as np

STATES = ("neutral", "attacking", "dodging", "recovering", "hurt")
TARGETS = ("jump", "light", "heavy", "dodge", "throw", "hit")  # Wu Shang: decisions since each press, and since the last hit
OPP_TARGETS = ("hits_soon", "hit")  # Mordex: Wu Shang hit within SOON decisions (2 classes); decisions since Mordex was hit
BINS = (2, 4, 6, 9, 14)  # since <= 2, <= 4, <= 6, <= 9, <= 14, more: 6 classes
K = len(BINS) + 1
SOON = 10                # decisions: 0.5 s
AHEAD = 2                # decisions: the second head's horizon (100 ms)
T = 6                    # frames per example: the labelled one and the 5 before it
SEGMENT = 87500          # decisions: an eighth of a 700,000-decision night
FOLDS = 4


def bin_since(decisions: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.array(BINS), decisions, side="left").astype(np.int8)


def build(d: int = 128):
    import torch
    from torch import nn

    def conv(i, o, k, s):
        return [nn.Conv2d(i, o, k, s, k // 2), nn.GroupNorm(8, o), nn.GELU()]

    class StateNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(*conv(1, 32, 5, 2), *conv(32, 64, 3, 2), *conv(64, 96, 3, 2), *conv(96, 128, 3, 2),
                                         nn.Flatten(), nn.Linear(128 * 6 * 6, d))
            self.pos = nn.Parameter(torch.zeros(1, T, d))
            layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.1, batch_first=True, norm_first=True)
            self.temporal = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)
            self.pretrain_head = nn.Linear(d, len(TARGETS) * K)
            self.opp_head = nn.Linear(d, 2 + K)
            self.state_head = nn.Linear(d, len(STATES))
            self.ahead_head = nn.Linear(d, len(STATES))

        def embed(self, x):  # x: [B, t <= T, 96, 96] in 0..1, the newest frame last
            b, t = x.shape[:2]
            z = self.encoder(x.reshape(b * t, 1, 96, 96)).reshape(b, t, -1) + self.pos[:, T - t:]
            if t > 1:
                z = self.temporal(z)
            return self.norm(z[:, -1])

    return StateNet()


def augment(x, gen):
    """The same shift, mirror and brightness over an example's frames; x: [B, t, 96, 96] in 0..1."""
    import torch
    import torch.nn.functional as F

    b = x.shape[0]
    theta = torch.zeros(b, 2, 3, device=x.device)
    flip = torch.where(torch.rand(b, device=x.device, generator=gen) < 0.5, -1.0, 1.0)  # facing does not change a state
    theta[:, 0, 0], theta[:, 1, 1] = flip, 1.0
    theta[:, :, 2] = (torch.rand(b, 2, device=x.device, generator=gen) - 0.5) * 0.16  # up to 8 px
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)
    gain = 0.8 + 0.4 * torch.rand(b, 1, 1, 1, device=x.device, generator=gen)
    bias = (torch.rand(b, 1, 1, 1, device=x.device, generator=gen) - 0.5) * 0.2
    return (x * gain + bias).clamp(0, 1)


def roc_auc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label == 1], score[label == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# -- pretraining on the free targets -----------------------------------------------------------------------
def free_targets(obs: np.ndarray, events: np.ndarray, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """One trace shard's free targets: Wu Shang [N, 6] and Mordex [N, 2] classes (-1: unknown). ``obs`` row t is
    what decision t saw (its crops too); ``events`` row t is what happened during decision t."""
    f = {n: i for i, n in enumerate(features)}
    clip = 80  # brawl.observation.CLIP: the timers are decisions / 80
    since = lambda name: np.rint(obs[:, f[name]].astype(np.float32) * clip).astype(int)  # noqa: E731
    agent = np.stack([bin_since(since(f"since_press_{b}")) for b in TARGETS[:-1]] + [bin_since(since("agent_since_hit"))], 1)
    taken = events[:, 1] > 0  # Wu Shang took damage during the decision
    soon = np.full(len(obs), -1, np.int8)
    n = len(obs) - SOON + 1
    if n > 0:
        hits = np.convolve(taken.astype(int), np.ones(SOON, int), "valid")  # hits in t .. t + SOON - 1
        soon[:n] = hits[:n] > 0
    opp = np.stack([soon, bin_since(since("opponent_since_hit"))], 1)
    return agent.astype(np.int8), opp.astype(np.int8)


class Pretrain:
    """Both fighters' crops and their free targets, from night-trace shards (brawl/learn.py: crops [N, 2, 96, 96],
    Wu Shang then Mordex; obs; events): windows of T frames inside one shard."""

    def __init__(self, files: list[Path]) -> None:
        self.crops, self.agent, self.opp, index = [], [], [], []
        for i, f in enumerate(files):
            z = np.load(f)
            self.crops.append(z["crops"])
            agent, opp = free_targets(z["obs"], z["events"], list(z["features"]))
            self.agent.append(agent)
            self.opp.append(opp)
            index += [(i, t) for t in range(T - 1, len(z["crops"]))]
        self.index = np.array(index)

    def batch(self, ids, who: int, device):
        import torch

        x = np.stack([self.crops[s][t - T + 1:t + 1, who] for s, t in self.index[ids]])
        y = np.stack([(self.agent if who == 0 else self.opp)[s][t] for s, t in self.index[ids]])
        return torch.as_tensor(x, device=device).float().div_(255), torch.as_tensor(y, device=device).long()


def pretrain_out(net, x, y, who: int):
    """(loss, predicted classes [B, targets], probability of "hit within 0.5 s" or None)."""
    import torch
    import torch.nn.functional as F

    z = net.embed(x)
    if who == 0:
        logits = net.pretrain_head(z).reshape(-1, len(TARGETS), K)
        return F.cross_entropy(logits.reshape(-1, K), y.reshape(-1), ignore_index=-1), logits.argmax(-1), None
    out = net.opp_head(z)
    soon, since = out[:, :2], out[:, 2:]
    loss = F.cross_entropy(soon, y[:, 0], ignore_index=-1) + F.cross_entropy(since, y[:, 1], ignore_index=-1)
    return loss, torch.stack([soon.argmax(-1), since.argmax(-1)], 1), torch.softmax(soon, -1)[:, 1]


def pretrain(files: list[Path], steps: int, device: str, log=print, batch: int = 128, seed: int = 0):
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    test_files = files[5::10] or files[-1:]  # a tenth, spread over the nights
    train, test = Pretrain([f for f in files if f not in test_files]), Pretrain(test_files)
    log(f"pretraining on {len(train.index)} decisions of both fighters ({len(files) - len(test_files)} shards), {len(test.index)} held out")
    net = build().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 500) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    gen = torch.Generator(device=device).manual_seed(seed)
    t0 = time.perf_counter()
    net.train()
    for step in range(1, steps + 1):
        loss = 0.0
        for who in (0, 1):  # half the batch each fighter
            x, y = train.batch(rng.integers(len(train.index), size=batch // 2), who, device)
            loss = loss + pretrain_out(net, augment(x, gen), y, who)[0]
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == steps:
            log(f"  {step}/{steps}: loss {loss.item():.3f} ({(time.perf_counter() - t0) / step * 1000:.0f} ms per update)")
    net.eval()
    names = [f"Wu Shang {n}" for n in TARGETS] + [f"Mordex {n}" for n in OPP_TARGETS]
    right, known = np.zeros(len(names)), np.zeros(len(names))
    counts = np.zeros((len(names), K))
    soon_p, soon_y = [], []
    with torch.no_grad():
        for who, first in ((0, 0), (1, len(TARGETS))):
            for a in range(0, len(test.index), 512):
                x, y = test.batch(np.arange(a, min(a + 512, len(test.index))), who, device)
                _, pred, p = pretrain_out(net, x, y, who)
                y, pred = y.cpu().numpy(), pred.cpu().numpy()
                for j in range(y.shape[1]):
                    ok = y[:, j] >= 0
                    right[first + j] += (pred[ok, j] == y[ok, j]).sum()
                    known[first + j] += ok.sum()
                    counts[first + j] += np.bincount(y[ok, j], minlength=K)[:K]
                if p is not None:
                    ok = y[:, 0] >= 0
                    soon_p.append(p.cpu().numpy()[ok])
                    soon_y.append(y[ok, 0])
    acc, base = right / np.maximum(known, 1), counts.max(1) / np.maximum(known, 1)
    auc = roc_auc(np.concatenate(soon_p), np.concatenate(soon_y))
    log("  held-out accuracy, " + "; ".join(f"{n}: {a:.3f} (commonest {b:.3f})" for n, a, b in zip(names, acc, base)) +
        f"; Wu Shang hit within 0.5 s from Mordex's crops: AUC {auc:.3f}")
    return net, {"names": names, "accuracy": acc, "baseline": base, "hits_soon_auc": auc, "decisions": len(train.index)}


# -- the hand labels ----------------------------------------------------------------------------------------
def window(x, frames: int):
    """From a clip's T frames (the labelled one last): the input of the "now" head (the last ``frames``) and of
    the "ahead" head (the last ``frames`` of those that end AHEAD decisions earlier)."""
    early = x[:, :T - AHEAD]
    return x[:, T - frames:], early[:, max(0, early.shape[1] - frames):]


def score(prob: np.ndarray, y: np.ndarray) -> dict:
    pred = prob.argmax(1)
    recall = np.array([(pred[y == c] == c).mean() if (y == c).any() else np.nan for c in range(len(STATES))])
    precision = np.array([(y[pred == c] == c).mean() if (pred == c).any() else np.nan for c in range(len(STATES))])
    both = np.nan_to_num(recall) + np.nan_to_num(precision)
    f1 = np.where(both > 0, 2 * np.nan_to_num(recall) * np.nan_to_num(precision) / np.maximum(both, 1e-9), 0.0)
    present = ~np.isnan(recall)
    return {"accuracy": float((pred == y).mean()), "balanced": float(np.nanmean(recall)), "macro_f1": float(f1[present].mean()),
            "log_loss": float(-np.log(np.clip(prob[np.arange(len(y)), y], 1e-6, 1)).mean()), "recall": recall}


def finetune(x_all: np.ndarray, y_all: np.ndarray, train: np.ndarray, frames: int, init, steps: int, device: str,
             seed: int = 0, batch: int = 64):
    """Both heads on the hand labels ``train`` (indices); ``frames``: 1 or T; ``init``: a pretrained net or None."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = (copy.deepcopy(init) if init is not None else build()).to(device)
    counts = np.bincount(y_all[train], minlength=len(STATES)).astype(float)
    weight = np.where(counts > 0, 1 / np.sqrt(np.maximum(counts, 1)), 0.0)
    weight = torch.as_tensor(weight / weight[counts > 0].mean(), dtype=torch.float32, device=device)  # rare states count more
    opt = torch.optim.AdamW(net.parameters(), lr=1.5e-4 if init is not None else 3e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 200) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    gen = torch.Generator(device=device).manual_seed(seed)
    net.train()
    for _ in range(steps):
        ids = train[rng.integers(len(train), size=batch)]
        x = augment(torch.as_tensor(x_all[ids], device=device).float().div_(255), gen)
        y = torch.as_tensor(y_all[ids], device=device).long()
        now, early = window(x, frames)
        loss = (F.cross_entropy(net.state_head(net.embed(now)), y, weight=weight) +
                F.cross_entropy(net.ahead_head(net.embed(early)), y, weight=weight))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    net.eval()
    return net


def predict(net, x_all: np.ndarray, ids: np.ndarray, frames: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Probabilities of the "now" head and of the "ahead" head."""
    import torch

    now, ahead = [np.zeros((0, len(STATES)))], [np.zeros((0, len(STATES)))]
    with torch.no_grad():
        for a in range(0, len(ids), 256):
            x = torch.as_tensor(x_all[ids[a:a + 256]], device=device).float().div_(255)
            xn, xe = window(x, frames)
            now.append(torch.softmax(net.state_head(net.embed(xn)), -1).cpu().numpy())
            ahead.append(torch.softmax(net.ahead_head(net.embed(xe)), -1).cpu().numpy())
    return np.concatenate(now), np.concatenate(ahead)


def table(results: dict) -> list[str]:
    names = list(results)
    base = results[names[0]]
    lines = ["| Variant | Accuracy | Balanced accuracy | Macro F1 | Log loss | " + " | ".join(f"Recall {s}" for s in STATES) + " |",
             "|---" * (5 + len(STATES)) + "|"]
    for name in names:
        cells = []
        for key in ("accuracy", "balanced", "macro_f1", "log_loss"):
            v = np.array([f[key] for f in results[name]])
            d = v - np.array([f[key] for f in base])
            spread = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
            cells.append(f"{v.mean():.3f}" + ("" if name == names[0] else f" ({d.mean():+.3f} ± {spread:.3f})"))
        recall = np.nanmean(np.array([f["recall"] for f in results[name]]), 0)
        lines.append(f"| {name} | " + " | ".join(cells) + " | " + " | ".join(f"{v:.2f}" for v in recall) + " |")
    return lines


def report(now: dict, ahead: dict, labels: np.ndarray, pre: dict | None, confusion: np.ndarray, timing: dict, args) -> str:
    counts = np.bincount(labels, minlength=len(STATES))
    lines = ["# What is Mordex doing? The fighter-state network", "",
             f"{time.strftime('%Y-%m-%d %H:%M')}. scripts/train_states.py. {len(labels)} hand labels (" +
             ", ".join(f"{s} {c}" for s, c in zip(STATES, counts)) + f"), {FOLDS} folds by time, {args.steps} updates per fold. "
             "Mean over folds; in brackets the mean difference from `frame` over the same folds, and its standard error.", ""]
    if pre is not None:
        lines += [f"Pretraining: {args.pretrain_steps} updates on {pre['decisions']} decisions of both fighters. Held-out accuracy "
                  "(in brackets: always the commonest class): " +
                  ", ".join(f"{n} {a:.2f} ({b:.2f})" for n, a, b in zip(pre["names"], pre["accuracy"], pre["baseline"])) +
                  f". Wu Shang hit within 0.5 s, from Mordex's crops alone: AUC {pre['hits_soon_auc']:.3f}.", ""]
    lines += ["## Mordex's state now", ""] + table(now)
    lines += ["", f"## Mordex's state {AHEAD * 50} ms ahead (from the frames that end {AHEAD} decisions before the labelled one)",
              ""] + table(ahead)
    lines += ["", f"Confusion of `{list(now)[-1]}`, now, over the held-out folds (rows: your label, columns: the network's):", "",
              "| | " + " | ".join(STATES) + " |", "|---" * (len(STATES) + 1) + "|"]
    lines += [f"| {s} | " + " | ".join(str(int(v)) for v in row) + " |" for s, row in zip(STATES, confusion)]
    lines += ["", "Cost per decision (one new frame encoded, the 5 before it cached as tokens, both heads): " +
              ", ".join(f"{k} {v:.2f} ms" for k, v in timing.items()), ""]
    return "\n".join(lines)


def cost(net, device: str) -> float:
    import torch

    net = net.to(device).eval()
    with torch.no_grad():
        one, cached = torch.rand(1, 1, 96, 96, device=device), torch.rand(1, T - 1, net.pos.shape[-1], device=device)

        def once():
            tok = net.encoder(one)[:, None]
            z = net.norm(net.temporal(torch.cat([cached, tok], 1) + net.pos)[:, -1])
            return net.state_head(z), net.ahead_head(z)

        for _ in range(5):
            once()
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            once()
        if device == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / 50 * 1000


def main(argv=None) -> int:
    import torch

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", type=Path, required=True, help="clips_labelled.npz (scripts/pack_states.py)")
    p.add_argument("--pretrain", type=Path, help="directory of shard_*.npz with both fighters' crops and free targets")
    p.add_argument("--encoder", type=Path, help="a pretrained.pt from an earlier run: no pretraining")
    p.add_argument("--pretrain-steps", type=int, default=20000)
    p.add_argument("--steps", type=int, default=3000, help="updates per fold on the hand labels")
    p.add_argument("--out", type=Path, default=Path("."))
    args = p.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def log(m):
        print(m, flush=True)

    z = np.load(args.labels)
    x_all, y_all, step = z["crops"], z["label"].astype(int), z["step"]
    fold = (step // SEGMENT) % FOLDS
    log(f"{len(y_all)} labelled clips on {device}; per fold {np.bincount(fold, minlength=FOLDS).tolist()}")
    (args.out / "states").mkdir(parents=True, exist_ok=True)
    pre, pre_info = None, None
    if args.encoder is not None:
        pre = build()
        pre.load_state_dict(torch.load(args.encoder, map_location="cpu"))
        log(f"pretrained network from {args.encoder}")
    elif args.pretrain is not None:
        pre, pre_info = pretrain(sorted(args.pretrain.rglob("shard_*.npz")), args.pretrain_steps, device, log)
        pre = pre.cpu()
        torch.save(pre.state_dict(), args.out / "states" / "pretrained.pt")
    variants = {"frame": (1, None), "context": (T, None)}
    if pre is not None:
        variants.update({"frame, pretrained": (1, pre), "context, pretrained": (T, pre)})
    now = {name: [] for name in variants}
    ahead = {name: [] for name in variants}
    confusion = np.zeros((len(STATES), len(STATES)), int)
    last = list(variants)[-1]
    for f in range(FOLDS):
        train, test = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        if not len(test) or not len(train):
            continue
        for name, (frames, init) in variants.items():
            net = finetune(x_all, y_all, train, frames, init, args.steps, device, seed=f)
            p_now, p_ahead = predict(net, x_all, test, frames, device)
            now[name].append(score(p_now, y_all[test]))
            ahead[name].append(score(p_ahead, y_all[test]))
            if name == last:
                np.add.at(confusion, (y_all[test], p_now.argmax(1)), 1)
            a, b = now[name][-1], ahead[name][-1]
            log(f"fold {f}, {name}: now balanced {a['balanced']:.3f}, macro F1 {a['macro_f1']:.3f}; "
                f"{AHEAD * 50} ms ahead balanced {b['balanced']:.3f}, macro F1 {b['macro_f1']:.3f}")
    frames, init = variants[last]
    final = finetune(x_all, y_all, np.arange(len(y_all)), frames, init, args.steps, device)
    torch.save({"state": final.cpu().state_dict(), "frames": frames, "states": STATES, "ahead": AHEAD, "variant": last},
               args.out / "states" / "state_net.pt")
    timing = {dev: cost(final, dev) for dev in ["cpu"] + (["cuda"] if device == "cuda" else [])}
    text = report(now, ahead, y_all, pre_info, confusion, timing, args)
    out = args.out / "results" / "states" / "fighter_state.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    log(text)
    log(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
