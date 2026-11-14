"""Recurrent Q-learning from demonstrations and live play: the agent's main learner.

R2D2-style (Kapturowski et al. 2019) with one actor: a GRU Q-network over the observation vector, a
dueling head over the 54 actions, double Q, n-step returns, and sequences replayed with a burn-in that
rebuilds the memory before the loss is taken. DQfD-style (Hester et al. 2018): the human
demonstrations stay in replay for good, and on the demonstrations of winning stretches a margin loss
keeps the human's action above the others, so the agent starts from what the human did and Q-learning
can then move away from it.

The actor (the game loop: brawl/learn.py ``train_rdqn``) and the learner (``learner_main``, its own
process, on the CPU by default so that the detector keeps the GPU) share a queue of new decisions,
a weights file, and two counters. In real time the learner never makes the game wait; in stepped mode
(the game paused between decisions) the actor waits for the learner to keep up.
"""
from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import numpy as np

from brawl import actions, observation

N_ACTIONS = actions.SHAPE[0] * actions.SHAPE[1]
# The last actions, one-hot: left out of the network's input by default. With them the pretrained network
# learned to repeat whatever the history said (88-90% of its choices on held-out play, 2026-11-13), and
# playing greedily it then repeated its own last action until it walked off the stage. The GRU and the
# timers still carry what happened.
HISTORY = [i for i, name in enumerate(observation.FEATURES) if name.startswith("action_")]


def joint(action) -> int:
    return int(action[0]) * actions.SHAPE[1] + int(action[1])


def split(j: int) -> tuple[int, int]:
    return divmod(int(j), actions.SHAPE[1])


def build(n_in: int, hidden: int, device: str = "cpu", drop: list[int] | None = None):
    import torch
    from torch import nn

    keep = [i for i in range(n_in) if i not in set(drop or [])]

    class QNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.drop = list(drop or [])
            self.register_buffer("keep", torch.as_tensor(keep, dtype=torch.long), persistent=False)
            self.inp = nn.Sequential(nn.Linear(len(keep), 256), nn.ReLU())
            self.gru = nn.GRU(256, hidden, batch_first=True)
            self.value = nn.Sequential(nn.Linear(hidden, 256), nn.ReLU(), nn.Linear(256, 1))
            self.advantage = nn.Sequential(nn.Linear(hidden, 256), nn.ReLU(), nn.Linear(256, N_ACTIONS))

        def forward(self, x, h=None):
            """x [B, T, n_in] -> Q [B, T, 54], the GRU's last state."""
            z, h = self.gru(self.inp(x.index_select(-1, self.keep)), h)
            a = self.advantage(z)
            return self.value(z) + a - a.mean(-1, keepdim=True), h

    return QNet().to(device)


def save(net, path: Path, n_in: int, hidden: int) -> None:
    """Atomically: a reader never sees half a file."""
    import torch

    tmp = Path(path).with_suffix(".tmp")
    torch.save({"state": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "n_in": n_in, "hidden": hidden,
                "features": list(observation.FEATURES), "drop": list(getattr(net, "drop", []))}, tmp)
    for attempt in range(50):  # Windows refuses while the actor is reading the old file: wait for it
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.02 * (attempt + 1))
    raise PermissionError(f"could not replace {path}")


def load(path: Path, device: str = "cpu"):
    import torch

    saved = torch.load(path, map_location=device, weights_only=False)
    if saved["features"] != list(observation.FEATURES):
        raise SystemExit(f"{path}: trained on another observation vector")
    net = build(saved["n_in"], saved["hidden"], device, saved.get("drop", []))
    net.load_state_dict(saved["state"])
    return net, saved


class Greedy:
    """A trained Q-network as a player: the best action at each decision, memory carried along."""

    def __init__(self, path: Path, epsilon: float = 0.0, seed: int = 0) -> None:
        import torch

        self.torch, self.net, self.h = torch, load(path)[0].eval(), None
        self.epsilon, self.rng = epsilon, np.random.default_rng(seed)

    def reset(self) -> None:
        self.h = None

    def __call__(self, obs: np.ndarray) -> tuple[int, int]:
        with self.torch.no_grad():
            q, self.h = self.net(self.torch.as_tensor(obs, dtype=self.torch.float32)[None, None], self.h)
        if self.rng.random() < self.epsilon:
            return split(int(self.rng.integers(N_ACTIONS)))
        return split(int(q[0, 0].argmax()))


class Replay:
    """A stream of decisions sampled as sequences that never cross a cut (the start of a session, or
    a gap in the live run). Observations are kept in float16: 224 bytes a decision."""

    def __init__(self, n_in: int, capacity: int) -> None:
        self.obs = np.zeros((capacity, n_in), np.float16)
        self.action = np.zeros(capacity, np.int64)
        self.reward = np.zeros(capacity, np.float32)
        self.margin = np.zeros(capacity, bool)  # the margin loss applies (a demonstration of a winning stretch)
        self.cuts = np.zeros(capacity + 1, np.int64)  # cuts[i]: cuts among rows 0..i-1
        self.n, self.capacity = 0, capacity

    def add(self, obs, action, reward, cut, margin) -> None:
        k = len(obs)
        if self.n + k > self.capacity:  # drop the oldest tenth
            drop = max(k, self.capacity // 10)
            keep = self.n - drop
            for a in (self.obs, self.action, self.reward, self.margin):
                a[:keep] = a[drop:self.n]
            self.cuts[1:keep + 1] = self.cuts[drop + 1:self.n + 1] - self.cuts[drop]
            self.n = keep
            cut = np.array(cut, bool)
            cut[0] = True
        s = slice(self.n, self.n + k)
        self.obs[s], self.action[s], self.reward[s], self.margin[s] = obs, action, reward, margin
        self.cuts[self.n + 1:self.n + k + 1] = self.cuts[self.n] + np.cumsum(np.asarray(cut, np.int64))
        self.n += k

    def sample(self, k: int, length: int, rng) -> np.ndarray | None:
        """[k, length] row indices of sequences with no cut after their first row; None if too few."""
        if self.n < length + 1:
            return None
        out = []
        for _ in range(k * 50):
            s = int(rng.integers(0, self.n - length + 1))
            if self.cuts[s + length] - self.cuts[s + 1] == 0:  # no cut in rows s+1 .. s+length-1
                out.append(np.arange(s, s + length))
                if len(out) == k:
                    return np.stack(out)
        return np.stack(out) if out else None


def demonstrations(roots, k: int, damage_weight: float, ko_weight: float, margin_keep=("won",), log=print) -> Replay:
    """The human demonstrations as one replay: each session is a stream, its reward from the same events
    as the live reward, and the margin loss on the stretches ending in ``margin_keep``."""
    from brawl.learn import stretches

    parts = []
    for root in roots:
        path = Path(root) / f"decisions_k{k}.npz"
        z = np.load(path)
        if tuple(z["features"]) != observation.FEATURES:
            log(f"{Path(root).name}: prepared with another feature list; skipped")
            continue
        reward = damage_weight * (z["dealt"] - z["taken"]) + ko_weight * (z["ko_scored"] - z["ko_lost"])
        margin = np.zeros(len(reward), bool)
        for a, b, outcome in stretches(z["ko_scored"], z["ko_lost"]):
            margin[a:b] = outcome in margin_keep
        parts.append((z["obs"], z["action"][:, 0] * actions.SHAPE[1] + z["action"][:, 1], reward, margin))
    if not parts:
        raise SystemExit("no demonstrations to learn from (scripts/prepare_demos.py)")
    replay = Replay(observation.SIZE, sum(len(p[0]) for p in parts))
    for obs, action, reward, margin in parts:
        cut = np.zeros(len(obs), bool)
        cut[0] = True
        replay.add(obs, action, reward, cut, margin)
    return replay


def losses(net, target, obs, action, reward, margin_mask, cfg: dict, gamma: float):
    """TD (n-step, double Q, Huber) and margin losses over the trained part of a batch of sequences."""
    import torch
    import torch.nn.functional as fn

    b, t, n = int(cfg["burn_in"]), int(cfg["train_length"]), int(cfg["n_step"])
    q, _ = net(obs)
    with torch.no_grad():
        q_target, _ = target(obs)
        steps = torch.arange(b, b + t, device=obs.device)
        ret = sum(gamma ** j * reward[:, steps + j] for j in range(n))
        if cfg.get("bootstrap", "double_q") == "sarsa":  # the value of the action actually taken next
            best = action[:, steps + n, None]
        else:  # double Q: the online network picks, the target network values
            best = q[:, steps + n].detach().argmax(-1, keepdim=True)
        y = ret + gamma ** n * q_target[:, steps + n].gather(-1, best).squeeze(-1)
    q_t = q[:, steps]
    taken = action[:, steps]
    q_a = q_t.gather(-1, taken[..., None]).squeeze(-1)
    td = fn.smooth_l1_loss(q_a, y)
    m = torch.full_like(q_t, float(cfg["margin"]))
    m.scatter_(-1, taken[..., None], 0.0)
    mask = margin_mask[:, steps].float()
    margin = (((q_t + m).max(-1).values - q_a) * mask).sum() / mask.sum().clamp(min=1.0)
    return td, margin, float(q_a.detach().mean())


class Learner:
    def __init__(self, n_in: int, cfg: dict, gamma: float, device: str, init: Path | None = None, seed: int = 0):
        import copy

        import torch

        torch.manual_seed(seed)
        self.cfg, self.gamma, self.device, self.n_in = cfg, gamma, device, n_in
        drop = HISTORY if cfg.get("drop_history", True) else []
        self.net = load(init, device)[0] if init else build(n_in, int(cfg["hidden"]), device, drop)
        self.target = copy.deepcopy(self.net).eval()
        self.target.gru.flatten_parameters()
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=float(cfg["learning_rate"]), weight_decay=1e-5)
        self.updates, self.rng = 0, np.random.default_rng(seed)
        self.length = int(cfg["burn_in"]) + int(cfg["train_length"]) + int(cfg["n_step"])

    def batch(self, sources: list[tuple[Replay, int, bool]]):
        """sources: (replay, sequences, is_demonstration)."""
        import torch

        rows = []
        for replay, k, is_demo in sources:
            idx = replay.sample(k, self.length, self.rng) if k else None
            if idx is not None:
                rows.append((replay, idx, is_demo))
        if not rows:
            return None
        cat = lambda f: np.concatenate([f(r, i, d) for r, i, d in rows])  # noqa: E731
        as_t = lambda a, dtype: torch.as_tensor(a, dtype=dtype, device=self.device)  # noqa: E731
        return (as_t(cat(lambda r, i, d: r.obs[i].astype(np.float32)), torch.float32),
                as_t(cat(lambda r, i, d: r.action[i]), torch.long),
                as_t(cat(lambda r, i, d: r.reward[i]), torch.float32),
                as_t(cat(lambda r, i, d: r.margin[i] & d), torch.bool))  # the margin loss only on demonstrations

    def step(self, sources, margin_weight: float) -> dict | None:
        import torch

        b = self.batch(sources)
        if b is None:
            return None
        self.net.train()
        td, margin, q = losses(self.net, self.target, *b, self.cfg, self.gamma)
        loss = td + margin_weight * margin
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        self.opt.step()
        self.updates += 1
        if self.updates % int(self.cfg["target_every"]) == 0:
            self.target.load_state_dict(self.net.state_dict())
        return {"td": float(td.detach()), "margin": float(margin.detach()), "q": q}


def pretrain(learner: Learner, demos: Replay, updates: int, log=print) -> None:
    """Learn from the demonstrations alone, before the first live decision."""
    t0, acc = time.perf_counter(), []
    for i in range(1, updates + 1):
        r = learner.step([(demos, int(learner.cfg["batch"]), True)], float(learner.cfg["margin_weight"]))
        acc.append((r["td"], r["margin"], r["q"]))
        if i % 2000 == 0 or i == updates:
            m = np.mean(acc, 0)
            log(f"  pretrain {i}/{updates}: td {m[0]:.4f}, margin {m[1]:.4f}, mean Q {m[2]:.3f} "
                f"({(time.perf_counter() - t0) / i * 1000:.1f} ms per update)")
            acc = []


LEARNER_LOG = ("updates", "decisions", "live", "td", "margin", "q", "updates_per_s", "wall_s")


def learner_main(cfg: dict, gamma: float, demo_roots: list[str], k: int, reward_weights: tuple[float, float],
                 run_dir: str, init: str, queue, decisions, updates, stop, seed: int) -> None:
    """The learner process: replay = the demonstrations + the live decisions arriving on ``queue``;
    updates at most ``updates_per_decision`` times the decisions played; weights to latest.pt."""
    import queue as queues

    import torch

    torch.set_num_threads(int(cfg["threads"]))
    run = Path(run_dir)
    demos = demonstrations(demo_roots, k, *reward_weights, log=lambda m: None)
    learner = Learner(observation.SIZE, cfg, gamma, cfg["device"], Path(init), seed)
    live = Replay(observation.SIZE, int(cfg["capacity"]))
    per = float(cfg["updates_per_decision"])
    batch, share = int(cfg["batch"]), float(cfg["demo_fraction"])
    log = open(run / "learner.csv", "w", newline="", encoding="utf-8")
    rows = csv.DictWriter(log, fieldnames=LEARNER_LOG)
    rows.writeheader()
    t0, last, acc = time.perf_counter(), (time.perf_counter(), 0), []
    try:
        while not stop.is_set():
            try:
                while True:
                    obs, action, reward, cut = queue.get_nowait()
                    live.add(obs, action, reward, cut, np.zeros(len(obs), bool))
            except queues.Empty:
                pass
            if learner.updates >= per * decisions.value:
                time.sleep(0.01)
                continue
            n_demo = batch if live.n < learner.length * 4 else max(1, round(batch * share))
            r = learner.step([(demos, n_demo, True), (live, batch - n_demo, False)], float(cfg["live_margin_weight"]))
            if r is None:
                time.sleep(0.01)
                continue
            updates.value = learner.updates
            acc.append((r["td"], r["margin"], r["q"]))
            if learner.updates % int(cfg["sync_every"]) == 0:
                try:
                    save(learner.net, run / "latest.pt", observation.SIZE, int(cfg["hidden"]))
                except PermissionError:  # the next sync will do
                    pass
            if learner.updates % 500 == 0:
                now = time.perf_counter()
                m = np.mean(acc, 0)
                rows.writerow({"updates": learner.updates, "decisions": decisions.value, "live": live.n,
                               "td": round(float(m[0]), 5), "margin": round(float(m[1]), 5), "q": round(float(m[2]), 4),
                               "updates_per_s": round((learner.updates - last[1]) / max(now - last[0], 1e-9), 2),
                               "wall_s": round(now - t0, 1)})
                log.flush()
                last, acc = (now, learner.updates), []
    finally:
        save(learner.net, run / "latest.pt", observation.SIZE, int(cfg["hidden"]))
        log.close()
