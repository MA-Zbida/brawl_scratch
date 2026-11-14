"""brawl/rdqn.py: replay that never crosses a cut, the losses, learning from demonstrations, the live loop."""
from __future__ import annotations

import numpy as np
import pytest

from brawl import actions, observation, rdqn


def stream(n, cut_at=(), n_in=4):
    obs = np.arange(n * n_in, dtype=np.float32).reshape(n, n_in)
    cut = np.zeros(n, bool)
    cut[list(cut_at)] = True
    return obs, np.arange(n) % 54, np.ones(n, np.float32), cut, np.zeros(n, bool)


def test_sequences_never_cross_a_cut_and_dropping_keeps_them_straight():
    r = rdqn.Replay(4, 100)
    obs, act, rew, cut, margin = stream(60, cut_at=(0, 30))
    r.add(obs, act, rew, cut, margin)
    idx = r.sample(200, 10, np.random.default_rng(0))
    assert idx.shape == (200, 10)
    assert not ((idx[:, 0] < 30) & (idx[:, -1] >= 30)).any()  # no sequence holds rows on both sides of row 30
    r.add(*stream(60, cut_at=(0,)))  # over capacity: the oldest rows go
    assert r.n <= 100 and r.cuts[r.n] - r.cuts[0] >= 1
    idx = r.sample(200, 10, np.random.default_rng(1))
    first = r.n - 60  # the second stream starts here: a cut
    assert not ((idx[:, 0] < first) & (idx[:, -1] >= first)).any()
    assert rdqn.Replay(4, 50).sample(4, 10, np.random.default_rng(0)) is None


def test_joint_actions_round_trip():
    assert all(rdqn.joint(rdqn.split(j)) == j for j in range(rdqn.N_ACTIONS))
    assert rdqn.split(rdqn.joint((actions.DIRECTION_NAMES.index("down-left"), 3))) == (7, 3)


def test_the_margin_loss_teaches_the_demonstrated_action_and_td_learns_the_return(tmp_path):
    import torch

    cfg = dict(hidden=32, burn_in=4, train_length=12, n_step=3, batch=16, learning_rate=3e-3, target_every=50, margin=0.5,
               margin_weight=1.0)
    rng = np.random.default_rng(0)
    n = 3000
    obs = rng.normal(size=(n, observation.SIZE)).astype(np.float32)
    human = np.where(obs[:, 0] > 0, 2 * 6 + 2, 1 * 6 + 3)  # right+light, or left+heavy, by one feature
    demos = rdqn.Replay(observation.SIZE, n)
    cut = np.zeros(n, bool)
    cut[0] = True
    demos.add(obs, human, np.full(n, 0.1, np.float32), cut, np.ones(n, bool))
    learner = rdqn.Learner(observation.SIZE, cfg, 0.9, "cpu", seed=0)
    rdqn.pretrain(learner, demos, 400, log=lambda m: None)
    with torch.no_grad():
        q, _ = learner.net(torch.as_tensor(obs[:500])[None])
    assert (q[0].argmax(-1).numpy() == human[:500]).mean() > 0.9
    assert 0.3 < float(q[0].max(-1).values.mean()) < 1.5  # a reward of 0.1 forever at gamma 0.9 is worth 1.0
    rdqn.save(learner.net, tmp_path / "q.pt", observation.SIZE, 32)
    player = rdqn.Greedy(tmp_path / "q.pt")
    assert rdqn.joint(player(obs[0])) == human[0]


def test_the_live_loop_plays_learns_in_another_process_and_saves(tmp_path):
    from test_env import Game

    from brawl import learn
    from brawl.brawlenv import BrawlEnv
    from brawl.sense.perception import Perception
    from types import SimpleNamespace

    n = 400
    demo = tmp_path / "demo"
    demo.mkdir()
    rng = np.random.default_rng(0)
    np.savez(demo / "decisions_k3.npz", obs=rng.normal(size=(n, observation.SIZE)).astype(np.float32),
             action=np.stack([rng.integers(9, size=n), rng.integers(6, size=n)], 1), dealt=np.zeros(n), taken=np.zeros(n),
             ko_scored=np.r_[np.zeros(n - 1), 1], ko_lost=np.zeros(n), features=np.array(observation.FEATURES))
    q = dict(hidden=32, burn_in=4, train_length=12, n_step=3, batch=4, learning_rate=1e-3, target_every=50, margin=0.5,
             margin_weight=1.0, live_margin_weight=0.3, demo_fraction=0.25, pretrain_updates=20, updates_per_decision=0.5,
             epsilon=0.1, explore="human", rollback=1.0, sync_every=10, capacity=5000, threads=1, device="cpu",
             watchdog_decisions=10000)
    cfg = dict(init=None, total_steps=240, n_steps=60, checkpoint_every=2, gamma=0.99, seed=0, rdqn=q)
    game = Game()
    game.keys = SimpleNamespace(release_all=lambda: None)
    env = BrawlEnv(game, game, Perception(game), window=0)
    saved = learn.train_rdqn(env, cfg, tmp_path / "run", [demo], realtime=False, reward_weights=(0.01, 1.0),
                             log=lambda m: None)
    import csv

    rows = list(csv.DictReader(open(tmp_path / "run" / "progress.csv", encoding="utf-8")))
    assert saved.name == "qnet.pt" and [r["steps"] for r in rows] == ["60", "120", "180", "240"]
    assert int(rows[-1]["updates"]) >= 0.5 * 240 - 1   # stepped: the actor waited for the learner
    assert (tmp_path / "run" / "pretrained.pt").exists() and len(list((tmp_path / "run" / "checkpoints").glob("*.pt"))) == 2
    assert env.calls["advance"] == 240
    assert rdqn.load(saved)[1]["n_in"] == observation.SIZE
