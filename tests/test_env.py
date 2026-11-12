"""brawl.brawlenv and brawl.learn against a drawn game: one step runs each part once; cloning learns a rule."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from brawl import actions, learn, observation
from brawl.brawlenv import BrawlEnv
from brawl.game.clock import Timing
from brawl.game.keys import HEAVY, RIGHT
from brawl.sense.perception import Perception
from brawl.sense.truth import Box
from test_hud import draw


class Game:
    """A clock, a screen and a detector in one: Wu Shang moves right 2 px per frame that RIGHT is
    held, and Mordex takes 10 damage in every decision that HEAVY is held."""

    def __init__(self):
        self.frame, self.x, self.damage, self.plans = 0, 800.0, 0.0, []

    def advance(self, plan):
        self.plans.append(list(plan))
        for keys in plan:
            self.x += 2 * (RIGHT in keys)
            self.frame += 1
        self.damage += 10 * (HEAVY in plan[0])
        return Timing(frames=len(plan))

    def grab(self, fresh=True):
        ring = (255, 255, round(255 - self.damage * 255 / 50))
        return SimpleNamespace(image=draw(("mordex", ring, None), ("wu_shang", (255, 255, 255), None)), fresh=True)

    def __call__(self, image):
        return [Box("wu_shang_spear", self.x - 30, 542, self.x + 30, 622, 0.9, point=(self.x, 582)),
                Box("mordex_unarmed", 1070, 542, 1130, 622, 0.9, point=(1100, 582))]

    def stop(self):
        self.stopped = True


def test_one_step_plays_the_keys_perceives_once_and_rewards_from_the_hud():
    game = Game()
    env = BrawlEnv(game, game, Perception(game), window=2)
    obs, _ = env.reset()
    assert obs.shape == (observation.SIZE,) and env.observation_space.contains(obs)
    right_heavy = (actions.DIRECTION_NAMES.index("right"), actions.BUTTON_NAMES.index("heavy"))
    obs, reward, terminated, truncated, info = env.step(right_heavy)
    assert game.plans == [[{RIGHT, HEAVY}] * 3] and info["frame"] == 3
    assert reward == pytest.approx(0.1, abs=0.01) and not terminated and not truncated
    assert obs[observation.FEATURES.index("agent_vx")] == pytest.approx(6 / observation.V)
    assert obs[observation.FEATURES.index("action_4_btn_heavy")] == 1
    *_, truncated, _ = env.step((0, 0))
    assert truncated  # the window ended; the game goes on
    assert env.calls == {"advance": 2, "grab": 3, "perceive": 3, "build": 3}  # once per step, and once at reset
    env.close()
    assert game.stopped


def test_winning_stretches_are_kept_and_the_rest_dropped(tmp_path):
    n = 12
    scored, lost = np.zeros(n), np.zeros(n)
    scored[3], lost[7] = 1, 1  # a stock won, then one lost, then the recording ends
    assert learn.stretches(scored, lost) == [(0, 4, "won"), (4, 8, "lost"), (8, 12, "open")]
    np.savez(tmp_path / "decisions_k3.npz", obs=np.zeros((n, observation.SIZE), np.float32),
             action=np.tile([[2, 1]], (n, 1)), ko_scored=scored, ko_lost=lost, features=np.array(observation.FEATURES))
    demos = learn.load([tmp_path])
    assert len(demos.obs) == 4 and demos.previous[0].tolist() == [0, 0] and demos.previous[1].tolist() == [2, 1]
    assert len(learn.load([tmp_path], keep=("won", "lost", "open")).obs) == n


def test_cloning_learns_what_the_demonstrator_does(tmp_path):
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(3000, observation.SIZE)).astype(np.float32)
    action = np.stack([np.where(obs[:, 0] > 0, 2, 1), np.where(obs[:, 1] > 0.5, 3, 0)], 1)  # a rule on two features
    demos = learn.Demos(obs, action, np.zeros_like(action), np.repeat(np.arange(30), 100), [])
    r = learn.clone(demos, tmp_path, epochs=40, batch=64, lr=1e-3, net=(64, 64), log=lambda _: None)
    assert r["valid"]["both"] > 0.8 and (tmp_path / "policy.zip").exists()
    policy = learn.load_policy(tmp_path / "policy.zip")
    import torch

    picked = policy._predict(torch.as_tensor(obs[:200]), deterministic=True).numpy()
    assert (picked == action[:200]).all(1).mean() > 0.8  # chance is about 0.35


def test_play_reports_blocks_and_virtual_matches():
    import importlib.util

    from brawl.ledger import Events
    from brawl.paths import REPO_ROOT

    spec = importlib.util.spec_from_file_location("play", REPO_ROOT / "scripts" / "play.py")
    play = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(play)

    class Env:  # one second per step; stocks taken at steps 2, 5, 8 and 25, lost at 12, 13 and 20
        def __init__(self):
            self.t, self.calls, self.steps = 0.0, {}, 0

        def reset(self):
            return np.zeros(3), {}

        def step(self, action):
            e = Events(damage_dealt=1.0, ko_scored=int(self.steps in (2, 5, 8, 25)), ko_lost=int(self.steps in (12, 13, 20)))
            self.steps += 1
            self.t += 1.0
            info = {"events": e, "timing": Timing(frames=3, late_frames=1), "sense_s": 0.02,
                    "sense": SimpleNamespace(fighters={"wu_shang": None})}
            return np.zeros(3), 0.01 + e.ko_scored - e.ko_lost, False, False, info

    env = Env()
    r = play.play(env, lambda obs: (0, 2), block_s=10, blocks=3, now=lambda: env.t, log=lambda _: None)
    assert [(b["scored"], b["lost"], b["decisions"]) for b in r["blocks"]] == [(3, 0, 10), (0, 2, 10), (1, 1, 10)]
    assert (r["matches_won"], r["matches_lost"]) == (1, 1) and r["buttons"] == 30 and r["late_frames"] == 30
    text = "\n".join(play.report(r, "test", "A test.", 3, env.calls, env.steps))
    assert "| Stocks taken - lost | 4 - 3 |" in text and "won - lost | 1 - 1 (50%)" in text
    assert "| 2 | 0 - 2 |" in text and "Over the 3 full blocks" in text
    stopped = play.play(Env(), lambda obs: (0, 0), block_s=10, blocks=3, stop=lambda: True, log=lambda _: None)
    assert stopped["blocks"][0]["decisions"] == 0
