"""Learning from demonstrations (research/DESIGN.md section 8): the data, the policy, behaviour cloning.

The policy is Stable-Baselines3's actor-critic MLP over the observation vector with a
MultiDiscrete([9, 6]) head, so the cloned weights are the starting point of any SB3 learner. It
runs on the CPU when it plays (a small MLP is faster there than a round trip to the GPU, which
the detector keeps for itself).

**Winning stretches.** A recording is cut at every KO. A stretch that ends with the opponent's KO
is kept; one that ends with the agent's own KO is dropped, and so is the unfinished stretch at the
end. The teacher the policy copies is then the human while winning the stock.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from brawl import actions, observation


@dataclass
class Demos:
    obs: np.ndarray        # [N, SIZE]
    action: np.ndarray     # [N, 2]
    previous: np.ndarray   # [N, 2] the action before each one (the first of a stretch: none)
    stretch: np.ndarray    # [N] index of the stretch each decision belongs to
    summary: list[str]


def stretches(ko_scored: np.ndarray, ko_lost: np.ndarray) -> list[tuple[int, int, str]]:
    """(start, end, outcome) for each stretch between KOs; ``end`` is exclusive and includes the
    decision the KO happened in. Outcome: ``won``, ``lost`` or ``open`` (the recording ended)."""
    out, start = [], 0
    for i in np.flatnonzero((ko_scored > 0) | (ko_lost > 0)):
        out.append((start, int(i) + 1, "lost" if ko_lost[i] > 0 else "won"))
        start = int(i) + 1
    if start < len(ko_scored):
        out.append((start, len(ko_scored), "open"))
    return out


def load(recordings: list[Path], k: int = 3, keep: tuple[str, ...] = ("won",)) -> Demos:
    obs, action, previous, stretch, summary, count = [], [], [], [], [], 0
    for root in recordings:
        path = root / f"decisions_k{k}.npz"
        if not path.exists():
            summary.append(f"{root.name}: not prepared (scripts/prepare_demos.py)")
            continue
        z = np.load(path)
        if tuple(z["features"]) != observation.FEATURES:
            summary.append(f"{root.name}: prepared with another feature list; prepare it again with --force")
            continue
        parts = stretches(z["ko_scored"], z["ko_lost"])
        kept = [(a, b) for a, b, outcome in parts if outcome in keep]
        for a, b in kept:
            obs.append(z["obs"][a:b])
            action.append(z["action"][a:b])
            previous.append(np.vstack([np.zeros((1, 2), np.int64), z["action"][a:b - 1]]))
            stretch.append(np.full(b - a, count))
            count += 1
        n = sum(b - a for a, b in kept)
        outcomes = [o for *_, o in parts]
        summary.append(f"{root.name}: {outcomes.count('won')} stocks won, {outcomes.count('lost')} lost; "
                       f"kept {n} of {len(z['obs'])} decisions ({n * k / 3600:.1f} of {len(z['obs']) * k / 3600:.1f} min)")
    if not obs:
        raise SystemExit("no decisions to learn from:\n" + "\n".join(summary))
    return Demos(np.concatenate(obs), np.concatenate(action), np.concatenate(previous), np.concatenate(stretch), summary)


def make_policy(net: tuple[int, ...] = (256, 256), device: str = "cpu"):
    import gymnasium as gym
    from stable_baselines3.common.policies import ActorCriticPolicy

    space = gym.spaces.Box(-5.0, 5.0, (observation.SIZE,), np.float32)
    return ActorCriticPolicy(space, gym.spaces.MultiDiscrete(list(actions.SHAPE)), lambda _: 3e-4,
                             net_arch=dict(pi=list(net), vf=list(net))).to(device)


def load_policy(path: Path, device: str = "cpu"):
    from stable_baselines3.common.policies import ActorCriticPolicy

    return ActorCriticPolicy.load(str(path), device=device)


def scores(policy, obs: np.ndarray, action: np.ndarray, previous: np.ndarray, batch: int = 8192) -> dict:
    """How well the policy predicts the human: loss, and accuracy of its most likely action. The
    action history is part of the observation, so repeating the previous action is already right
    most of the time; ``changed`` scores only the decisions where the human did something new."""
    import torch

    logp, best = [], []
    with torch.no_grad():
        for i in range(0, len(obs), batch):
            o = torch.as_tensor(obs[i:i + batch], device=policy.device)
            a = torch.as_tensor(action[i:i + batch], device=policy.device)
            logp.append(policy.evaluate_actions(o, a)[1].cpu().numpy())
            best.append(policy._predict(o, deterministic=True).cpu().numpy())
    best = np.concatenate(best).astype(np.int64)
    right = best == action
    changed = (action != previous).any(1)
    pressed = (action[:, 1] != 0) & (action[:, 1] != previous[:, 1])
    return {"loss": float(-np.concatenate(logp).mean()), "direction": float(right[:, 0].mean()),
            "button": float(right[:, 1].mean()), "both": float(right.all(1).mean()),
            "repeat_previous": float((previous == action).all(1).mean()),
            "changed": float(right.all(1)[changed].mean()) if changed.any() else float("nan"),
            "new_press_recall": float(right[:, 1][pressed].mean()) if pressed.any() else float("nan"),
            "press_rate_human": float((action[:, 1] != 0).mean()), "press_rate_policy": float((best[:, 1] != 0).mean())}


def clone(demos: Demos, out: Path, *, epochs: int = 40, batch: int = 512, lr: float = 3e-4, valid_share: float = 0.15,
          net: tuple[int, ...] = (256, 256), seed: int = 0, device: str = "cpu", log=print) -> dict:
    """Behaviour cloning by maximum likelihood. Whole stretches are held out for validation; the
    weights with the lowest validation loss are saved to ``out``/policy.zip."""
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ids = rng.permutation(demos.stretch.max() + 1)
    held = set(ids[:max(1, round(len(ids) * valid_share))].tolist())
    valid = np.isin(demos.stretch, list(held))
    train = np.flatnonzero(~valid)
    policy = make_policy(net, device)
    optim = torch.optim.Adam(policy.parameters(), lr=lr)
    obs, action = torch.as_tensor(demos.obs, device=device), torch.as_tensor(demos.action, device=device)
    out.mkdir(parents=True, exist_ok=True)
    best, history = None, []
    for epoch in range(1, epochs + 1):
        policy.train()
        for i in np.array_split(rng.permutation(train), max(1, len(train) // batch)):
            i = torch.as_tensor(i, device=device)
            loss = -policy.evaluate_actions(obs[i], action[i])[1].mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
        policy.eval()
        v = scores(policy, demos.obs[valid], demos.action[valid], demos.previous[valid])
        history.append(v["loss"])
        if best is None or v["loss"] < best["loss"]:
            best = {**v, "epoch": epoch}
            policy.save(str(out / "policy.zip"))
        log(f"epoch {epoch:3d}: validation loss {v['loss']:.4f}, both right {v['both']:.3f}, when the human changes "
            f"{v['changed']:.3f}, new presses {v['new_press_recall']:.3f}")
        if epoch - best["epoch"] >= 8:
            break
    policy = load_policy(out / "policy.zip", device)
    return {"train_decisions": int(len(train)), "valid_decisions": int(valid.sum()), "stretches": int(len(ids)),
            "valid": best, "train": scores(policy, demos.obs[train], demos.action[train], demos.previous[train]),
            "history": history}


# -- reinforcement learning in the live game -------------------------------------------------------
PPO_SETTINGS = {"learning_rate": float, "n_steps": int, "batch_size": int, "n_epochs": int, "gamma": float,
                "gae_lambda": float, "clip_range": float, "ent_coef": float, "vf_coef": float, "max_grad_norm": float}
PROGRESS = ("rollout", "steps", "game_frames", "wall_s", "play_s", "stall_s", "stall_frames", "stocks_taken",
            "stocks_lost", "damage_dealt", "damage_taken", "reward", "stall_stocks_lost", "stall_damage_taken",
            "late_frames", "sense_ms", "over_budget", "pauses",
            "approx_kl", "clip_fraction", "entropy", "value_loss", "explained_variance")  # of the update before the rollout
TRAIN_METRICS = {"approx_kl": "train/approx_kl", "clip_fraction": "train/clip_fraction", "entropy": "train/entropy_loss",
                 "value_loss": "train/value_loss", "explained_variance": "train/explained_variance"}


def train_ppo(env, cfg: dict, run_dir: Path, *, realtime: bool, stop=lambda: False, focused=lambda: True,
              now=None, sleep=None, log=print) -> Path:
    """Stable-Baselines3 PPO on ``env`` (brawl.brawlenv.BrawlEnv), from ``cfg["init"]`` when given.

    The task never ends, so a rollout is ``n_steps`` decisions wherever the game is. In real time
    the game goes on during the update: the keys are released before it, and afterwards the game is
    read again (``env.resume``); what happened in between is logged as the cost of the stall and is
    not the reward of any action. Every ``checkpoint_every`` rollouts the policy is saved in the
    format scripts/play.py loads. ``stop()`` ends the run cleanly; while ``focused()`` is false
    (another window has the keyboard) nothing is pressed and training waits.

    Writes <run_dir>/progress.csv (one row per rollout), checkpoints/ and policy.zip
    (policy_stopped.zip if the run was stopped before its end).
    """
    import csv
    import time

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    now, sleep = now or time.perf_counter, sleep or time.sleep
    torch.set_num_threads(1)
    net = [int(n) for n in cfg["net"]]
    model = PPO("MlpPolicy", env, policy_kwargs=dict(net_arch=dict(pi=net, vf=net)), seed=int(cfg["seed"]), device="cpu",
                **{key: cast(cfg[key]) for key, cast in PPO_SETTINGS.items()})
    if cfg.get("init"):
        model.policy.load_state_dict(load_policy(Path(cfg["init"])).state_dict())
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    budget_s = env.k / 60

    class Live(BaseCallback):
        def _init_callback(self) -> None:
            self.file = open(run_dir / "progress.csv", "w", newline="", encoding="utf-8")
            self.rows = csv.DictWriter(self.file, fieldnames=PROGRESS)
            self.rows.writeheader()
            self.t0, self.ended, self.rollout, self.stopped = now(), None, 0, False
            self.stall = dict(stall_s=0.0, stall_frames=0, stall_stocks_lost=0, stall_damage_taken=0.0)

        def _on_rollout_start(self) -> None:
            seen = self.model.logger.name_to_value  # what the update before this rollout recorded
            self.update = {k: (-seen[v] if k == "entropy" else seen[v]) if v in seen else float("nan")
                           for k, v in TRAIN_METRICS.items()}
            if realtime and self.ended is not None:  # the game went on during the update
                obs, events, skipped = env.resume()
                self.model._last_obs = obs[None].copy()
                self.stall = dict(stall_s=now() - self.ended, stall_frames=skipped, stall_stocks_lost=events.ko_lost,
                                  stall_damage_taken=events.damage_taken)
            self.started, self.sense = now(), []
            self.sum = dict(stocks_taken=0, stocks_lost=0, damage_dealt=0.0, damage_taken=0.0, reward=0.0,
                            late_frames=0, game_frames=0, pauses=0)

        def _on_step(self) -> bool:
            info, s = self.locals["infos"][0], self.sum
            e, timing = info["events"], info["timing"]
            s["stocks_taken"] += e.ko_scored
            s["stocks_lost"] += e.ko_lost
            s["damage_dealt"] += e.damage_dealt
            s["damage_taken"] += e.damage_taken
            s["reward"] += float(self.locals["rewards"][0])
            s["late_frames"] += timing.late_frames
            s["game_frames"] += timing.frames
            self.sense.append(info["sense_s"])
            if stop():
                self.stopped = True
                return False
            if self.n_calls % 20 == 0 and not focused():  # never type into another window
                env.idle()
                log("the game lost the keyboard: waiting for its window to have focus again")
                while not focused() and not stop():
                    sleep(1.0)
                self.locals["new_obs"][0] = env.resume()[0]  # in place: learning continues from the game as it is now
                s["pauses"] += 1
            return True

        def _on_rollout_end(self) -> None:
            if realtime:
                env.idle()
            self.rollout += 1
            s, sense = self.sum, np.array(self.sense)
            row = {"rollout": self.rollout, "steps": self.num_timesteps, "wall_s": round(now() - self.t0, 1),
                   "play_s": round(now() - self.started, 1), "sense_ms": round(float(np.median(sense)) * 1000, 1),
                   "over_budget": round(float(np.mean(sense > budget_s)), 4), **s, **self.stall,
                   **{k: float(v) for k, v in self.update.items()}}
            self.rows.writerow({key: (round(v, 4) if isinstance(v, float) else v) for key, v in row.items()})
            self.file.flush()
            minutes = max(s["game_frames"] / 3600, 1e-9)
            log(f"rollout {self.rollout} ({self.num_timesteps} steps): stocks {s['stocks_taken']} - {s['stocks_lost']}, "
                f"damage {s['damage_dealt']:.0f} / {s['damage_taken']:.0f}, reward per minute {s['reward'] / minutes:.2f}; "
                f"the update before it took {self.stall['stall_s']:.2f} s")
            if self.rollout % int(cfg["checkpoint_every"]) == 0:
                self.model.policy.save(str(run_dir / "checkpoints" / f"policy_{self.num_timesteps:07d}.zip"))
            self.ended = now()

        def _on_training_end(self) -> None:
            self.file.close()

    live = Live()
    try:
        model.learn(total_timesteps=int(cfg["total_steps"]), callback=live)
    finally:
        env.idle()
    name = "policy_stopped.zip" if live.stopped else "policy.zip"  # only a run that reached its end is finished
    model.policy.save(str(run_dir / name))
    return run_dir / name
