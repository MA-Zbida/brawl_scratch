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
                "gae_lambda": float, "clip_range": float, "ent_coef": float, "vf_coef": float, "max_grad_norm": float,
                "target_kl": lambda v: None if v is None else float(v)}
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
                **{key: cast(cfg[key]) for key, cast in PPO_SETTINGS.items() if key in cfg})
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
            warm = self.rollout < int(cfg.get("vf_warmup_rollouts") or 0)  # the update after this rollout: value only
            for p in (*self.model.policy.mlp_extractor.policy_net.parameters(), *self.model.policy.action_net.parameters()):
                p.requires_grad_(not warm)
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


RDQN_PROGRESS = ("rollout", "steps", "game_frames", "wall_s", "play_s", "stall_s", "stall_frames", "stocks_taken",
                 "stocks_lost", "damage_dealt", "damage_taken", "reward", "stall_stocks_lost", "stall_damage_taken",
                 "late_frames", "sense_ms", "over_budget", "pauses", "updates", "learner_lag", "weights_loaded", "fell",
                 "opponent_fell", "rollbacks", "beta")
TRACE_ROWS = 6000  # decisions per shard of a run's trace (5 minutes)
BEST_WINDOW = 30  # blocks (30 minutes at 1,200 decisions): the window over which the best policy is judged


def train_rdqn(env, cfg: dict, run_dir: Path, demo_roots: list, *, realtime: bool, reward_weights: tuple[float, float],
               stop=lambda: False, focused=lambda: True, now=None, sleep=None, log=print) -> Path:
    """The recurrent Q-learner (brawl/rdqn.py) on ``env``: this process plays, a second one learns.

    Without ``cfg["init"]`` the network is first pretrained on the demonstrations (on the GPU if there is
    one). Each decision: the network, its memory carried, picks the best action (a random one with
    probability epsilon); the decision goes to the learner in chunks of 64; every 100 decisions the
    newest weights are taken. In stepped mode the actor waits for the learner to keep
    ``updates_per_decision``; in real time it never waits. A gap (lost focus) cuts the stream: no return
    is bootstrapped across it and the memory starts afresh.

    Writes <run_dir>/progress.csv (one row per ``n_steps`` decisions), learner.csv, checkpoints/,
    trace/ (every decision: observation, action, reward, events; shards of TRACE_ROWS), best.pt (the
    weights playing during the best BEST_WINDOW blocks so far, by reward) and qnet.pt (qnet_stopped.pt
    if the run did not reach its end).
    """
    import csv
    import multiprocessing as mp
    import os
    import shutil
    import time

    import torch

    from brawl import observation, rdqn, vision

    now, sleep = now or time.perf_counter, sleep or time.sleep
    q = cfg["rdqn"]
    torch.set_num_threads(1)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    init = Path(cfg["init"]) if cfg.get("init") else None
    if init is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"pretraining on {len(demo_roots)} demonstration sessions ({device}) ...")
        learner = rdqn.Learner(observation.SIZE, q, float(cfg["gamma"]), device, seed=int(cfg["seed"]))
        rdqn.pretrain(learner, rdqn.demonstrations(demo_roots, env.k, *reward_weights, log=log), int(q["pretrain_updates"]), log)
        init = run_dir / "pretrained.pt"
        rdqn.save(learner.net, init, observation.SIZE, int(q["hidden"]))
        del learner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    net = rdqn.load(init)[0].eval()
    latest = run_dir / "latest.pt"
    shutil.copy(init, latest)
    ctx = mp.get_context("spawn")
    queue, decisions, updates, done = ctx.Queue(maxsize=400), ctx.Value("q", 0), ctx.Value("q", 0), ctx.Event()
    rollback = ctx.Value("i", 0)
    if q.get("explore", "uniform") == "human":  # how often the human took each action, over the demonstrations
        counts = np.ones(rdqn.N_ACTIONS)
        for r in demo_roots:
            a = np.load(Path(r) / f"decisions_k{env.k}.npz")["action"]
            counts += np.bincount(a[:, 0] * actions.SHAPE[1] + a[:, 1], minlength=rdqn.N_ACTIONS)
        explore_p = counts / counts.sum()
    else:
        explore_p = np.full(rdqn.N_ACTIONS, 1 / rdqn.N_ACTIONS)
    os.environ["BRAWL_NO_EARLY_CAPTURE"] = "1"  # the learner process must not take the screen
    learner = ctx.Process(target=rdqn.learner_main, daemon=True,
                          args=(q, float(cfg["gamma"]), [str(r) for r in demo_roots], env.k, reward_weights, str(run_dir),
                                str(init), queue, decisions, updates, done, int(cfg["seed"]), rollback))
    learner.start()
    rng = np.random.default_rng(int(cfg["seed"]))
    torch.manual_seed(int(cfg["seed"]))
    eps, per, block = float(q["epsilon"]), float(q["updates_per_decision"]), int(cfg["n_steps"])
    prior = load_policy(Path(q["prior"])) if q.get("prior") else None  # the clone the agent samples from
    beta_max, after, ramp, clip = (float(q.get(key) or 0) for key in ("beta", "beta_after", "beta_ramp", "adv_clip"))
    beta = 0.0
    file = open(run_dir / "progress.csv", "w", newline="", encoding="utf-8")
    rows = csv.DictWriter(file, fieldnames=RDQN_PROGRESS)
    rows.writeheader()
    budget_s, t0, stopped, unreadable, loaded = env.k / 60, now(), False, 0, 0
    seen = latest.stat().st_mtime
    buf: list = []
    cut_next = True

    def new() -> dict:
        return dict(stocks_taken=0, stocks_lost=0, damage_dealt=0.0, damage_taken=0.0, reward=0.0, late_frames=0,
                    game_frames=0, pauses=0, fell=0, opponent_fell=0)

    (run_dir / "trace").mkdir(exist_ok=True)
    trace: list = []

    keep_crops = bool(q.get("trace_crops"))  # both fighters' 96 px crops with every traced decision (vision data)

    def dump(final: bool = False) -> None:
        if trace and (final or len(trace) >= TRACE_ROWS):
            cols = list(zip(*trace))
            extra = {"crops": np.stack(cols[6]).astype(np.uint8)} if keep_crops else {}
            np.savez_compressed(run_dir / "trace" / f"shard_{int(cols[0][0]):07d}.npz", step=np.array(cols[0]), **extra,
                                obs=np.array(cols[1], np.float16), action=np.array(cols[2]), reward=np.array(cols[3], np.float32),
                                events=np.array(cols[4], np.float32), readable=np.array(cols[5]),
                                events_doc=np.array(["damage_dealt", "damage_taken", "ko_scored", "ko_lost", "fell",
                                                     "opponent_fell"]), features=np.array(observation.FEATURES))
            trace.clear()

    s, sense, started = new(), [], now()
    window: list = []  # (reward, game frames) of the last BEST_WINDOW blocks
    best = -float("inf")

    def send() -> None:
        if buf:
            obs, act, rew, cut = (np.array(x) for x in zip(*buf))
            try:
                queue.put((obs.astype(np.float16), act, rew.astype(np.float32), cut), timeout=5)
            except Exception:  # noqa: BLE001 - a learner that stopped reading: the decisions are lost, play goes on
                pass
            buf.clear()

    obs, _ = env.reset()
    h = None
    try:
        for step in range(1, int(cfg["total_steps"]) + 1):
            if stop():
                stopped = True
                break
            if not learner.is_alive():
                log("the learner process stopped: ending the run")
                stopped = True
                break
            if step % 20 == 0 and not focused():  # never type into another window
                env.idle()
                log("the game lost the keyboard: waiting for its window to have focus again")
                while not focused() and not stop():
                    sleep(1.0)
                obs, h, cut_next = env.resume()[0], None, True
                s["pauses"] += 1
            if step % 100 == 0 and latest.stat().st_mtime != seen:
                try:
                    net.load_state_dict(torch.load(latest, map_location="cpu", weights_only=False)["state"])
                    seen, loaded = latest.stat().st_mtime, loaded + 1
                except Exception:  # noqa: BLE001 - being replaced this instant: next time
                    pass
            with torch.no_grad():
                values, h = net(torch.as_tensor(obs, dtype=torch.float32)[None, None], h)
            if prior is not None:  # the clone's own odds, shifted towards what the values favour
                with torch.no_grad():
                    d = prior.get_distribution(torch.as_tensor(obs, dtype=torch.float32)[None]).distribution
                    logp = (d[0].logits[0][:, None] + d[1].logits[0][None, :]).reshape(-1)
                    advantage = (values[0, 0] - (logp.exp() * values[0, 0]).sum()).clamp(-clip, clip)
                    beta = beta_max * min(1.0, max(0.0, (step - after) / max(ramp, 1.0)))
                    j = int(torch.distributions.Categorical(logits=logp + beta * advantage).sample())
            elif rng.random() < eps:
                j = int(rng.choice(rdqn.N_ACTIONS, p=explore_p))
            else:
                j = int(values[0, 0].argmax())
            crops_now = (np.stack([vision.crop(env.frame, env.builder.last[who]) for who in (observation.AGENT, observation.OPPONENT)])
                    if keep_crops else None)  # what the decision saw, cut from the full-size frame
            nxt, reward, _, _, info = env.step(rdqn.split(j))
            before = obs
            buf.append((before, j, reward, cut_next))
            cut_next, obs = False, nxt
            decisions.value = step
            if len(buf) >= 64:
                send()
            e, timing = info["events"], info["timing"]
            s["stocks_taken"] += e.ko_scored
            s["stocks_lost"] += e.ko_lost
            s["damage_dealt"] += e.damage_dealt
            s["damage_taken"] += e.damage_taken
            s["fell"] += e.fell
            s["opponent_fell"] += e.opponent_fell
            s["reward"] += float(reward)
            s["late_frames"] += timing.late_frames
            s["game_frames"] += timing.frames
            sense.append(info["sense_s"])
            trace.append((step, before, j, reward, (e.damage_dealt, e.damage_taken, e.ko_scored, e.ko_lost, e.fell,
                                                         e.opponent_fell), info["sense"].readable, crops_now))
            dump()
            unreadable = 0 if info["sense"].readable else unreadable + 1
            if unreadable >= int(q["watchdog_decisions"]):
                log("the HUD has not been read for a minute: the game is not showing the fight; ending the run")
                stopped = True
                break
            if not realtime:  # the game waits: let the learner keep its pace
                while updates.value < per * step and learner.is_alive() and not stop():
                    sleep(0.002)
            if step % block == 0:
                n = step // block
                row = {"rollout": n, "steps": step, "wall_s": round(now() - t0, 1), "play_s": round(now() - started, 1),
                       "stall_s": 0.0, "stall_frames": 0, "stall_stocks_lost": 0, "stall_damage_taken": 0.0,
                       "sense_ms": round(float(np.median(sense)) * 1000, 1),
                       "over_budget": round(float(np.mean(np.array(sense) > budget_s)), 4), "updates": updates.value,
                       "learner_lag": round(updates.value / max(per * step, 1e-9), 3), "weights_loaded": loaded,
                       "rollbacks": rollback.value, "beta": round(beta, 3), **s}
                rows.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
                file.flush()
                minutes = max(s["game_frames"] / 3600, 1e-9)
                log(f"block {n} ({step} decisions): stocks {s['stocks_taken']} - {s['stocks_lost']} ({s['fell']} fell), damage "
                    f"{s['damage_dealt']:.0f} / {s['damage_taken']:.0f}, reward per minute {s['reward'] / minutes:.2f}; "
                    f"{updates.value} updates" + (f"; beta {beta:.2f}" if prior is not None else ""))
                if n % int(cfg["checkpoint_every"]) == 0:
                    shutil.copy(latest, run_dir / "checkpoints" / f"qnet_{step:07d}.pt")
                window = (window + [(s["reward"], s["game_frames"])])[-BEST_WINDOW:]
                if len(window) == BEST_WINDOW:
                    rate = sum(r for r, _ in window) / max(sum(f for _, f in window) / 3600, 1e-9)
                    if rate > best:  # the weights playing now were good over the last half hour
                        best = rate
                        shutil.copy(latest, run_dir / "best.pt")
                        log(f"best so far: {rate:.2f} reward per minute over the last {BEST_WINDOW} blocks -> best.pt")
                    elif rate < best - float(q.get("rollback") or float("inf")):
                        rollback.value += 1
                        window = []  # judge the restored weights on a fresh half hour
                        log(f"the last {BEST_WINDOW} blocks ({rate:.2f}) fell {best - rate:.2f} below the best ({best:.2f}): "
                            f"back to best.pt (rollback {rollback.value})")
                s, sense, started = new(), [], now()
        send()
    finally:
        dump(final=True)
        env.idle()
        done.set()
        queue.cancel_join_thread()  # decisions not yet read are dropped rather than block the exit
        learner.join(timeout=120)
        if learner.is_alive():
            learner.terminate()
        file.close()
    final = run_dir / ("qnet_stopped.pt" if stopped else "qnet.pt")
    shutil.copy(latest, final)
    return final
