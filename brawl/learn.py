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
