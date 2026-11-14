"""Pretrain the recurrent Q-network on the human demonstrations, check it, and save it for the live runs.

    python scripts/pretrain_rdqn.py                    # -> runs/rdqn/pretrain_all/qnet.pt
    python scripts/pretrain_rdqn.py --config configs/experiments/rdqn_medium_v2.yaml --name v2

First a check: the network is pretrained on two thirds of the sessions and plays the held-out third
greedily, memory carried through each session; it is scored like behaviour cloning (actions right,
right where the human changes action, button presses) and checked for collapse (how many different
actions it uses). Then it is pretrained on all sessions and saved. The live runs start from that file
(configs/experiments/rdqn_medium.yaml: train.init). Writes results/rdqn/pretrain_<name>.md.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from brawl import actions, observation, rdqn  # noqa: E402
from brawl.config import load_experiment, resolve_config  # noqa: E402
from brawl.paths import CONFIGS, RESULTS, RUNS  # noqa: E402
from train import demo_sessions  # noqa: E402


def play_back(net, root: Path, k: int, device: str, shuffle_history: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """The network's greedy action at every decision of a session, and the human's (joint indices). With
    ``shuffle_history`` the action history in each observation comes from a random other decision, and the
    human action returned is that decision's last one: how often the network just repeats its history."""
    import torch

    z = np.load(root / f"decisions_k{k}.npz")
    human = z["action"][:, 0] * actions.SHAPE[1] + z["action"][:, 1]
    obs = z["obs"]
    if shuffle_history:
        perm = np.random.default_rng(0).permutation(len(human))
        obs = obs.copy()
        obs[:, rdqn.HISTORY] = obs[perm][:, rdqn.HISTORY]
        human = np.concatenate([[0], human[:-1]])[perm]
    net.eval()
    out, h = [], None
    with torch.no_grad():
        for a in range(0, len(human), 2048):
            q, h = net(torch.as_tensor(obs[a:a + 2048], dtype=torch.float32, device=device)[None], h)
            out.append(q[0].argmax(-1).cpu().numpy())
    return np.concatenate(out), human


def score(pairs: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    agent = np.concatenate([a for a, _ in pairs])
    human = np.concatenate([h for _, h in pairs])
    previous = np.concatenate([np.concatenate([[0], h[:-1]]) for _, h in pairs])
    changed = human != previous
    button = lambda j: j % actions.SHAPE[1]  # noqa: E731
    pressed = (button(human) != 0) & (button(human) != button(previous))
    return {"both": float((agent == human).mean()), "repeat_previous": float((previous == human).mean()),
            "changed": float((agent == human)[changed].mean()),
            "new_press": float((button(agent) == button(human))[pressed].mean()),
            "buttons_human": float((button(human) != 0).mean()), "buttons_agent": float((button(agent) != 0).mean()),
            "actions_used": int(len(np.unique(agent))), "top_action": float(np.bincount(agent).max() / len(agent))}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="all")
    p.add_argument("--updates", type=int, default=None, help="default: train.rdqn.pretrain_updates in configs/base.yaml")
    p.add_argument("--no-check", action="store_true", help="skip the held-out check")
    p.add_argument("--config", type=Path, default=None, help="take the settings from this experiment (default: base.yaml)")
    args = p.parse_args(argv)

    import torch

    base = load_experiment(args.config).resolve(load_experiment(args.config).variants()[0][0], 0) if args.config \
        else resolve_config(CONFIGS / "base.yaml")
    cfg, env = base["train"]["rdqn"], base["env"]
    k, gamma, weights = int(env["k"]), float(base["train"]["gamma"]), (float(env["damage_weight"]), float(env["ko_weight"]))
    updates = args.updates or int(cfg["pretrain_updates"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    roots = demo_sessions(cfg["demo_levels"], k)
    print(f"{len(roots)} demonstration sessions; {updates} updates on {device}", flush=True)
    lines = [f"# Pretrained recurrent Q-network: {args.name}", "",
             f"{datetime.now():%Y-%m-%d %H:%M}. {len(roots)} human sessions ({', '.join(cfg['demo_levels'])}), {updates} updates "
             f"of batch {cfg['batch']} sequences, margin {cfg['margin']}, n-step {cfg['n_step']}, gamma {gamma}, target every "
             f"{cfg['target_every']}, learning rate {cfg['learning_rate']}"
             f"{' (' + str(args.config) + ')' if args.config else ''}. scripts/pretrain_rdqn.py.", ""]
    if not args.no_check:
        test = roots[::3]
        train = [r for r in roots if r not in test]
        learner = rdqn.Learner(observation.SIZE, cfg, gamma, device, seed=0)
        rdqn.pretrain(learner, rdqn.demonstrations(train, k, *weights), updates, lambda m: print(m, flush=True))
        s = score([play_back(learner.net, r, k, device) for r in test])
        copy = np.mean([np.mean(np.equal(*play_back(learner.net, r, k, device, shuffle_history=True))) for r in test])
        lines += [f"Held-out check: trained on {len(train)} sessions, played back greedily on {len(test)} others "
                  f"({', '.join(r.name for r in test)}), memory carried through each session.", "",
                  "| Measure | Value |", "|---|---|",
                  f"| Action right (direction and button) | {s['both']:.3f} |",
                  f"| Baseline: repeat the human's previous action | {s['repeat_previous']:.3f} |",
                  f"| Right where the human changes action | {s['changed']:.3f} |",
                  f"| Human's new button presses matched | {s['new_press']:.3f} |",
                  f"| Decisions with a button: human / network | {s['buttons_human']:.3f} / {s['buttons_agent']:.3f} |",
                  f"| Different actions used by the network (of 54) | {s['actions_used']} |",
                  f"| Share of its most used action | {s['top_action']:.3f} |",
                  f"| Repeats an action history taken from elsewhere (copying itself; 2026-11-13 network: 0.88-0.90) | "
                  f"{copy:.3f} |", "",
                  "For scale, behaviour cloning (medium_v4) is right at 0.224 where the human changes action, on held-out "
                  "stretches.", ""]
        print("\n".join(lines[-15:]), flush=True)
        del learner
    learner = rdqn.Learner(observation.SIZE, cfg, gamma, device, seed=0)
    rdqn.pretrain(learner, rdqn.demonstrations(roots, k, *weights), updates, lambda m: print(m, flush=True))
    out = RUNS / "rdqn" / f"pretrain_{args.name}"
    out.mkdir(parents=True, exist_ok=True)
    rdqn.save(learner.net, out / "qnet.pt", observation.SIZE, int(cfg["hidden"]))
    lines.append(f"Trained on all {len(roots)} sessions -> {(out / 'qnet.pt').relative_to(RUNS.parent)}")
    path = RESULTS / "rdqn" / f"pretrain_{args.name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {out / 'qnet.pt'}\n-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
