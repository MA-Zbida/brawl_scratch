"""Does a fighter's crop show what the fighter is doing? A direct test, with labels from the key log.

    python scripts/probe_state.py

We know when Wu Shang attacks, dodges and jumps: the human's presses are recorded. So his state at
each decision is labelled for free: a light or heavy pressed in the last 4 decisions (0.2 s), a
dodge in the last 4, a jump in the last 3. Three inputs are trained to read those three states on
held-out sessions (3 folds):

- ``kinematics``: Wu Shang's position, velocity, held weapon and place on the stage (no pictures,
  no action history, no timers: those would give the answer away);
- ``crop``: his 96 px crop alone, one frame;
- ``crop + kinematics``.

Then the crop model, trained on Wu Shang only, is applied to Mordex's crops: both legends share
the unarmed and gauntlet moves. Its "attacking" output for Mordex is scored as a predictor of
Wu Shang taking damage in the next 0.5 s, beside the vector's own prediction from
scripts/probe_vision.py, to see whether it is a signal the vector lacks.

Writes results/probes/state.md and <recording>/state_k<k>.npz (the three outputs for both fighters
at every decision, from the model that never saw that session), for use as observation entries.
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

import probe_vision as pv  # noqa: E402
from brawl import actions, observation  # noqa: E402
from brawl.paths import DATA, RESULTS  # noqa: E402
from brawl.record import Recording  # noqa: E402

F = observation.FEATURES
STATES = ("attacking", "dodging", "jumping")
KINEMATICS = [F.index(f"agent_{n}") for n in ("x", "y", "vx", "vy", "unarmed", "weapon_1", "weapon_2", "grounded",
                                              "offstage", "below", "to_left_ledge", "to_right_ledge")]
INPUTS = {"kinematics": (False, True), "crop": (True, False), "crop + kinematics": (True, True)}  # (crop, kinematics)


def labels(action: np.ndarray) -> np.ndarray:
    """[N, 3]: was a light or heavy, a dodge, a jump pressed in the decisions just before each one?"""
    button = action[:, 1]
    onset = lambda names: np.isin(button, [actions.BUTTON_NAMES.index(n) for n in names]) & (  # noqa: E731
        button != np.concatenate([[0], button[:-1]]))
    recent = lambda press, back: np.array([press[max(0, i - back):i].any() for i in range(len(press))])  # noqa: E731
    return np.stack([recent(onset(("light", "heavy")), 4), recent(onset(("dodge",)), 4), recent(onset(("jump",)), 3)],
                    1).astype(np.float32)


def auc(p: np.ndarray, y: np.ndarray) -> float:
    ranks = np.empty(len(p))
    ranks[np.argsort(p)] = np.arange(1, len(p) + 1)
    return float((ranks[y > 0].sum() - y.sum() * (y.sum() + 1) / 2) / max(y.sum() * (len(y) - y.sum()), 1))


def build(crop: bool, kinematics: bool, device: str):
    import torch
    from torch import nn

    class Reader(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(nn.Conv2d(1, 32, 8, 4), nn.ReLU(), nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
                                     nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten(), nn.LazyLinear(256), nn.ReLU()) if crop else None
            self.mlp = nn.Sequential(nn.Linear(len(KINEMATICS), 128), nn.ReLU()) if kinematics else None
            self.head = nn.Sequential(nn.Linear(256 * crop + 128 * kinematics, 128), nn.ReLU(), nn.Linear(128, len(STATES)))

        def forward(self, img, vec):
            parts = ([self.cnn(img)] if self.cnn is not None else []) + ([self.mlp(vec)] if self.mlp is not None else [])
            return self.head(torch.cat(parts, 1))

    model = Reader().to(device)
    with torch.no_grad():
        model(torch.zeros(1, 1, 96, 96, device=device), torch.zeros(1, len(KINEMATICS), device=device))
    return model


def shifted(x):
    """Random shift of up to 4 px, one per picture."""
    import torch
    import torch.nn.functional as fn

    theta = torch.zeros(len(x), 2, 3, device=x.device)
    theta[:, 0, 0] = theta[:, 1, 1] = 1
    theta[:, :, 2] = (torch.rand(len(x), 2, device=x.device) * 2 - 1) * (8 / 96)
    return fn.grid_sample(x, fn.affine_grid(theta, x.shape, align_corners=False), padding_mode="zeros", align_corners=False)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--difficulty", default="medium")
    p.add_argument("--epochs", type=int, default=40, help="at most; each input stops when its held-back part does")
    p.add_argument("--k", type=int, default=3)
    args = p.parse_args(argv)

    import torch
    import torch.nn.functional as fn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    roots = [q for q in sorted((DATA / "demos").iterdir()) if (q / f"crops_k{args.k}_96.npy").exists()
             and Recording(q).meta["setting"].get("difficulty") in set(args.difficulty.split(","))]
    sessions = [pv.Session(r, args.k, 96) for r in roots]
    n = [len(s.obs) for s in sessions]
    y = [labels(s.action) for s in sessions]
    here = [s.obs[:, F.index("agent_present")] > 0 for s in sessions]          # Wu Shang is on the screen
    there = [s.obs[:, F.index("opponent_present")] > 0 for s in sessions]
    kin = [torch.as_tensor(s.obs[:, KINEMATICS], device=device) for s in sessions]
    wu = [torch.as_tensor(np.ascontiguousarray(s.crops[:, 0]), device=device) for s in sessions]  # 1.2 GB for 110 min
    folds = [[i for i in range(len(roots)) if i % 3 == f] for f in range(3)]
    rates = np.concatenate([a[h] for a, h in zip(y, here)]).mean(0)
    print(f"{len(roots)} sessions, {sum(n)} decisions; Wu Shang is {', '.join(f'{s} in {r:.1%}' for s, r in zip(STATES, rates))}",
          flush=True)

    def predict(model, i, fighter=0):
        model.eval()
        out = []
        with torch.no_grad():
            for a in range(0, n[i], 2048):
                img = wu[i][a:a + 2048] if fighter == 0 else torch.as_tensor(
                    np.ascontiguousarray(sessions[i].crops[a:a + 2048, fighter]), device=device)
                out.append(torch.sigmoid(model(img[:, None].float() / 255, kin[i][a:a + 2048])).cpu().numpy())
        return np.concatenate(out)

    scores: dict[str, list] = {name: [] for name in INPUTS}
    held = {i: None for i in range(len(roots))}  # the crop model's outputs for sessions it never saw: (Wu Shang, Mordex)
    t0 = time.perf_counter()
    for name, (crop, kinematics) in INPUTS.items():
        for f, test in enumerate(folds):
            torch.manual_seed(f)
            rng = np.random.default_rng(f)
            train = [i for i in range(len(roots)) if i not in test]
            pool = [(i, j) for i in train for j in np.flatnonzero(here[i][:int(n[i] * 0.88)])]
            model = build(crop, kinematics, device)
            optim = torch.optim.Adam(model.parameters(), lr=5e-4)
            best, state, since = None, None, 0
            for epoch in range(args.epochs):
                model.train()
                order = rng.permutation(len(pool))
                for b in range(0, len(order), 512):
                    idx = [pool[j] for j in order[b:b + 512]]
                    img = shifted(torch.stack([wu[i][j] for i, j in idx])[:, None].float() / 255) if crop else None
                    vec = torch.stack([kin[i][j] for i, j in idx])
                    target = torch.as_tensor(np.stack([y[i][j] for i, j in idx]), device=device)
                    loss = fn.binary_cross_entropy_with_logits(model(img if crop else torch.zeros(1), vec), target)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                stop = np.mean([np.mean([auc(predict(model, i)[int(n[i] * 0.88):, s][here[i][int(n[i] * 0.88):]],
                                             y[i][int(n[i] * 0.88):, s][here[i][int(n[i] * 0.88):]]) for s in range(3)])
                                for i in train])
                if best is None or stop > best:
                    best, since, state = stop, 0, {k: t.detach().clone() for k, t in model.state_dict().items()}
                else:
                    since += 1
                    if since >= 4:
                        break
            model.load_state_dict(state)
            p = {i: predict(model, i) for i in test}
            row = [auc(np.concatenate([p[i][here[i], s] for i in test]), np.concatenate([y[i][here[i], s] for i in test]))
                   for s in range(3)]
            scores[name].append(row)
            print(f"{name}, fold {f}: " + ", ".join(f"{s} AUC {v:.3f}" for s, v in zip(STATES, row))
                  + f" ({epoch + 1} epochs, {time.perf_counter() - t0:.0f} s)", flush=True)
            if name == "crop":
                for i in test:
                    held[i] = (p[i], predict(model, i, fighter=1))
    # Mordex, read by the model trained on Wu Shang: does "attacking" come before Wu Shang's damage?
    hit = np.concatenate([s.hit[s.hit_ok & t] for s, t in zip(sessions, there)])
    mordex = np.concatenate([held[i][1][s.hit_ok & t] for i, (s, t) in enumerate(zip(sessions, there))])
    close = np.concatenate([(np.abs(s.obs[:, F.index("opponent_dx")]) * observation.X < 200)[s.hit_ok & t]
                            for s, t in zip(sessions, there)])
    transfer = {s: auc(mordex[:, j], hit) for j, s in enumerate(STATES)}
    near = {s: auc(mordex[close, j], hit[close]) for j, s in enumerate(STATES)}
    distance = auc(-np.concatenate([(np.abs(s.obs[:, F.index("opponent_dx")]) + np.abs(s.obs[:, F.index("opponent_dy")]))[
        s.hit_ok & t] for s, t in zip(sessions, there)]), hit)
    for i, root in enumerate(roots):
        np.savez_compressed(root / f"state_k{args.k}.npz", agent=held[i][0], opponent=held[i][1], states=np.array(STATES))
    mean = lambda rows: np.mean(rows, 0)  # noqa: E731
    spread = lambda rows: np.std(rows, 0)  # noqa: E731
    lines = ["# Probe: does a crop show what its fighter is doing?", "",
             f"{datetime.now():%Y-%m-%d %H:%M}. {len(roots)} {args.difficulty} sessions, held out whole (3 folds); AUC, mean ± "
             f"spread over the folds (0.5 is chance). Labels from the human's key presses: Wu Shang is "
             + ", ".join(f"{s} in {r:.1%} of decisions" for s, r in zip(STATES, rates)) + ". scripts/probe_state.py.", "",
             "| Input (Wu Shang only) | " + " | ".join(s.capitalize() for s in STATES) + " |", "|---|---|---|---|",
             *(f"| {name} | " + " | ".join(f"{m:.3f} ± {d:.3f}" for m, d in zip(mean(rows), spread(rows))) + " |"
               for name, rows in scores.items()), "",
             "The crop model, trained on Wu Shang only, applied to Mordex's crop: how well each output ranks the decisions "
             "after which Wu Shang takes damage within 0.5 s (AUC; held-out sessions):", "",
             "| Output for Mordex | All decisions | Fighters within 200 px |", "|---|---|---|",
             *(f"| {s} | {transfer[s]:.3f} | {near[s]:.3f} |" for s in STATES),
             f"| (for scale: being close, by the vector's distance) | {distance:.3f} | |", "",
             "The vector's own prediction of that damage reaches 0.76 (results/probes/vision.md)."]
    path = RESULTS / "probes" / "state.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), f"\n-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
