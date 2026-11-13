"""Train the recurrent readers (brawl/reader.py), one per fighter, and score them on human play.

    python scripts/play.py scripted --playing mordex --difficulty medium --minutes 60     # the labelled play
    python scripts/play.py scripted --playing wu_shang --difficulty medium --minutes 60
    python scripts/train_reader.py

For each fighter: trained on the scripted sessions in which that fighter is the one played (its
key presses are the labels; the last 15% of each session is held back to stop on), then scored on
human sessions with that fighter, which the reader never saw and whose crops come from recorded
JPEG frames, not from the live screen. Two inputs are compared, so that what the picture adds is
measured: the fighter's kinematics alone through the same GRU, and crop + kinematics.

Writes runs/reader/<fighter>.pt, results/probes/reader.md, and for every human Wu Shang session
<recording>/reader_k<k>.npz: both readers' outputs and hidden states at every decision, for the
imitation probe (scripts/probe_vision.py). Those sessions were never used to train a reader.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from brawl import observation, reader  # noqa: E402
from brawl.paths import DATA, RESULTS, RUNS  # noqa: E402

LENGTH, BATCH, PATIENCE, TAIL = 128, 16, 25, 0.15  # 35 updates per epoch and hour of play: patience in epochs is long
NAMES = {"wu_shang": "Wu Shang", "mordex": "Mordex"}


def auc(p: np.ndarray, y: np.ndarray) -> float:
    ranks = np.empty(len(p))
    ranks[np.argsort(p)] = np.arange(1, len(p) + 1)
    return float((ranks[y > 0].sum() - y.sum() * (y.sum() + 1) / 2) / max(y.sum() * (len(y) - y.sum()), 1))


class Session:
    def __init__(self, root: Path, k: int, who: str, source: str):
        z = np.load(root / f"decisions_k{k}.npz")
        self.name, n = root.name, len(z["obs"])
        self.kin = reader.kinematics(z["obs"], who)
        self.crops = np.load(root / f"crops_k{k}_96.npy", mmap_mode="r")  # [>= n, 2, 96, 96]
        self.index = reader.SIDES[who][1]
        self.present = self.kin[:, reader.KINEMATICS.index("present")] > 0
        self.y = reader.targets(z["action"], self.kin[:, reader.KINEMATICS.index("grounded")] > 0, reader.DELAY[source])
        self.n = n


def score(out: np.ndarray, sessions: list[Session], parts: list[slice]) -> dict:
    y = np.concatenate([s.y[p][s.present[p]] for s, p in zip(sessions, parts)])
    r = {name: auc(out[:, i], y[:, i]) for i, name in enumerate(reader.STATES)}
    r["since_dodge"] = float(np.abs(out[:, 3] - y[:, 3]).mean() * observation.CLIP)  # decisions
    r["jumps"] = float(np.abs(out[:, 4] - y[:, 4]).mean() * 4)                       # jumps
    return r


def train(sessions: list[Session], crop: bool, device: str, epochs: int, seed: int, log):
    import torch
    import torch.nn.functional as fn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = reader.build(crop, device)
    optim = torch.optim.Adam(model.parameters(), lr=5e-4)
    cut = [int(s.n * (1 - TAIL)) for s in sessions]
    gpu = [torch.as_tensor(np.ascontiguousarray(s.crops[:s.n, s.index]), device=device) if crop else None for s in sessions]
    kin = [torch.as_tensor(s.kin, device=device) for s in sessions]
    y = [torch.as_tensor(s.y, device=device) for s in sessions]
    mask = [torch.as_tensor(s.present, device=device).float() for s in sessions]
    tails = [slice(c, s.n) for s, c in zip(sessions, cut)]

    def held_back():
        out = np.concatenate([reader.read(model, s.crops[c:s.n, s.index], s.kin[c:], device)[0][s.present[c:]]
                              for s, c in zip(sessions, cut)])
        return score(out, sessions, tails)

    best, state, since, t0 = None, None, 0, time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        windows = [(i, a) for i in range(len(sessions)) for a in range(int(rng.integers(LENGTH)), cut[i] - LENGTH, LENGTH)]
        order = rng.permutation(len(windows))
        for b in range(0, len(order), BATCH):
            batch = [windows[j] for j in order[b:b + BATCH]]
            img = None
            if crop:
                img = torch.stack([gpu[i][a:a + LENGTH] for i, a in batch]).float().div_(255)
                theta = torch.zeros(len(batch), 2, 3, device=device)  # one random shift of up to 4 px per window
                theta[:, 0, 0] = theta[:, 1, 1] = 1
                theta[:, :, 2] = (torch.rand(len(batch), 2, device=device) * 2 - 1) * (8 / 96)
                grid = fn.affine_grid(theta, (len(batch), LENGTH, 96, 96), align_corners=False)
                img = fn.grid_sample(img, grid, padding_mode="zeros", align_corners=False)
            out = model(img, torch.stack([kin[i][a:a + LENGTH] for i, a in batch]))[0]
            target = torch.stack([y[i][a:a + LENGTH] for i, a in batch])
            m = torch.stack([mask[i][a:a + LENGTH] for i, a in batch])
            loss = (fn.binary_cross_entropy_with_logits(out[..., :3], target[..., :3], reduction="none").mean(-1)
                    + 4 * ((out[..., 3:] - target[..., 3:]) ** 2).mean(-1))
            loss = (loss * m).sum() / m.sum().clamp(min=1)
            optim.zero_grad()
            loss.backward()
            optim.step()
        v = held_back()
        value = float(np.mean([v[s] for s in reader.STATES])) - v["since_dodge"] / observation.CLIP - v["jumps"] / 4
        if best is None or value > best:
            best, since, state = value, 0, {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            since += 1
        log(f"    epoch {epoch}: held back " + ", ".join(f"{s} {v[s]:.3f}" for s in reader.STATES)
            + f"; {(time.perf_counter() - t0) / epoch:.0f} s per epoch")
        if since >= PATIENCE:
            break
    model.load_state_dict(state)
    return model, held_back()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--min-minutes", type=float, default=10.0, help="scripted sessions shorter than this are left out")
    args = p.parse_args(argv)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    found = []
    for root in sorted(q for q in (DATA / "demos").iterdir() if (q / "meta.json").exists()):
        meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        if (root / f"crops_k{args.k}_96.npy").exists() and (root / f"decisions_k{args.k}.npz").exists() \
                and not meta["setting"].get("overlays"):
            found.append((root, meta.get("source"), meta["setting"].get("agent", "Wu Shang"), meta["rows"] / 3600))
    log = lambda m: print(m, flush=True)  # noqa: E731
    lines, models = [], {}
    for who in ("wu_shang", "mordex"):
        scripted = [Session(r, args.k, who, src) for r, src, agent, minutes in found
                    if src == "scripted" and agent == NAMES[who] and minutes >= args.min_minutes]
        human = [Session(r, args.k, who, src) for r, src, agent, _ in found if src == "human" and agent == NAMES[who]]
        if not scripted or not human:
            raise SystemExit(f"{who}: needs scripted and human sessions with crops (found {len(scripted)} and {len(human)})")
        rates = np.concatenate([s.y[s.present] for s in human]).mean(0)
        log(f"{who}: {sum(s.n for s in scripted)} scripted decisions to train on, {sum(s.n for s in human)} human to test on")
        lines += [f"## {NAMES[who]}", "",
                  f"Trained on {sum(s.n for s in scripted) * args.k / 3600:.0f} min of scripted play "
                  f"({', '.join(s.name for s in scripted)}); tested on {sum(s.n for s in human) * args.k / 3600:.0f} min of "
                  f"human play ({len(human)} sessions), where the fighter is attacking in {rates[0]:.1%} of decisions, "
                  f"dodging in {rates[1]:.1%}, jumping in {rates[2]:.1%}.", "",
                  "| Input | Play | Attacking (AUC) | Dodging (AUC) | Jumping (AUC) | Since the last dodge (error, decisions) | "
                  "Jumps since the ground (error) |", "|---|---|---|---|---|---|---|"]
        for crop in (False, True):
            name = "crop + kinematics" if crop else "kinematics"
            log(f"  {name}")
            model, back = train(scripted, crop, device, args.epochs, 0, log)
            out = np.concatenate([reader.read(model, s.crops[:s.n, s.index], s.kin, device)[0][s.present] for s in human])
            test = score(out, human, [slice(0, s.n) for s in human])
            for play, r in (("scripted, held back", back), ("human", test)):
                lines.append(f"| {name} | {play} | {r['attacking']:.3f} | {r['dodging']:.3f} | {r['jumping']:.3f} | "
                             f"{r['since_dodge']:.1f} | {r['jumps']:.2f} |")
            log(f"  -> human: " + ", ".join(f"{k} {v:.3f}" for k, v in test.items()))
            if crop:
                models[who] = model
                path = RUNS / "reader" / f"{who}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), path)
        lines.append("")
        del scripted
        torch.cuda.empty_cache()
    written = 0
    for root, src, agent, _ in found:  # both readers over every human Wu Shang session, for the imitation probe
        if src != "human" or agent != "Wu Shang":
            continue
        obs = np.load(root / f"decisions_k{args.k}.npz")["obs"]
        crops = np.load(root / f"crops_k{args.k}_96.npy", mmap_mode="r")
        states, hidden = zip(*(reader.read(models[who], crops[:len(obs), reader.SIDES[who][1]], reader.kinematics(obs, who), device)
                               for who in ("wu_shang", "mordex")))
        np.savez_compressed(root / f"reader_k{args.k}.npz", states=np.hstack(states).astype(np.float32),
                            hidden=np.hstack(hidden).astype(np.float16), outputs=np.array(reader.OUTPUTS))
        written += 1
    head = ["# The recurrent readers: what each fighter is doing, from its crop over time", "",
            f"{datetime.now():%Y-%m-%d %H:%M}. scripts/train_reader.py. Labels from key presses (brawl/reader.py). AUC: 0.5 is "
            "chance. A reader never saw the human sessions it is tested on; their crops come from recorded JPEG frames, "
            "the scripted ones from the live screen.", ""]
    path = RESULTS / "probes" / "reader.md"
    path.write_text("\n".join(head + lines) + f"\nOutputs and hidden states written for {written} human Wu Shang sessions "
                    f"(reader_k{args.k}.npz).\n", encoding="utf-8")
    print("\n".join(lines), f"\n-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
