"""Train one run in the live game.

Start Brawlhalla in training mode (Wu Shang vs a Mordex bot, Brawlhaven small, camera Fixed, no
overlays), not paused, with the bot at the level the config names. Run this and click into the
game during the countdown. Press **F10** to stop early.

    python scripts/train.py --config configs/experiments/smoke.yaml
    python scripts/train.py --config configs/experiments/s1_ppo.yaml --variant realtime-bc --seed 0
    python scripts/train.py --config configs/base.yaml --set train.init=runs/bc/medium_v4/policy.zip --name my_test

Outputs go to runs/<experiment>/<variant>/seed<seed>/: config.yaml, meta.json, progress.csv (one row
per rollout), checkpoints/ and policy.zip. A run is never resumed: the game cannot be put back
where it was. A run that stopped early stays as seed<seed>_attempt<n>/ and a new one starts; a
finished run is skipped unless --force. Any saved policy (policy.zip, qnet.pt) plays with scripts/play.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import brawl.game.capture as capture  # noqa: E402  first: the screen duplicator must exist before torch loads

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

import yaml  # noqa: E402

from brawl.config import load_experiment  # noqa: E402
from brawl.paths import REPO_ROOT  # noqa: E402

GAME = "brawlhalla"  # in the title of the window that may receive the keys


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def demo_sessions(levels, k: int) -> list[Path]:
    """The human Wu Shang sessions at ``levels`` that are prepared (scripts/prepare_demos.py)."""
    from brawl.paths import DATA
    from brawl.record import Recording, is_demo

    return [r for r in sorted((DATA / "demos").iterdir()) if (r / "meta.json").exists() and (r / f"decisions_k{k}.npz").exists()
            and is_demo(Recording(r).meta) and Recording(r).meta["setting"].get("difficulty") in set(levels)]


def claim(run_dir: Path, force: bool) -> bool:
    """Make ``run_dir`` ready for a new run. False if it already holds a finished one."""
    if run_dir.exists():
        if force:
            shutil.rmtree(run_dir)
        elif (run_dir / "policy.zip").exists() or (run_dir / "qnet.pt").exists():
            return False
        else:  # stopped early: kept as an attempt
            n = 1 + sum(1 for _ in run_dir.parent.glob(f"{run_dir.name}_attempt*"))
            run_dir.rename(run_dir.with_name(f"{run_dir.name}_attempt{n}"))
    run_dir.mkdir(parents=True)
    return True


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path, help="experiment YAML, or a plain config such as configs/base.yaml")
    p.add_argument("--variant", help="variant name (required if the experiment has several)")
    p.add_argument("--seed", type=int, default=None, help="default: the experiment's first seed")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override settings, e.g. train.learning_rate=1.0e-4")
    p.add_argument("--name", help="experiment name for ad-hoc runs (default: the config's name)")
    p.add_argument("--force", action="store_true", help="delete an existing run directory and start over")
    p.add_argument("--countdown", type=int, default=5)
    args = p.parse_args(argv)

    experiment = load_experiment(args.config)
    if args.name:
        experiment.name = args.name
    variants = [name for name, _ in experiment.variants()]
    variant = args.variant or (variants[0] if len(variants) == 1 else None)
    if variant is None:
        print(f"{experiment.name} has {len(variants)} variants; pass --variant. Choices:\n  " + "\n  ".join(variants))
        return 2
    seed = experiment.seeds[0] if args.seed is None else args.seed
    config = experiment.resolve(variant, seed, args.set)
    env_cfg, train_cfg = config["env"], config["train"]
    if train_cfg["learner"] not in ("ppo", "rdqn"):
        raise SystemExit(f"unknown learner {train_cfg['learner']!r}: ppo or rdqn")
    if train_cfg["init"] and not (REPO_ROOT / train_cfg["init"]).exists():
        raise SystemExit(f"train.init: {REPO_ROOT / train_cfg['init']} does not exist")
    run_dir = experiment.run_dir(variant, seed)
    if not claim(run_dir, args.force):
        print(f"already finished: {run_dir} (use --force to train it again)")
        return 0
    print(capture.status())
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")

    from brawl import learn
    from brawl.brawlenv import BrawlEnv
    from brawl.game.clock import RealtimeClock, SteppedClock
    from brawl.game.counter import FrameCounter
    from brawl.game.keys import DirectInput, KeyReader, Keys, focused_window
    from brawl.sense.detectors import Yolo
    from brawl.sense.perception import Perception

    realtime = env_cfg["regime"] == "realtime"
    perception = Perception(Yolo(REPO_ROOT / env_cfg["detector"], conf=0.05), step=int(env_cfg["k"]))
    screen = capture.Screen(timeout_s=0.05)
    keys = Keys(DirectInput(), tap_s=0.01)
    clock = RealtimeClock(keys) if realtime else SteppedClock(keys, counter=FrameCounter(screen), key_gap_s=0.01)
    reader = KeyReader()
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    meta = {"experiment": experiment.name, "variant": variant, "seed": seed, "git": git_commit(),
            "started": datetime.now().isoformat(timespec="seconds"), "capture": capture.status()}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    if train_cfg["init"]:
        train_cfg = {**train_cfg, "init": str(REPO_ROOT / train_cfg["init"])}
    if train_cfg["rdqn"].get("prior"):
        train_cfg = {**train_cfg, "rdqn": {**train_cfg["rdqn"], "prior": str(REPO_ROOT / train_cfg["rdqn"]["prior"])}}
    print(f"{experiment.name} / {variant} / seed {seed}: {train_cfg['total_steps']} decisions "
          f"({int(train_cfg['total_steps']) * int(env_cfg['k']) / 3600:.0f} min of game time), {env_cfg['regime']}, "
          f"bot {env_cfg['difficulty']} (set it in the game)")
    for i in range(args.countdown, 0, -1):
        print(f"click into the game window... {i}", flush=True)
        time.sleep(1)
    perception(screen.grab(fresh=True).image, 0)  # warm the detector up before the clock starts
    clock.start()
    env = BrawlEnv(clock, screen, perception, k=int(env_cfg["k"]), window=0, damage_weight=float(env_cfg["damage_weight"]),
                   ko_weight=float(env_cfg["ko_weight"]))
    print("training: press F10 to stop", flush=True)
    focused = lambda: GAME in focused_window().lower()  # noqa: E731
    try:
        if train_cfg["learner"] == "rdqn":
            demos = demo_sessions(train_cfg["rdqn"]["demo_levels"], int(env_cfg["k"]))
            meta["demonstrations"] = [r.name for r in demos]
            saved = learn.train_rdqn(env, train_cfg, run_dir, demos, realtime=realtime, stop=reader.stop_requested,
                                     focused=focused, log=lambda m: print(m, flush=True),
                                     reward_weights=(float(env_cfg["damage_weight"]), float(env_cfg["ko_weight"])))
        else:
            saved = learn.train_ppo(env, train_cfg, run_dir, realtime=realtime, stop=reader.stop_requested,
                                    focused=focused, log=lambda m: print(m, flush=True))
    finally:
        env.close()
        keys.release_all()
    meta["ended"] = datetime.now().isoformat(timespec="seconds")
    meta["calls"] = {**env.calls, "steps": env.steps}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"-> {saved}\nplay it: python scripts/play.py {saved.relative_to(REPO_ROOT)} --difficulty {env_cfg['difficulty']} --minutes 5")
    return 0


if __name__ == "__main__":
    sys.exit(main())
