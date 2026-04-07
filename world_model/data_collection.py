"""Collect (s_t, a_t, s_{t+1}) transitions from BrawlDeepEnv for world-model training.

Usage:
    python -m world_model.data_collection --episodes 50 --out world_model/data/transitions.npz
    python -m world_model.data_collection --episodes 50 --policy random --delay 3
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from env import BrawlDeepEnv, EnvConfig
from feature_extractor.memory.state_spec import StateSpec


def collect_transitions(
    n_episodes: int,
    max_steps_per_episode: int = 1200,
    policy: str = "random",
    env_config: EnvConfig | None = None,
) -> dict[str, np.ndarray]:
    """Run the env for *n_episodes* and return transition arrays.

    Returns dict with keys:
        states   : (N, state_dim) float32 – s_t  (includes prev_action features)
        actions  : (N, 4)  int64   – MultiDiscrete action [move, jump, dodge, attack]
        next_states : (N, state_dim) float32 – s_{t+1}
        dones    : (N,) bool – True when episode ended at this step
    """
    if env_config is None:
        env_config = EnvConfig(
            action_repeat_steps=2,
            action_repeat_min_steps=2,
            action_repeat_max_steps=2,
            yolo_infer_every_n_steps=3,
            tap_latch_steps=1,
            terminate_on_stock_out=True,
            max_episode_steps=max_steps_per_episode,
        )

    env = BrawlDeepEnv(config=env_config)

    states, actions, next_states, dones = [], [], [], []

    try:
        for ep in range(n_episodes):
            obs, _ = env.reset()
            obs = np.asarray(obs, dtype=np.float32)
            done = False
            step = 0

            while not done and step < max_steps_per_episode:
                if policy == "random":
                    action = env.action_space.sample()
                elif policy == "idle":
                    action = np.array([3, 0, 0, 0], dtype=np.int64)  # stand still
                else:
                    raise ValueError(f"Unknown policy: {policy}")

                next_obs, _, terminated, truncated, _ = env.step(action)
                next_obs = np.asarray(next_obs, dtype=np.float32)
                done = terminated or truncated

                states.append(obs.copy())
                actions.append(np.asarray(action, dtype=np.int64))
                next_states.append(next_obs.copy())
                dones.append(done)

                obs = next_obs
                step += 1

            print(f"Episode {ep + 1}/{n_episodes}  steps={step}")
    finally:
        env.close()

    return {
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.int64),
        "next_states": np.array(next_states, dtype=np.float32),
        "dones": np.array(dones, dtype=bool),
    }


def main():
    parser = argparse.ArgumentParser(description="Collect world-model transitions")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--policy", type=str, default="random", choices=["random", "idle"])
    parser.add_argument("--out", type=str, default="world_model/data/transitions.npz")
    parser.add_argument("--delay", type=float, default=3.0,
                        help="Seconds to wait before starting (switch to Brawlhalla)")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Collecting {args.episodes} episodes with '{args.policy}' policy")
    print(f"Starting in {args.delay:.0f}s — switch to Brawlhalla!")
    time.sleep(args.delay)

    data = collect_transitions(
        n_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        policy=args.policy,
    )

    np.savez_compressed(str(out_path), **data)
    n = len(data["states"])
    print(f"Saved {n:,} transitions to {out_path}")
    print(f"  states  : {data['states'].shape}")
    print(f"  actions : {data['actions'].shape}")
    print(f"  dones   : {data['dones'].sum()} terminal steps")


if __name__ == "__main__":
    main()
