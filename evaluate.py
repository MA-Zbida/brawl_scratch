#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence, cast

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parent))

from env import BrawlDeepEnv, EnvConfig
from train.curriculum_config import PHASES, build_phase_spec
from train.curriculum_goals import GOAL_DIM, GOAL_INDEX, clip_goal_target, default_goal_target
from train.llc_stage_common import StageGoalEnv
from feature_extractor.memory.state_spec import StateSpec
from wrappers.goal_env_wrapper import FlattenMultiDiscreteWrapper, StageGoalDictEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a PPO model for N episodes")
    p.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac"], help="Algorithm used by checkpoint")
    p.add_argument("--model", type=str, required=True, help="Path to PPO .zip checkpoint")
    p.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    p.add_argument("--phase", type=str, default=None, choices=list(PHASES), help="Use StageGoalEnv for this phase")
    p.add_argument(
        "--goal-mode",
        type=str,
        default="fixed",
        choices=["fixed", "phase", "none"],
        help="fixed: unarmed->weapon goal, armed->fight goal; phase: use StageGoalEnv; none: base env only",
    )
    p.add_argument("--max-episode-steps", type=int, default=0, help="Hard cap (0 = no cap)")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stochastic", action="store_true", help="Use stochastic actions instead of deterministic")

    p.add_argument("--yolo-every", type=int, default=2)
    p.add_argument("--yolo-blend-alpha", type=float, default=0.95)
    p.add_argument("--tracker-max-missing", type=int, default=2)
    p.add_argument("--tracker-smooth-alpha", type=float, default=0.85)

    p.add_argument("--death-penalty", type=float, default=1.0, help="Only used when --phase is set")
    return p.parse_args()


class FixedWeaponFightGoalEnv(gym.Wrapper):
    """Goal-conditioned obs wrapper with fixed hierarchical intent.

    - Unarmed: goal is to acquire weapon
    - Armed: goal is to fight/deal damage
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.goal_dim = int(GOAL_DIM)
        obs_shape = getattr(env.observation_space, "shape", None)
        if obs_shape is None or len(obs_shape) == 0:
            raise ValueError("Underlying env must expose a 1D observation space with known shape")
        self._base_dim = int(obs_shape[0])
        self._aug_dim = self._base_dim + (2 * self.goal_dim)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._aug_dim,),
            dtype=np.float32,
        )
        self._obs_buf = np.zeros((self._aug_dim,), dtype=np.float32)
        self._goal_target = np.zeros((self.goal_dim,), dtype=np.float32)
        self._goal_mask = np.zeros((self.goal_dim,), dtype=np.float32)

    def _build_fixed_goal(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
        o = np.asarray(obs, dtype=np.float32)
        target = default_goal_target()
        mask = np.zeros((self.goal_dim,), dtype=np.float32)

        has_weapon = float(np.clip(StateSpec.get(o, "player_has_weapon"), 0.0, 1.0)) > 0.5

        if not has_weapon:
            # Weapon acquisition sub-goal.
            target[GOAL_INDEX["player_has_weapon"]] = 1.0
            target[GOAL_INDEX["weapon_dx"]] = 0.5
            target[GOAL_INDEX["weapon_dy"]] = 0.5
            target[GOAL_INDEX["player_is_offstage"]] = 0.0

            mask[GOAL_INDEX["player_has_weapon"]] = 1.0
            mask[GOAL_INDEX["weapon_dx"]] = 1.0
            mask[GOAL_INDEX["weapon_dy"]] = 1.0
            mask[GOAL_INDEX["player_is_offstage"]] = 0.4
            mode = "weapon"
        else:
            # Damage/fight sub-goal.
            curr_op_dmg = float(np.clip(StateSpec.get(o, "opponent_damage_pct"), 0.0, 1.0))

            target[GOAL_INDEX["player_has_weapon"]] = 1.0
            target[GOAL_INDEX["in_strike_range"]] = 1.0
            target[GOAL_INDEX["rel_distance"]] = 0.03
            target[GOAL_INDEX["facing_opponent"]] = 1.0
            target[GOAL_INDEX["frame_advantage_estimate"]] = 0.7
            target[GOAL_INDEX["opponent_damage_pct"]] = float(np.clip(curr_op_dmg + 0.12, 0.0, 1.0))
            target[GOAL_INDEX["player_is_offstage"]] = 0.0

            mask[GOAL_INDEX["player_has_weapon"]] = 0.4
            mask[GOAL_INDEX["in_strike_range"]] = 1.0
            mask[GOAL_INDEX["rel_distance"]] = 0.6
            mask[GOAL_INDEX["facing_opponent"]] = 0.5
            mask[GOAL_INDEX["frame_advantage_estimate"]] = 0.3
            mask[GOAL_INDEX["opponent_damage_pct"]] = 1.0
            mask[GOAL_INDEX["player_is_offstage"]] = 0.2
            mode = "fight"

        return clip_goal_target(target), np.clip(mask, 0.0, 1.0).astype(np.float32), mode

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        np.copyto(self._obs_buf[: self._base_dim], np.asarray(obs, dtype=np.float32))
        np.copyto(self._obs_buf[self._base_dim : self._base_dim + self.goal_dim], self._goal_target)
        np.copyto(self._obs_buf[self._base_dim + self.goal_dim :], self._goal_mask)
        return self._obs_buf

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._goal_target, self._goal_mask, mode = self._build_fixed_goal(obs)
        info = dict(info)
        info["goal_target"] = self._goal_target.copy()
        info["goal_mask"] = self._goal_mask.copy()
        info["goal_mode"] = mode
        info["goal_active"] = 1.0
        info["stage_name"] = "fixed_weapon_fight"
        return self._augment(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._goal_target, self._goal_mask, mode = self._build_fixed_goal(obs)
        info = dict(info)
        info["goal_target"] = self._goal_target.copy()
        info["goal_mask"] = self._goal_mask.copy()
        info["goal_mode"] = mode
        info["goal_active"] = 1.0
        info["stage_name"] = "fixed_weapon_fight"
        return self._augment(obs), reward, terminated, truncated, info


def make_env(args: argparse.Namespace):
    config = EnvConfig(
        terminate_on_stock_out=True,
        max_episode_steps=max(0, int(args.max_episode_steps)),
        yolo_infer_every_n_steps=max(1, int(args.yolo_every)),
        yolo_obs_blend_alpha=float(np.clip(args.yolo_blend_alpha, 0.0, 1.0)),
        tracker_max_missing=max(1, int(args.tracker_max_missing)),
        tracker_smooth_alpha=float(np.clip(args.tracker_smooth_alpha, 0.0, 1.0)),
        action_repeat_steps=1,
        action_repeat_min_steps=1,
        action_repeat_max_steps=1,
        tap_latch_steps=1,
    )
    base_env = BrawlDeepEnv(config=config)
    stage_spec = None

    if args.goal_mode == "none":
        env: gym.Env = base_env
    elif args.goal_mode == "fixed":
        env = FixedWeaponFightGoalEnv(base_env)
    else:
        if args.phase is None:
            raise ValueError("--phase is required when --goal-mode=phase")

        stage_spec = build_phase_spec(
            phase=args.phase,
            death_penalty=float(args.death_penalty),
            terminate_on_death=False,
        )
        # During evaluation, terminate only when a stock-out happens (or max-steps if configured).
        stage_spec.terminate_on_death = False
        stage_spec.terminate_on_goal_success = False
        stage_spec.terminate_on_hit_event = False
        env = StageGoalEnv(base_env, stage_spec)

    if args.algo == "sac":
        env = FlattenMultiDiscreteWrapper(env)
        if stage_spec is not None:
            goal_dim = int(len(stage_spec.feature_names)) if stage_spec.feature_names is not None else int(
                np.asarray(stage_spec.mask, dtype=np.float32).reshape(-1).shape[0]
            )
            env = StageGoalDictEnv(
                env,
                proximity_scale=float(stage_spec.proximity_scale),
                success_threshold=float(stage_spec.success_threshold),
                success_bonus=float(stage_spec.success_bonus),
                mask=np.asarray(stage_spec.mask, dtype=np.float32),
                goal_extractor=stage_spec.goal_extractor,
                goal_dim=goal_dim,
                use_l2_error=bool(stage_spec.use_l2_error),
            )

    return env


def _resolve_outcome(self_stocks: float, op_stocks: float, truncated: bool) -> str:
    if op_stocks <= 0.0 and self_stocks > 0.0:
        return "WIN"
    if self_stocks <= 0.0 and op_stocks > 0.0:
        return "LOSS"
    if self_stocks <= 0.0 and op_stocks <= 0.0:
        return "DRAW"
    if truncated:
        return "TRUNC"
    return "UNRESOLVED"


def _to_env_action(action: Any, algo: str) -> Sequence[int] | int:
    if algo == "sac":
        return int(np.asarray(action, dtype=np.int64).reshape(-1)[0])

    arr = np.asarray(action, dtype=np.int64).reshape(-1)
    if arr.shape[0] < 4:
        raise ValueError(f"Predicted action must have 4 components, got shape {arr.shape}")
    return [int(arr[0]), int(arr[1]), int(arr[2]), int(arr[3])]


def _get_base_env(env) -> BrawlDeepEnv:
    base = env.unwrapped
    if not isinstance(base, BrawlDeepEnv):
        raise TypeError(f"Expected unwrapped env to be BrawlDeepEnv, got {type(base)}")
    return base


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    env = make_env(args)
    deterministic = not bool(args.stochastic)

    if args.algo == "sac":
        from algo.discrete_sac import DiscreteSAC

        model = DiscreteSAC.load(str(model_path), device=args.device)
    else:
        model = PPO.load(str(model_path), device=args.device)

    rewards: list[float] = []
    lengths: list[int] = []
    outcomes: list[str] = []
    self_stocks_list: list[float] = []
    op_stocks_list: list[float] = []
    op_dmg_totals: list[float] = []
    self_dmg_totals: list[float] = []

    print("=" * 84)
    print(f"Evaluating: {model_path}")
    phase_view = args.phase if args.goal_mode == "phase" else "n/a"
    print(
        f"Algo: {args.algo} | Episodes: {args.episodes} | Goal mode: {args.goal_mode} | "
        f"Phase: {phase_view} | Deterministic: {deterministic}"
    )
    print("Episode ends on stock-out of either side.")
    print("=" * 84)

    try:
        for ep in range(1, int(args.episodes) + 1):
            obs, _ = env.reset(seed=int(args.seed + ep))
            done = False
            truncated = False

            ep_reward = 0.0
            ep_len = 0
            ep_op_dmg = 0.0
            ep_self_dmg = 0.0

            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=deterministic)
                if isinstance(action, np.ndarray) and action.ndim > 1:
                    action = action[0]
                env_action = _to_env_action(action, args.algo)
                obs, reward, done, truncated, info = cast(Any, env).step(env_action)
                ep_reward += float(reward)
                ep_len += 1
                ep_op_dmg += float(info.get("op_delta_damage", 0.0))
                ep_self_dmg += float(info.get("self_delta_damage", 0.0))

            base = _get_base_env(env)
            self_stocks = float(base.memory.self_stocks_left)
            op_stocks = float(base.memory.op_stocks_left)
            outcome = _resolve_outcome(self_stocks, op_stocks, bool(truncated))

            rewards.append(ep_reward)
            lengths.append(ep_len)
            outcomes.append(outcome)
            self_stocks_list.append(self_stocks)
            op_stocks_list.append(op_stocks)
            op_dmg_totals.append(ep_op_dmg)
            self_dmg_totals.append(ep_self_dmg)

            print(
                f"Ep {ep:02d} | {outcome:10s} | reward={ep_reward:+9.3f} | steps={ep_len:5d} | "
                f"stocks(self/op)={self_stocks:.1f}/{op_stocks:.1f} | "
                f"dmg(self/op)={ep_self_dmg:.3f}/{ep_op_dmg:.3f}"
            )
    finally:
        env.close()

    rewards_np = np.asarray(rewards, dtype=np.float32)
    lengths_np = np.asarray(lengths, dtype=np.float32)
    self_stocks_np = np.asarray(self_stocks_list, dtype=np.float32)
    op_stocks_np = np.asarray(op_stocks_list, dtype=np.float32)
    op_dmg_np = np.asarray(op_dmg_totals, dtype=np.float32)
    self_dmg_np = np.asarray(self_dmg_totals, dtype=np.float32)

    wins = sum(1 for x in outcomes if x == "WIN")
    losses = sum(1 for x in outcomes if x == "LOSS")
    draws = sum(1 for x in outcomes if x == "DRAW")
    truncs = sum(1 for x in outcomes if x == "TRUNC")

    print("\n" + "=" * 84)
    print("Evaluation summary")
    print("=" * 84)
    print(f"Episodes           : {len(outcomes)}")
    print(f"W/L/D/TRUNC        : {wins}/{losses}/{draws}/{truncs}")
    print(f"Win rate           : {wins / max(1, len(outcomes)):.3f}")
    print(f"Avg reward         : {float(rewards_np.mean()):+.3f} +- {float(rewards_np.std()):.3f}")
    print(f"Avg episode length : {float(lengths_np.mean()):.1f} steps")
    print(f"Avg stocks self/op : {float(self_stocks_np.mean()):.2f}/{float(op_stocks_np.mean()):.2f}")
    print(f"Avg dmg self/op    : {float(self_dmg_np.mean()):.3f}/{float(op_dmg_np.mean()):.3f}")
    print(f"Avg stock diff     : {float((self_stocks_np - op_stocks_np).mean()):+.3f}")


if __name__ == "__main__":
    main()
