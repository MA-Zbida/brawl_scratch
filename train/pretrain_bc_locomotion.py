#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from torch.nn.utils import clip_grad_norm_

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env import BrawlDeepEnv, EnvConfig
from feature_extractor.film_extractor import StageGoalFiLMExtractor
from train.curriculum_config import PHASES, build_phase_spec
from train.llc_stage_common import StageGoalEnv
from hierarchical.goals import extract_goal_features


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Behavioral cloning pretraining for curriculum PPO (all phases)")
    p.add_argument("--phase", type=str, default="locomotion", choices=list(PHASES))
    p.add_argument("--demos", type=str, default="")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--entropy-coef", type=float, default=1e-3)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output", type=str, default="")
    p.add_argument("--vecnorm-output", type=str, default="")
    p.add_argument("--max-episode-steps", type=int, default=1200)
    p.add_argument("--goal-relabel", dest="goal_relabel", action="store_true", default=True)
    p.add_argument("--no-goal-relabel", dest="goal_relabel", action="store_false")
    p.add_argument("--relabel-horizons", type=str, default="1,2,4,8")
    p.add_argument("--max-relabels-per-step", type=int, default=0,
                   help="0 = keep all valid horizons, >0 caps relabel samples per source step")
    p.add_argument("--no-include-original-samples", dest="include_original_samples", action="store_false")
    p.set_defaults(include_original_samples=True)
    return p.parse_args()


def _build_env(max_episode_steps: int, phase: str) -> StageGoalEnv:
    spec = build_phase_spec(phase, death_penalty=1.0, terminate_on_death=True)
    config = EnvConfig(
        terminate_on_stock_out=False,
        max_episode_steps=max_episode_steps,
        yolo_infer_every_n_steps=3,
        action_repeat_steps=1,
        action_repeat_min_steps=1,
        action_repeat_max_steps=1,
        tap_latch_steps=1,
    )
    base = BrawlDeepEnv(config=config)
    return StageGoalEnv(base, spec)


def _resolve_paths(args: argparse.Namespace) -> tuple[str, str, str]:
    demos = args.demos.strip() if args.demos else ""
    output = args.output.strip() if args.output else ""
    vecn = args.vecnorm_output.strip() if args.vecnorm_output else ""

    if not demos:
        demos = f"train/models/{args.phase}_demos.npz"
    if not output:
        output = f"train/models/llc_{args.phase}_bc_init.zip"
    if not vecn:
        vecn = f"train/models/llc_{args.phase}_bc_init.vecnormalize.pkl"
    return demos, output, vecn


def _load_dataset(path: str, expected_phase: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    if "obs" not in data or "actions" not in data or "dones" not in data:
        raise ValueError("Demo file must contain 'obs', 'actions', and 'dones' arrays")

    if "phase" in data:
        phase_arr = np.asarray(data["phase"]).reshape(-1)
        if phase_arr.size > 0:
            phase_value = str(phase_arr[0])
            if phase_value != expected_phase:
                raise ValueError(
                    f"Demo phase mismatch: file phase='{phase_value}', requested phase='{expected_phase}'"
                )

    obs = np.asarray(data["obs"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.int64)
    dones = np.asarray(data["dones"], dtype=bool)
    if obs.ndim != 2:
        raise ValueError(f"Expected obs shape (N,D), got {obs.shape}")
    if actions.ndim != 2 or actions.shape[1] != 4:
        raise ValueError(f"Expected actions shape (N,4), got {actions.shape}")
    if dones.ndim != 1:
        raise ValueError(f"Expected dones shape (N,), got {dones.shape}")
    if not (obs.shape[0] == actions.shape[0] == dones.shape[0]):
        raise ValueError("obs/actions/dones length mismatch")
    return obs, actions, dones


def _parse_horizons(raw: str) -> list[int]:
    vals: list[int] = []
    for tok in str(raw).split(","):
        tok = tok.strip()
        if not tok:
            continue
        h = int(tok)
        if h <= 0:
            continue
        vals.append(h)
    vals = sorted(set(vals))
    if not vals:
        raise ValueError("relabel_horizons must contain at least one positive integer")
    return vals


def _episode_ranges(dones: np.ndarray) -> list[tuple[int, int]]:
    n = int(dones.shape[0])
    if n == 0:
        return []
    ranges: list[tuple[int, int]] = []
    start = 0
    for i, d in enumerate(dones.tolist()):
        if bool(d):
            ranges.append((start, i))
            start = i + 1
    if start < n:
        ranges.append((start, n - 1))
    return ranges


def _extract_goal_from_base(base_obs: np.ndarray, spec) -> np.ndarray:
    if spec.goal_extractor is not None:
        goal = np.asarray(spec.goal_extractor(base_obs), dtype=np.float32).reshape(-1)
    else:
        goal = np.asarray(extract_goal_features(base_obs), dtype=np.float32).reshape(-1)
    return np.clip(goal, 0.0, 1.0)


def _build_goal_relabel_dataset(
    obs: np.ndarray,
    actions: np.ndarray,
    dones: np.ndarray,
    spec,
    horizons: list[int],
    max_relabels_per_step: int,
    include_original_samples: bool,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    goal_dim = int(len(spec.feature_names)) if spec.feature_names is not None else int(spec.mask.shape[0])
    total_dim = int(obs.shape[1])
    base_dim = total_dim - (2 * goal_dim)
    if base_dim <= 0:
        raise ValueError(
            f"Invalid dims for relabeling: total_dim={total_dim}, goal_dim={goal_dim}"
        )

    mask = np.asarray(spec.mask, dtype=np.float32).reshape(goal_dim)
    ranges = _episode_ranges(dones)

    out_obs: list[np.ndarray] = []
    out_actions: list[np.ndarray] = []
    relabeled_count = 0

    for start, end in ranges:
        for t in range(start, end + 1):
            src_obs = obs[t]
            src_action = actions[t]

            if include_original_samples:
                out_obs.append(src_obs.copy())
                out_actions.append(src_action.copy())

            relabeled_for_step = 0
            for h in horizons:
                ft = t + h
                if ft > end:
                    continue

                future_base = obs[ft, :base_dim]
                goal = _extract_goal_from_base(future_base, spec)
                if goal.shape[0] != goal_dim:
                    raise ValueError(
                        f"Goal extractor dim={goal.shape[0]} does not match goal_dim={goal_dim}"
                    )

                rel_obs = src_obs.copy()
                rel_obs[base_dim : base_dim + goal_dim] = goal
                rel_obs[base_dim + goal_dim : base_dim + 2 * goal_dim] = mask

                out_obs.append(rel_obs)
                out_actions.append(src_action.copy())
                relabeled_count += 1
                relabeled_for_step += 1

                if max_relabels_per_step > 0 and relabeled_for_step >= max_relabels_per_step:
                    break

    if len(out_obs) == 0:
        raise ValueError("Goal relabeling produced empty dataset")

    return (
        np.stack(out_obs).astype(np.float32),
        np.stack(out_actions).astype(np.int64),
        len(ranges),
        relabeled_count,
    )


def main() -> None:
    args = parse_args()
    demos_path, output_path, vecn_path = _resolve_paths(args)

    obs_np, act_np, dones_np = _load_dataset(demos_path, expected_phase=args.phase)
    n_samples_raw = int(obs_np.shape[0])

    if n_samples_raw < 1000:
        print(f"WARNING: only {n_samples_raw} raw samples. BC may underfit.")

    spec = build_phase_spec(args.phase, death_penalty=1.0, terminate_on_death=True)
    if args.goal_relabel:
        horizons = _parse_horizons(args.relabel_horizons)
        obs_np, act_np, episode_count, relabeled_count = _build_goal_relabel_dataset(
            obs=obs_np,
            actions=act_np,
            dones=dones_np,
            spec=spec,
            horizons=horizons,
            max_relabels_per_step=max(0, int(args.max_relabels_per_step)),
            include_original_samples=bool(args.include_original_samples),
        )
        print("=" * 68)
        print("GOAL RELABELING ENABLED")
        print(
            f"phase={args.phase} episodes={episode_count} raw={n_samples_raw} "
            f"relabeled={relabeled_count} final={obs_np.shape[0]} "
            f"x{(obs_np.shape[0] / max(1, n_samples_raw)):.2f}"
        )
        print(f"horizons={horizons} include_original={args.include_original_samples}")
        print("=" * 68)
    else:
        print("=" * 68)
        print("GOAL RELABELING DISABLED")
        print(f"phase={args.phase} samples={n_samples_raw}")
        print("=" * 68)

    n_samples = int(obs_np.shape[0])

    vec_base = VecMonitor(DummyVecEnv([lambda: _build_env(args.max_episode_steps, args.phase)]))
    vec_env = VecNormalize(vec_base, norm_obs=False, norm_reward=False, clip_obs=10.0)

    policy_kwargs = dict(
        features_extractor_class=StageGoalFiLMExtractor,
        features_extractor_kwargs=dict(
            goal_feature_names=list(spec.feature_names or []),
            features_dim=256,
        ),
        net_arch=dict(pi=[128], vf=[128]),
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=float(args.learning_rate),
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.15,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=float(args.max_grad_norm),
        seed=42,
        policy_kwargs=policy_kwargs,
        verbose=0,
        device=args.device,
    )

    device = model.policy.device
    obs_all = th.as_tensor(obs_np, dtype=th.float32, device=device)
    act_all = th.as_tensor(act_np, dtype=th.long, device=device)

    print("=" * 68)
    print(f"BC PRETRAIN — {args.phase.upper()} PPO")
    print(f"samples={n_samples} epochs={args.epochs} batch={args.batch_size}")
    print("=" * 68)

    for epoch in range(1, int(args.epochs) + 1):
        perm = th.randperm(n_samples, device=device)
        losses: list[float] = []
        entropies: list[float] = []

        for start in range(0, n_samples, int(args.batch_size)):
            idx = perm[start : start + int(args.batch_size)]
            if idx.numel() == 0:
                continue

            obs_b = obs_all[idx]
            act_b = act_all[idx]

            dist = model.policy.get_distribution(obs_b)
            log_prob = dist.log_prob(act_b)
            entropy = dist.entropy()

            loss = -log_prob.mean() - float(args.entropy_coef) * entropy.mean()

            model.policy.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.policy.parameters(), float(args.max_grad_norm))
            model.policy.optimizer.step()

            losses.append(float(loss.detach().cpu().item()))
            entropies.append(float(entropy.mean().detach().cpu().item()))

        with th.no_grad():
            pred = model.policy._predict(obs_all, deterministic=True)
            exact = (pred == act_all).all(dim=1).float().mean().item()
            per_dim = (pred == act_all).float().mean(dim=0).cpu().numpy()

        print(
            f"epoch {epoch:02d}/{args.epochs} loss={np.mean(losses):.4f} "
            f"entropy={np.mean(entropies):.4f} exact_acc={exact:.3f} "
            f"dim_acc={np.round(per_dim, 3)}"
        )

    out_model = Path(output_path)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_model))

    vecn = model.get_vec_normalize_env()
    if vecn is not None:
        out_vec = Path(vecn_path)
        out_vec.parent.mkdir(parents=True, exist_ok=True)
        vecn.save(str(out_vec))

    print("Saved BC-initialized PPO checkpoint")
    print(f"  model: {out_model}")
    print(f"  vecn : {vecn_path}")
    print("Fine-tune with:")
    print(f"  python -m train.train_curriculum --phase {args.phase} --resume {out_model}")


if __name__ == "__main__":
    main()
