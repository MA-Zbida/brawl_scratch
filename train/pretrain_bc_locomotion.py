#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from torch.nn.utils import clip_grad_norm_

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env import BrawlDeepEnv, EnvConfig
from feature_extractor.film_extractor import StageGoalFiLMExtractor
from feature_extractor.memory.state_spec import StateSpec
from hierarchical.goals import extract_goal_features
from train.curriculum_config import PHASES, build_phase_spec
from train.llc_stage_common import StageGoalEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Behavioral cloning pretraining for curriculum PPO (all phases)")
    p.add_argument("--phase", type=str, default="locomotion", choices=[*list(PHASES), "auto"])
    p.add_argument("--demos", type=str, default="")
    p.add_argument(
        "--resume",
        type=str,
        default="",
        help="Optional PPO .zip checkpoint to initialize BC pretraining from",
    )
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
    p.add_argument(
        "--max-relabels-per-step",
        type=int,
        default=0,
        help="0 = keep all valid horizons, >0 caps relabel samples per source step",
    )
    p.add_argument("--no-include-original-samples", dest="include_original_samples", action="store_false")
    p.set_defaults(include_original_samples=True)
    return p.parse_args()


def _build_env(max_episode_steps: int, phase: str) -> StageGoalEnv:
    spec = build_phase_spec(phase, death_penalty=1.0, terminate_on_death=True)
    config = EnvConfig(
        terminate_on_stock_out=False,
        max_episode_steps=max_episode_steps,
        yolo_infer_every_n_steps=1,
        action_repeat_steps=1,
        action_repeat_min_steps=1,
        action_repeat_max_steps=1,
        tap_latch_steps=1,
    )
    base = BrawlDeepEnv(config=config)
    return StageGoalEnv(base, spec)


def _resolve_paths(args: argparse.Namespace) -> tuple[str, str]:
    demos = args.demos.strip() if args.demos else ""
    output = args.output.strip() if args.output else ""
    phase_for_default_paths = str(args.phase).strip().lower()
    if phase_for_default_paths == "auto":
        phase_for_default_paths = "locomotion"

    if not demos:
        demos = f"train/models/{phase_for_default_paths}_demos.npz"
    if not output:
        output = f"train/models/llc_{phase_for_default_paths}_bc_init.zip"
    return demos, output


def _phases_are_equivalent(phase_a: str, phase_b: str) -> bool:
    if str(phase_a) == str(phase_b):
        return True
    try:
        spec_a = build_phase_spec(str(phase_a), death_penalty=1.0, terminate_on_death=True)
        spec_b = build_phase_spec(str(phase_b), death_penalty=1.0, terminate_on_death=True)
        return str(spec_a.name) == str(spec_b.name)
    except Exception:
        return False


def _align_demo_lengths(
    obs: np.ndarray,
    actions: np.ndarray,
    dones: np.ndarray,
    *,
    path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_obs = int(obs.shape[0])
    n_actions = int(actions.shape[0])
    n_dones = int(dones.shape[0])

    if n_obs == n_actions == n_dones:
        return obs, actions, dones

    n_pair = min(n_obs, n_actions)
    if n_pair <= 0:
        raise ValueError(f"Demo file has no usable samples after alignment: {path}")
    obs = obs[:n_pair]
    actions = actions[:n_pair]

    if n_dones < n_pair:
        pad = n_pair - n_dones
        if pad <= 8:
            dones = np.concatenate([dones, np.zeros((pad,), dtype=bool)], axis=0)
            dones[-1] = True
            print(
                f"[BC pretrain] WARNING: repaired dones length in {path}: "
                f"obs={n_obs}, actions={n_actions}, dones={n_dones} -> padded {pad} and set final done=True"
            )
        else:
            obs = obs[:n_dones]
            actions = actions[:n_dones]
            print(
                f"[BC pretrain] WARNING: large length mismatch in {path}: "
                f"obs={n_obs}, actions={n_actions}, dones={n_dones} -> truncated obs/actions to dones length"
            )
    elif n_dones > n_pair:
        dones = dones[:n_pair]
        print(
            f"[BC pretrain] WARNING: trimmed extra dones in {path}: "
            f"obs={n_obs}, actions={n_actions}, dones={n_dones}"
        )

    if dones.size > 0 and not bool(dones[-1]):
        dones = dones.copy()
        dones[-1] = True

    if not (obs.shape[0] == actions.shape[0] == dones.shape[0]):
        raise ValueError(
            f"Could not align demo lengths in {path}: obs={obs.shape[0]} actions={actions.shape[0]} dones={dones.shape[0]}"
        )

    return obs, actions, dones


def _resolve_actions_for_ppo(data: np.lib.npyio.NpzFile) -> np.ndarray:
    raw_actions = np.asarray(data["actions"])

    if "actions_multidiscrete" in data:
        actions = np.asarray(data["actions_multidiscrete"], dtype=np.int64)
    elif raw_actions.ndim == 2 and raw_actions.shape[1] == 4:
        actions = raw_actions.astype(np.int64)
    else:
        raise ValueError(
            "Demo actions must be MultiDiscrete with shape (N,4). "
            "Please recollect demos with the current PPO-only collector."
        )

    if actions.ndim != 2 or actions.shape[1] != 4:
        raise ValueError(f"Expected PPO actions shape (N,4), got {actions.shape}")
    return actions


def _load_dataset(path: str, expected_phase: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    data = np.load(path)
    if "obs" not in data or "actions" not in data or "dones" not in data:
        raise ValueError("Demo file must contain 'obs', 'actions', and 'dones' arrays")

    file_phase = ""
    if "phase" in data:
        phase_arr = np.asarray(data["phase"]).reshape(-1)
        if phase_arr.size > 0:
            file_phase = str(phase_arr[0]).strip()
            if str(expected_phase).strip().lower() != "auto" and not _phases_are_equivalent(file_phase, expected_phase):
                raise ValueError(
                    f"Demo phase mismatch: file phase='{file_phase}', requested phase='{expected_phase}'"
                )

    obs = np.asarray(data["obs"], dtype=np.float32)
    actions = _resolve_actions_for_ppo(data)
    dones = np.asarray(data["dones"], dtype=bool)
    if obs.ndim != 2:
        raise ValueError(f"Expected obs shape (N,D), got {obs.shape}")
    if dones.ndim != 1:
        raise ValueError(f"Expected dones shape (N,), got {dones.shape}")

    if "op_ko_events" in data:
        op_ko_events = np.asarray(data["op_ko_events"], dtype=bool)
        if op_ko_events.shape[0] != obs.shape[0]:
            op_ko_events = np.zeros(obs.shape[0], dtype=bool)
    else:
        op_ko_events = np.zeros(obs.shape[0], dtype=bool)

    obs, actions, dones = _align_demo_lengths(obs, actions, dones, path=path)
    op_ko_events = op_ko_events[: obs.shape[0]]

    expected = str(expected_phase).strip().lower()
    if expected == "auto":
        resolved_phase = file_phase if file_phase else "locomotion"
    else:
        resolved_phase = str(expected_phase)

    return obs, actions, dones, op_ko_events, resolved_phase


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
    op_ko_events: np.ndarray | None = None,
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

    n_steps = int(obs.shape[0])
    if op_ko_events is not None and op_ko_events.shape[0] == n_steps:
        ko_cumsum = np.cumsum(op_ko_events.astype(np.int32))
    else:
        ko_cumsum = np.zeros(n_steps, dtype=np.int32)

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

                if ko_cumsum[ft] != ko_cumsum[t]:
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
    demos_path, output_path = _resolve_paths(args)
    if args.vecnorm_output:
        print("[BC pretrain] vecnorm-output is ignored (no .pkl files are saved).")

    obs_np, act_np, dones_np, op_ko_np, resolved_phase = _load_dataset(
        demos_path,
        expected_phase=args.phase,
    )
    if str(args.phase).strip().lower() == "auto":
        print(f"[BC pretrain] Auto phase detected from dataset: {resolved_phase}")
        if not args.output.strip():
            output_path = f"train/models/llc_{resolved_phase}_bc_init.zip"

    n_samples_raw = int(obs_np.shape[0])
    if n_samples_raw < 1000:
        print(f"WARNING: only {n_samples_raw} raw samples. BC may underfit.")

    spec = build_phase_spec(resolved_phase, death_penalty=1.0, terminate_on_death=True)
    goal_dim = int(len(spec.feature_names)) if spec.feature_names is not None else int(np.asarray(spec.mask, dtype=np.float32).shape[0])
    expected_obs_dim = int(StateSpec.dim() + (2 * goal_dim))
    if int(obs_np.shape[1]) != expected_obs_dim:
        raise ValueError(
            f"Demo observation dim mismatch for phase='{resolved_phase}': file dim={obs_np.shape[1]}, "
            f"expected dim={expected_obs_dim}. Recollect demos with current curriculum goals or use a matching checkpoint/schema."
        )

    if args.goal_relabel:
        horizons = _parse_horizons(args.relabel_horizons)

        ep_ranges = _episode_ranges(dones_np)
        if ep_ranges:
            ep_lens = [end - start + 1 for start, end in ep_ranges]
            median_ep_len = int(np.median(ep_lens))
            if median_ep_len < max(horizons):
                horizons = [h for h in horizons if h < median_ep_len]
                if not horizons:
                    horizons = [1]
                print(
                    f"[BC pretrain] Adapted relabel horizons to {horizons} "
                    f"(median episode length={median_ep_len})"
                )

        obs_np, act_np, episode_count, relabeled_count = _build_goal_relabel_dataset(
            obs=obs_np,
            actions=act_np,
            dones=dones_np,
            spec=spec,
            horizons=horizons,
            max_relabels_per_step=max(0, int(args.max_relabels_per_step)),
            include_original_samples=bool(args.include_original_samples),
            op_ko_events=op_ko_np,
        )
        print("=" * 68)
        print("GOAL RELABELING ENABLED")
        print(
            f"phase={resolved_phase} episodes={episode_count} raw={n_samples_raw} "
            f"relabeled={relabeled_count} final={obs_np.shape[0]} "
            f"x{(obs_np.shape[0] / max(1, n_samples_raw)):.2f}"
        )
        print(f"horizons={horizons} include_original={args.include_original_samples}")
        print("=" * 68)
    else:
        print("=" * 68)
        print("GOAL RELABELING DISABLED")
        print(f"phase={resolved_phase} samples={n_samples_raw}")
        print("=" * 68)

    n_samples = int(obs_np.shape[0])

    def _make_env_for_ppo():
        return _build_env(args.max_episode_steps, resolved_phase)

    vec_base = VecMonitor(DummyVecEnv([_make_env_for_ppo]))
    vec_env = VecNormalize(vec_base, norm_obs=False, norm_reward=False, clip_obs=10.0)

    ppo_policy_kwargs = dict(
        features_extractor_class=StageGoalFiLMExtractor,
        features_extractor_kwargs=dict(
            goal_feature_names=list(spec.feature_names or []),
            features_dim=256,
        ),
        net_arch=dict(pi=[128], vf=[128]),
    )

    resume_path = str(args.resume).strip()
    if resume_path:
        print(f"[BC pretrain] Resuming PPO policy from {resume_path}")
        try:
            model = PPO.load(
                resume_path,
                env=vec_env,
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
                device=args.device,
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not load resume checkpoint for PPO BC pretrain. "
                "Use a compatible PPO .zip (same action/observation schema) or run without --resume."
            ) from exc
    else:
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
            policy_kwargs=ppo_policy_kwargs,
            verbose=0,
            device=args.device,
        )

    device = model.policy.device
    obs_all = th.as_tensor(obs_np, dtype=th.float32, device=device)
    act_all = th.as_tensor(act_np, dtype=th.long, device=device)

    print("=" * 68)
    print(f"BC PRETRAIN - {resolved_phase.upper()} PPO")
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

            ppo_policy = cast(Any, model).policy
            obs_b = obs_all[idx]
            act_b = act_all[idx]

            dist = ppo_policy.get_distribution(obs_b)
            log_prob = dist.log_prob(act_b)
            entropy = dist.entropy()
            if entropy is None:
                entropy_mean = th.zeros((), dtype=log_prob.dtype, device=log_prob.device)
            else:
                entropy_mean = entropy.mean()

            loss = -log_prob.mean() - float(args.entropy_coef) * entropy_mean

            ppo_policy.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(ppo_policy.parameters(), float(args.max_grad_norm))
            ppo_policy.optimizer.step()

            losses.append(float(loss.detach().cpu().item()))
            entropies.append(float(entropy_mean.detach().cpu().item()))

        with th.no_grad():
            ppo_policy = cast(Any, model).policy
            pred = ppo_policy._predict(obs_all, deterministic=True)
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

    print("Saved BC-initialized PPO checkpoint")
    print(f"  model: {out_model}")
    print("Fine-tune with:")
    print(f"  python -m train.train_curriculum --phase {resolved_phase} --resume {out_model}")


if __name__ == "__main__":
    main()
