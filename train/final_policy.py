"""Build, convert, load, and validate the goal-independent final policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from action_space import ACTION_DIM
from feature_extractor.film_extractor import StageGoalFiLMExtractor
from feature_extractor.final_extractor import FinalStateExtractor
from feature_extractor.memory.state_spec import StateSpec
from train.checkpoint_roles import (
    FINAL_AGENT,
    GOAL_CONDITIONED_SKILL,
    CheckpointRoleError,
    make_checkpoint_metadata,
    metadata_path,
    read_checkpoint_metadata,
    require_checkpoint_role,
    save_typed_checkpoint,
)
from train.policy_transfer import TransferReport, transfer_skill_policy


BASE_OBSERVATION_DIM = StateSpec.observation_dim((2, 4, 8))


class _FinalContractEnv(gym.Env):
    """Headless contract environment used only to materialise converted archives."""

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(BASE_OBSERVATION_DIM,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(ACTION_DIM)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros((BASE_OBSERVATION_DIM,), dtype=np.float32), {}

    def step(self, action):
        return np.zeros((BASE_OBSERVATION_DIM,), dtype=np.float32), 0.0, False, False, {}


def final_policy_kwargs() -> dict[str, Any]:
    return {
        "features_extractor_class": FinalStateExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": {"pi": [128], "vf": [128]},
    }


def build_final_model(
    env: gym.Env,
    *,
    learning_rate: float = 1e-4,
    n_steps: int = 512,
    batch_size: int = 256,
    gamma: float = 0.999,
    gae_lambda: float = 0.95,
    clip_range: float = 0.1,
    ent_coef: float = 0.01,
    n_epochs: int = 4,
    target_kl: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    seed: int = 42,
    device: str = "cpu",
    verbose: int = 1,
) -> PPO:
    shape = getattr(env.observation_space, "shape", None)
    if shape != (BASE_OBSERVATION_DIM,):
        raise ValueError(
            f"final policy requires raw state/history shape {(BASE_OBSERVATION_DIM,)}, got {shape}"
        )
    action_n = getattr(env.action_space, "n", None)
    if int(action_n or 0) != ACTION_DIM:
        raise ValueError(f"final policy requires Discrete({ACTION_DIM}), got {env.action_space}")
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=float(learning_rate),
        n_steps=int(n_steps),
        batch_size=int(batch_size),
        gamma=float(gamma),
        gae_lambda=float(gae_lambda),
        clip_range=float(clip_range),
        ent_coef=float(ent_coef),
        n_epochs=int(n_epochs),
        target_kl=float(target_kl),
        vf_coef=float(vf_coef),
        max_grad_norm=float(max_grad_norm),
        seed=int(seed),
        policy_kwargs=final_policy_kwargs(),
        verbose=int(verbose),
        device=device,
    )


def _load_skill_model(
    checkpoint: str | Path,
    *,
    device: str,
    allow_untyped_skill_checkpoint: bool,
) -> tuple[PPO, dict[str, Any] | None]:
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"skill checkpoint does not exist: {path}")
    metadata: dict[str, Any] | None
    if metadata_path(path).exists():
        metadata = require_checkpoint_role(path, GOAL_CONDITIONED_SKILL)
    elif allow_untyped_skill_checkpoint:
        metadata = None
    else:
        raise CheckpointRoleError(
            f"Legacy untyped skill checkpoint {path} has no role metadata. "
            "Pass --allow-untyped-skill-checkpoint to verify and convert it explicitly."
        )

    model = PPO.load(str(path), device=device)
    extractor = model.policy.features_extractor
    if not isinstance(extractor, StageGoalFiLMExtractor):
        raise CheckpointRoleError(
            f"{path} is not a goal-conditioned skill policy: extractor={type(extractor).__name__}"
        )
    if int(extractor.base_dim) != BASE_OBSERVATION_DIM:
        raise CheckpointRoleError(
            f"{path} state prefix is {extractor.base_dim}, expected {BASE_OBSERVATION_DIM}"
        )
    if getattr(model.action_space, "n", None) != ACTION_DIM:
        raise CheckpointRoleError(f"{path} does not use Discrete({ACTION_DIM})")
    if metadata is not None:
        if int(metadata["observation_dim"]) != int(model.observation_space.shape[0]):
            raise CheckpointRoleError(f"{path} metadata observation_dim does not match the archive")
        if int(metadata["goal_dim"]) != int(extractor.G):
            raise CheckpointRoleError(f"{path} metadata goal_dim does not match the FiLM extractor")
    return model, metadata


def initialize_final_model(
    skill_checkpoint: str | Path,
    env: gym.Env,
    *,
    allow_untyped_skill_checkpoint: bool = False,
    device: str = "cpu",
    **ppo_kwargs: Any,
) -> tuple[PPO, TransferReport, dict[str, Any]]:
    skill, source_metadata = _load_skill_model(
        skill_checkpoint,
        device=device,
        allow_untyped_skill_checkpoint=allow_untyped_skill_checkpoint,
    )
    final = build_final_model(env, device=device, **ppo_kwargs)
    report = transfer_skill_policy(skill.policy, final.policy)
    metadata = make_checkpoint_metadata(
        policy_type=FINAL_AGENT,
        observation_dim=BASE_OBSERVATION_DIM,
        action_dim=ACTION_DIM,
        initialized_from=str(skill_checkpoint),
        extra={
            "initialization": "goal_pretrained_state_and_actor_transfer",
            "source_metadata_status": "typed" if source_metadata is not None else "verified_legacy_untyped",
            "transfer_summary": {
                "transferred_tensors": report.transferred_count,
                "skipped_tensors": report.skipped_count,
            },
        },
    )
    return final, report, metadata


def _write_transfer_report(checkpoint: str | Path, report: TransferReport) -> Path:
    archive = Path(checkpoint)
    report_path = archive.with_name(archive.name + ".transfer.json")
    report_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    return report_path


def convert_skill_checkpoint(
    skill_checkpoint: str | Path,
    output_checkpoint: str | Path,
    *,
    allow_untyped_skill_checkpoint: bool = False,
    device: str = "cpu",
) -> TransferReport:
    env = _FinalContractEnv()
    final, report, metadata = initialize_final_model(
        skill_checkpoint,
        env,
        allow_untyped_skill_checkpoint=allow_untyped_skill_checkpoint,
        device=device,
        n_steps=8,
        batch_size=4,
        verbose=0,
    )
    archive = save_typed_checkpoint(final, output_checkpoint, metadata)
    _write_transfer_report(archive, report)
    env.close()
    return report


def load_final_model(
    checkpoint: str | Path,
    *,
    env: gym.Env | None = None,
    device: str = "cpu",
    **load_kwargs: Any,
) -> PPO:
    path = Path(checkpoint)
    metadata = require_checkpoint_role(path, FINAL_AGENT)
    model = PPO.load(str(path), env=env, device=device, **load_kwargs)
    if not isinstance(model.policy.features_extractor, FinalStateExtractor):
        raise CheckpointRoleError(
            f"{path} declares final_agent but uses "
            f"{type(model.policy.features_extractor).__name__}"
        )
    if model.observation_space.shape != (BASE_OBSERVATION_DIM,):
        raise CheckpointRoleError(
            f"{path} final observation shape is {model.observation_space.shape}, "
            f"expected {(BASE_OBSERVATION_DIM,)}"
        )
    if int(metadata["observation_dim"]) != BASE_OBSERVATION_DIM:
        raise CheckpointRoleError(f"{path} metadata observation contract is stale")
    if getattr(model.action_space, "n", None) != ACTION_DIM:
        raise CheckpointRoleError(f"{path} does not use Discrete({ACTION_DIM})")
    return model
