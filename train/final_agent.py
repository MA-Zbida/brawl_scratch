#!/usr/bin/env python
"""Train one goal-independent policy directly on full-match reward."""

from __future__ import annotations

# DXGI duplication must be acquired before stable_baselines3 imports torch.
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import capture_first  # noqa: E402,F401

import csv
import json
import time
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from env import BrawlDeepEnv, EnvConfig
from train.checkpoint_roles import read_checkpoint_metadata, save_typed_checkpoint
from train.final_config import FinalAgentConfig, parse_args
from train.final_policy import initialize_final_model, load_final_model


class FinalTrainingMetricsCallback(BaseCallback):
    """Write match-oriented evidence without introducing a goal contract."""

    def __init__(self, path: Path, *, enabled: bool = True) -> None:
        super().__init__(verbose=0)
        self.path = path
        self.enabled = bool(enabled)
        self._file = None
        self._writer: csv.DictWriter | None = None
        self._started = 0.0

    def _on_training_start(self) -> None:
        self._started = time.perf_counter()
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="", encoding="utf-8")
        fields = [
            "environment_step",
            "interaction_seconds",
            "reward",
            "action_id",
            "damage_dealt",
            "damage_received",
            "op_stock_lost",
            "self_stock_lost",
            "player_offstage",
            "player_has_weapon",
        ]
        self._writer = csv.DictWriter(self._file, fieldnames=fields)
        self._writer.writeheader()

    def _on_step(self) -> bool:
        if self._writer is None:
            return True
        infos = self.locals.get("infos", [])
        info = infos[0] if infos else {}
        rewards = self.locals.get("rewards", [0.0])
        actions = self.locals.get("actions", [0])
        action_id = int(actions[0]) if len(actions) else 0
        self._writer.writerow(
            {
                "environment_step": int(self.num_timesteps),
                "interaction_seconds": round(time.perf_counter() - self._started, 6),
                "reward": float(rewards[0]) if len(rewards) else 0.0,
                "action_id": action_id,
                "damage_dealt": float(info.get("op_delta_damage", 0.0)),
                "damage_received": float(info.get("self_delta_damage", 0.0)),
                "op_stock_lost": float(info.get("op_stock_lost_step", 0.0)),
                "self_stock_lost": float(info.get("self_stock_lost_step", 0.0)),
                "player_offstage": float(info.get("player_is_offstage", 0.0)),
                "player_has_weapon": float(info.get("player_has_weapon", 0.0)),
            }
        )
        return True

    def _on_training_end(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            self._writer = None


def _make_env(cfg: FinalAgentConfig) -> BrawlDeepEnv:
    return BrawlDeepEnv(
        config=EnvConfig(
            terminate_on_stock_out=True,
            max_episode_steps=int(cfg.max_episode_steps),
            yolo_infer_every_n_steps=max(1, int(cfg.yolo_every)),
            action_repeat_steps=1,
            action_repeat_min_steps=1,
            action_repeat_max_steps=1,
            tap_latch_steps=1,
        )
    )


def _ppo_overrides(cfg: FinalAgentConfig) -> dict[str, Any]:
    return {
        "learning_rate": cfg.learning_rate,
        "n_steps": cfg.n_steps,
        "batch_size": cfg.batch_size,
        "gamma": cfg.gamma,
        "gae_lambda": cfg.gae_lambda,
        "clip_range": cfg.clip_range,
        "ent_coef": cfg.ent_coef,
        "vf_coef": cfg.vf_coef,
        "max_grad_norm": cfg.max_grad_norm,
        "seed": cfg.seed,
    }


def main() -> None:
    cfg = parse_args()
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = save_dir / f"{cfg.model_name}.zip"

    vec_env = VecMonitor(DummyVecEnv([lambda: _make_env(cfg)]))
    transfer_report = None
    if cfg.init_from:
        model, transfer_report, metadata = initialize_final_model(
            cfg.init_from,
            vec_env,
            allow_untyped_skill_checkpoint=cfg.allow_untyped_skill_checkpoint,
            device=cfg.device,
            **_ppo_overrides(cfg),
        )
        print(transfer_report.format())
    else:
        assert cfg.resume is not None
        model = load_final_model(
            cfg.resume,
            env=vec_env,
            device=cfg.device,
            **_ppo_overrides(cfg),
        )
        metadata = {
            **read_checkpoint_metadata(cfg.resume),
            "resumed_from": str(cfg.resume),
        }

    print("=" * 70)
    print("FINAL AGENT TRAINING -- one policy, raw state/history, match reward")
    print(f"observation: {model.observation_space.shape} (no goal target or mask)")
    print(f"action space: {model.action_space}")
    print(f"device: {cfg.device} | timesteps: {cfg.timesteps:,}")
    print("=" * 70)

    metrics = FinalTrainingMetricsCallback(
        save_dir / f"{cfg.model_name}_steps.csv",
        enabled=cfg.log_csv,
    )
    interrupted = False
    if cfg.timesteps > 0:
        print(f"Starting in {cfg.delay:.0f}s - switch to Brawlhalla")
        time.sleep(max(0.0, cfg.delay))
        try:
            model.learn(
                total_timesteps=int(cfg.timesteps),
                progress_bar=True,
                callback=metrics,
            )
        except KeyboardInterrupt:
            interrupted = True
            print("\nInterrupted; saving the current final policy.")
        finally:
            metrics._on_training_end()

    metadata = {
        **metadata,
        "training_stage": "final_agent",
        "objective": "match",
        "environment_steps": int(getattr(model, "num_timesteps", 0)),
    }
    archive = save_typed_checkpoint(model, checkpoint, metadata)
    if transfer_report is not None:
        report_path = archive.with_name(archive.name + ".transfer.json")
        report_path.write_text(
            json.dumps(transfer_report.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Transfer report: {report_path}")
    vec_env.close()
    print(f"Saved final-agent checkpoint: {archive}")
    if interrupted:
        print("Checkpoint records the interrupted environment-step count.")


if __name__ == "__main__":
    main()
