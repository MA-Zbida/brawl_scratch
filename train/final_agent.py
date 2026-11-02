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
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from env import BrawlDeepEnv, EnvConfig, RUNTIME_CONTRACT
from feature_extractor.memory.state_spec import StateSpec
from reward.match_objective import MatchObjective, NoMatchReward
from train.checkpoint_roles import read_checkpoint_metadata
from train.final_config import FinalAgentConfig, parse_args
from train.final_policy import initialize_final_model, load_final_model
from train.match_env import MatchTrainingEnv, ReplayGymWrapper
from train.final_campaign import FinalCampaign
from world_model.replay_log import WorldReplayWriter


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
        action_values = np.asarray(actions).reshape(-1)
        action_id = int(action_values[0]) if action_values.size else 0
        observations = np.asarray(self.locals.get("new_obs", []))
        state = info.get("terminal_observation")
        if state is None and observations.size:
            state = observations.reshape(-1, observations.shape[-1])[0]
        player_offstage = 0.0
        player_has_weapon = 0.0
        if state is not None:
            player_offstage = StateSpec.get(state, "player_is_offstage")
            player_has_weapon = StateSpec.get(state, "player_has_weapon")
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
                "player_offstage": float(player_offstage),
                "player_has_weapon": float(player_has_weapon),
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
    if cfg.weapon_state_source == "visual_fusion":
        # Catch path typos before environment creation; never substitute an older model.
        checkpoint = Path(cfg.weapon_classifier_checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Weapon classifier checkpoint not found: {checkpoint}")
        print(f"Weapon classifier checkpoint: {checkpoint.resolve()}")
    return BrawlDeepEnv(
        reward_provider=NoMatchReward(),
        config=EnvConfig(
            terminate_on_stock_out=False,
            max_episode_steps=0,
            yolo_infer_every_n_steps=max(1, int(cfg.yolo_every)),
            action_repeat_steps=1,
            action_repeat_min_steps=1,
            action_repeat_max_steps=1,
            tap_latch_steps=1,
            weapon_classifier_enabled=cfg.weapon_state_source == "visual_fusion",
            weapon_classifier_checkpoint=cfg.weapon_classifier_checkpoint,
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
        "n_epochs": cfg.n_epochs,
        "target_kl": cfg.target_kl,
    }


def main() -> None:
    cfg = parse_args()
    # Each invocation keeps its own logs and checkpoint lineage, including resume.
    session = f"{cfg.model_name}_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"
    save_dir = Path(cfg.save_dir) / session
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "config.json").write_text(
        json.dumps(asdict(cfg) | {"reward_version": MatchObjective.VERSION} | RUNTIME_CONTRACT, indent=2),
        encoding="utf-8",
    )
    print(f"Run directory: {save_dir}")
    print(f"Mode: {cfg.mode} | policy device: {cfg.device} | budget per block: {cfg.timesteps:,}")
    print(f"Weapon state source: {cfg.weapon_state_source} (policy device remains {cfg.device})")
    print(f"Runtime contracts: {RUNTIME_CONTRACT}; old checkpoints retain weights, not old input/HUD behaviour.")
    print(f"Starting in {cfg.delay:.0f}s - switch to Brawlhalla with attacking Easy Mordex")
    time.sleep(max(0., cfg.delay))
    match = MatchTrainingEnv(
        _make_env(cfg), mode=cfg.mode, max_steps=cfg.max_episode_steps,
        matches_path=save_dir / "windows.csv",
        steps_path=save_dir / "transitions.csv" if cfg.log_csv else None)
    env = match
    if cfg.world_replay:
        writer = WorldReplayWriter(
            cfg.world_replay, phase="final_agent",
            metadata=asdict(cfg) | {"reward_version": MatchObjective.VERSION,
                                   "run_directory": str(save_dir)} | RUNTIME_CONTRACT)
        env = ReplayGymWrapper(match, writer)
        print(f"World replay: {writer.dir}")
    vec_env = DummyVecEnv([lambda: env])
    campaign = None
    try:
        if cfg.init_from:
            model, report, metadata = initialize_final_model(
                cfg.init_from, vec_env, device=cfg.device,
                allow_untyped_skill_checkpoint=cfg.allow_untyped_skill_checkpoint,
                **_ppo_overrides(cfg))
            (save_dir / "transfer.json").write_text(
                json.dumps(report.to_dict(), indent=2), encoding="utf-8")
            print(report.format())
        else:
            model = load_final_model(cfg.resume, env=vec_env, device=cfg.device,
                                     **_ppo_overrides(cfg))
            metadata = read_checkpoint_metadata(cfg.resume) | {"resumed_from": cfg.resume}
        model.set_logger(configure(str(save_dir / "ppo"), ["stdout", "csv"]))
        campaign = FinalCampaign(cfg, match, save_dir,
                                 metadata | {"reward_version": MatchObjective.VERSION} | RUNTIME_CONTRACT)
        campaign.init_callback(model)
        campaign.save("initializer" if cfg.init_from else "resume_start")
        while cfg.timesteps > 0:
            model.learn(
                total_timesteps=cfg.timesteps, reset_num_timesteps=False,
                progress_bar=True, callback=campaign,
            )
            # SB3 dumps before each update; explicitly retain the final update's
            # KL, entropy and losses even when this is only a one-rollout pilot.
            model.logger.dump(step=model.num_timesteps)
            campaign.save("latest")
            if not cfg.forever:
                break
        if cfg.timesteps and cfg.eval_every:
            campaign.evaluate(cfg.eval_episodes)
    except KeyboardInterrupt:
        print("\nInterrupted. Saving latest policy and flushing replay.")
    finally:
        try:
            match.release_inputs()
            if campaign is not None:
                campaign.save("latest")
        finally:
            try:
                vec_env.close()
            finally:
                if campaign is not None:
                    model.logger.close()
    print(f"Saved final policy: {save_dir / 'latest.zip'}")
    from analysis.plot_final_training import plot_run
    output = plot_run(save_dir)
    if output:
        print(f"Progress figure: {output}")


if __name__ == "__main__":
    main()
