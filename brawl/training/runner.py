"""Shared live training/evaluation loop, independent of action-space entry points."""
from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Callable

from brawl.training.run_utils import write_json as _json


@dataclass(frozen=True)
class TrainingHooks:
    """Only backend-specific bookkeeping; optimization belongs to the model."""
    metadata: Callable
    agreement: Callable
    split_episodes: Callable
    retention_factory: Callable | None = None
    monitor_each_update: bool = False


def run_training(model, wrapped, match, args, directory, data=None, *, hooks: TrainingHooks):
    import numpy as np
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import DummyVecEnv
    from brawl.pipelines.final_agent.campaign import FinalCampaign
    metadata, agreement, split_episodes = hooks.metadata, hooks.agreement, hooks.split_episodes

    args.mode = 'training_room'
    vec = DummyVecEnv([lambda: wrapped])
    model.set_env(vec)
    model.set_logger(configure(str(directory / 'ppo'), ['stdout', 'csv']))
    campaign = FinalCampaign(args, match, directory, metadata())
    validation = split_episodes(data, seed=args.seed)[1] if data is not None else None
    started = time.perf_counter()
    retention_guard = None
    if hooks.retention_factory is not None and data is not None:
        retention_guard = hooks.retention_factory()

    class Monitor(BaseCallback):
        def _on_rollout_start(self):
            if data is not None and (hooks.monitor_each_update or self.num_timesteps % args.checkpoint_every < args.n_steps):
                log_agreement()

        def _on_step(self):
            return not args.hours or time.perf_counter() - started < args.hours * 3600

    def log_agreement():
        row = dict(training_step=model.num_timesteps,
                   teacher_agreement=agreement(model, data, validation),
                   interpretation='Offline teacher agreement, not measured live skill retention')
        if retention_guard is not None:
            row['retention'] = retention_guard.observe(row['teacher_agreement'], model.num_timesteps)
        with (directory / 'bc_agreement.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row) + '\n')
        decision = row.get('retention', {})
        if decision.get('action') == 'warn':
            print(f'[retention @{model.num_timesteps}] Warning: {decision["reason"]}; see bc_agreement.jsonl', flush=True)
        if decision.get('action') == 'stop':
            match.release_inputs()
            campaign.save('retention_stop')
            _json(directory / 'retention_stop.json', row)
            raise RuntimeError(f'Primitive retention gate failed: {decision["reason"]} '
                               f'{decision["failures"]}; saved retention_stop.zip before next rollout')

    campaign.init_callback(model)
    try:
        if args.stage == 'evaluate':
            model._last_obs = vec.reset()
            model._last_episode_starts = np.ones((1,), dtype=bool)
            campaign.evaluate(args.eval_episodes)
        else:
            # Diagnostics run before the campaign refreshes live state, so their
            # CPU time cannot leave PPO sampling from a pre-diagnostic snapshot.
            model.learn(args.timesteps, reset_num_timesteps=False, callback=[Monitor(), campaign])
            if hooks.monitor_each_update and data is not None:
                log_agreement()  # The final optimizer update also needs a gate.
            model.logger.dump(step=model.num_timesteps)
            campaign.save('latest')
            if args.eval_every and (not args.hours or time.perf_counter()-started < args.hours*3600):
                campaign.evaluate(args.eval_episodes)
            if data is not None:
                log_agreement()
    except KeyboardInterrupt:
        match.release_inputs()
        campaign.save('interrupted')
        raise
    except Exception:
        match.release_inputs()
        campaign.save('failed_run')
        raise
    finally:
        vec.close()
    from brawl.analysis.plot_final_training import plot_run
    plot_run(directory)
