"""Timed frozen-policy evaluation and live-rollout boundary handling."""
from __future__ import annotations

import json
import time
import numpy as np

from brawl.pipelines.round_a.models import Predictor, save_model
from brawl.training.run_utils import write_json


class SkillMetrics:
    """Observed opportunities, not privileged game-state success labels.

    Death closes a recovery as failure BEFORE grounding is inspected, so a
    respawn can never be counted as a successful return. Missing identity and
    block boundaries censor open events instead of inventing outcomes.
    """
    def __init__(self):
        self.offstage = self.weapon = False
        self.land_count = 0
        self.counts = dict(offstage_events=0, recoveries=0, offstage_deaths=0,
            pickup_opportunities=0, armed_after_opportunity=0,
            eligible_unarmed_seconds=0., uncertain_seconds=0., censored_recoveries=0,
            censored_pickup_opportunities=0)

    def observe(self, obs, info, seconds):
        from brawl.state.state_spec import StateSpec
        if isinstance(obs, dict): obs = obs['structured']
        value = lambda name: StateSpec.get(obs, name)
        death = float(info.get('self_stock_lost_step', 0.)) > .5
        if death:
            self.counts['offstage_deaths'] += int(self.offstage)
            self.offstage = self.weapon = False
            self.land_count = 0
            return
        if value('identity_observed') <= .5 or value('player_missing_frames') > 0:
            self.counts['uncertain_seconds'] += seconds
            self.counts['censored_recoveries'] += int(self.offstage)
            self.counts['censored_pickup_opportunities'] += int(self.weapon)
            self.offstage = self.weapon = False
            self.land_count = 0
            return
        offstage = value('player_is_offstage') > .5
        if offstage and not self.offstage:
            self.offstage = True
            self.counts['offstage_events'] += 1
        self.land_count = self.land_count + 1 if not offstage and value('player_grounded') > .5 else 0
        if self.offstage and self.land_count >= 2:
            self.counts['recoveries'] += 1
            self.offstage = False
        armed = value('player_has_weapon') > .5
        eligible = not armed and value('weapon_on_ground') > .5 and not offstage
        if armed and self.weapon:
            self.counts['armed_after_opportunity'] += 1
            self.weapon = False
        if eligible:
            self.counts['eligible_unarmed_seconds'] += seconds
            if not self.weapon:
                self.counts['pickup_opportunities'] += 1
                self.weapon = True
        elif not armed and self.weapon:
            self.counts['censored_pickup_opportunities'] += 1
            self.weapon = False

    def row(self):
        result = dict(self.counts)
        result['censored_recoveries'] += int(self.offstage)
        result['censored_pickup_opportunities'] += int(self.weapon)
        return result


def evaluate_block(env, predictor, obs, seconds, *, clock=time.perf_counter, clip=None):
    if seconds <= 0 or not np.isfinite(seconds):
        raise ValueError('Evaluation seconds must be finite and positive')
    began = clock()
    row = dict(interactions=0, damage_dealt=0., damage_received=0., stocks_taken=0., stocks_lost=0.)
    metrics = SkillMetrics()
    start = True
    previous_time = began
    try:
        while clock() - began < seconds:
            action = predictor.predict(obs, start=start)
            obs, _, terminated, truncated, info = env.step(action)
            now = clock()
            if clip is not None:
                clip.write(env.unwrapped._last_frame, elapsed=now-began, action=action, info=info)
            metrics.observe(obs, info, now-previous_time)
            previous_time = now
            row['interactions'] += 1
            for key, source in [('damage_dealt', 'op_delta_damage'), ('damage_received', 'self_delta_damage'),
                                ('stocks_taken', 'op_stock_lost_step'), ('stocks_lost', 'self_stock_lost_step')]:
                row[key] += float(info.get(source, 0.))
            start = bool(terminated or truncated)
            if start:
                obs, _ = env.reset()
    finally:
        env.release_inputs()
        if clip is not None:
            clip.close()
    row['seconds'] = clock() - began
    row['net_stocks_per_minute'] = 60 * (row['stocks_taken'] - row['stocks_lost']) / max(row['seconds'], 1e-6)
    row['damage_trade'] = row['damage_dealt'] - row['damage_received']
    row['mode'] = 'training_room_stock_exchanges_not_match_wins'
    row.update(metrics.row())
    return row, obs


def evaluate(model, adapter, match, seconds, block_seconds, directory, label, *, video_seconds=0., run_started=None):
    match.usage = 'update_gap'
    match.force_boundary = True
    obs, _, ended, truncated, _ = adapter.step(0)
    if ended or truncated:
        obs, _ = adapter.reset()
    match.usage = match._window_usage = 'evaluation'
    rows, remaining = [], seconds
    try:
        while remaining > 1e-6:
            # Block boundaries reset policy memory explicitly; no physical reset
            # is claimed. All arms see the same timed block protocol.
            clip = None
            if video_seconds:
                from brawl.pipelines.round_a.video import EvaluationClip
                clip = EvaluationClip(directory, f'{label}_{model.num_timesteps}_{len(rows)}',
                                      seconds=video_seconds, fps=1/adapter.period if adapter.period else 60.)
            row, obs = evaluate_block(adapter, Predictor(model), obs, min(remaining, block_seconds), clip=clip)
            row.update(label=label, training_step=int(model.num_timesteps), block=len(rows))
            row['elapsed_run_seconds'] = time.perf_counter()-run_started if run_started is not None else None
            row['cadence'] = adapter.cadence_report()
            rows.append(row)
            remaining -= row['seconds']
            with (directory / 'evaluations.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row) + '\n')
            print(f'[{label}] stocks {row["stocks_taken"]:.0f}-{row["stocks_lost"]:.0f} '
                  f'trade={row["damage_trade"]:+.1f} seconds={row["seconds"]:.1f}', flush=True)
    finally:
        match.usage = 'train'
        adapter.release_inputs()
    return rows


def run_training(model, adapter, match, protocol, directory, contract, *, data=None, video_seconds=0.):
    import torch as th
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import DummyVecEnv
    from brawl.pipelines.round_a.bc import evaluate_imitation
    model.set_env(DummyVecEnv([lambda: adapter]))
    model.set_logger(configure(str(directory / 'ppo'), ['stdout', 'csv']))
    start_step = int(model.num_timesteps)
    if start_step % protocol.n_steps:
        raise ValueError('Resume a completed-rollout checkpoint; partial-rollout interactions cannot be reconstructed')
    remaining = protocol.rollout_budget - start_step
    if remaining <= 0:
        raise ValueError('Training budget already exhausted; use --stage evaluate')
    began = time.perf_counter()

    class Campaign(BaseCallback):
        baseline = False
        gap_started = None
        gap_seconds = 0.
        next_eval = (start_step // protocol.eval_every + 1) * protocol.eval_every
        next_save = (start_step // protocol.checkpoint_every + 1) * protocol.checkpoint_every

        def refresh(self):
            # The game runs during updates. This neutral bridge is logged, but
            # never inserted into PPO or charged to its learning interaction budget.
            match.usage = 'update_gap'
            match.force_boundary = True
            obs, _, _, _ = self.training_env.step(np.array([0]))
            self.model._last_obs = obs
            self.model._last_episode_starts = np.ones(1, dtype=bool)
            if hasattr(self.model, '_last_lstm_states'):
                states = self.model._last_lstm_states
                self.model._last_lstm_states = type(states)(
                    tuple(th.zeros_like(x) for x in states.pi), tuple(th.zeros_like(x) for x in states.vf))
            match.usage = match._window_usage = 'train'

        def save(self, name):
            save_model(self.model, directory / f'{name}.zip', contract,
                       optimizer_gap_seconds=self.gap_seconds,
                       training_interactions_this_run=int(self.model.num_timesteps)-start_step)

        def imitation(self):
            if data is not None:
                report = evaluate_imitation(self.model, data, data.split(protocol.split_seed)['validation'],
                                            sequence_length=protocol.sequence_length)
                with (directory / 'retention.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(dict(training_step=int(self.model.num_timesteps),
                        validation=report, action='report_only', interpretation='Imitation, not live mastery')) + '\n')

        def _on_rollout_start(self):
            if self.gap_started is not None:
                self.gap_seconds += time.perf_counter() - self.gap_started
                self.gap_started = None
            step = int(self.model.num_timesteps)
            if not self.baseline:
                self.save('initializer')
                self.imitation()
                evaluate(self.model, adapter, match, protocol.baseline_seconds,
                         protocol.eval_block_seconds, directory, 'baseline' if start_step == 0 else 'resume_baseline',
                         video_seconds=video_seconds, run_started=began)
                self.baseline = True
            elif step >= self.next_eval:
                self.imitation()
                evaluate(self.model, adapter, match, protocol.interim_seconds,
                         protocol.eval_block_seconds, directory, 'interim', video_seconds=video_seconds, run_started=began)
                while self.next_eval <= step:
                    self.next_eval += protocol.eval_every
            if step >= self.next_save:
                self.save(f'step_{step:09d}')
                self.save('latest')
                while self.next_save <= step:
                    self.next_save += protocol.checkpoint_every
            adapter.release_inputs()
            self.refresh()

        def _on_rollout_end(self):
            adapter.release_inputs()
            self.gap_started = time.perf_counter()

        def _on_step(self):
            match.policy_version = int(self.model.num_timesteps // protocol.n_steps)
            return True

    campaign = Campaign()
    status = 'failed'
    try:
        model.learn(remaining, reset_num_timesteps=False, callback=campaign)
        if campaign.gap_started is not None:
            campaign.gap_seconds += time.perf_counter()-campaign.gap_started
            campaign.gap_started = None
        campaign.save('latest')
        campaign.imitation()
        evaluate(model, adapter, match, protocol.final_seconds, protocol.eval_block_seconds, directory, 'final',
                 video_seconds=video_seconds, run_started=began)
        status = 'completed'
    except BaseException:
        adapter.release_inputs()
        campaign.save('interrupted_or_failed')
        raise
    finally:
        adapter.release_inputs()
        write_json(directory / 'training_summary.json', dict(status=status,
            requested_interactions=protocol.timesteps, rollout_budget=protocol.rollout_budget,
            actual_training_interactions=int(model.num_timesteps)-start_step,
            cumulative_training_interactions=int(model.num_timesteps),
            total_environment_interactions=match.total_interactions,
            cadence=adapter.cadence_report(),
            elapsed_seconds=time.perf_counter()-began, optimizer_gap_seconds=campaign.gap_seconds))
