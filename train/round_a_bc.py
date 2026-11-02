"""One episode split, equal supervised exposure, ordered masked BC for all arms."""
from __future__ import annotations

import json
import numpy as np
import torch as th

from train.round_a_models import distribution
from train.primitive_training import gate_predictions
from train.semantic_demos import DemoBatch
from train.run_utils import write_json


def evaluate_imitation(model, data, ids, *, sequence_length=32):
    observations, actions, families, drills, episodes, predictions = [], [], [], [], [], []
    model.policy.set_training_mode(False)
    with th.no_grad():
        for ep_id in ids:
            entry = data.manifest['episodes'][ep_id]
            episode, state = data.episode(entry), None
            for begin in range(0, entry['rows'], sequence_length):
                idx = np.arange(begin, min(begin + sequence_length, entry['rows']))
                obs = data.observations(episode, idx, model.round_a_candidate)
                dist, state = distribution(model, obs, state, start=begin == 0)
                valid = episode['valid'][idx]
                predictions.extend(dist.get_actions(deterministic=True).cpu().numpy()[valid])
                observations.extend(episode['obs'][idx][valid])
                actions.extend(episode['actions'][idx][valid])
                families.extend(episode['families'][idx][valid])
                drills.extend([entry['drill']] * int(valid.sum()))
                episodes.extend([ep_id] * int(valid.sum()))
    selected = DemoBatch(np.asarray(observations, np.float32).reshape(-1, 146),
        np.asarray(actions, np.int64), np.asarray(episodes), np.asarray(families), np.asarray(drills))
    report = gate_predictions(selected, predictions)
    report['overall_accuracy'] = float(np.mean(np.asarray(predictions) == selected.actions)) if actions else None
    return report


def pretrain(model, data, protocol, directory, *, seed):
    split = data.split(protocol.split_seed)
    write_json(directory / 'bc_split.json', split)
    rng = np.random.default_rng(seed)
    th.manual_seed(seed)
    counts = np.zeros(27, np.int64)
    for ep_id in split['train']:
        episode = data.episode(data.manifest['episodes'][ep_id])
        counts += np.bincount(episode['actions'][episode['valid']], minlength=27)
    # Same per-label weighting for every architecture, without shuffling time
    # inside recurrent sequences. No validation/test frequencies enter this loss.
    weights = np.minimum(8., np.sqrt(max(1, counts.max()) / np.maximum(counts, 1)))
    optimizer = th.optim.Adam(model.policy.parameters(), lr=protocol.bc_learning_rate)
    for epoch in range(protocol.bc_epochs):
        model.policy.set_training_mode(True)
        losses, labels = [], 0
        for ep_id in rng.permutation(split['train']):
            entry = data.manifest['episodes'][ep_id]
            episode, state = data.episode(entry), None
            for begin in range(0, entry['rows'], protocol.sequence_length):
                idx = np.arange(begin, min(entry['rows'], begin + protocol.sequence_length))
                observations = data.observations(episode, idx, model.round_a_candidate)
                dist, state = distribution(model, observations, state, start=begin == 0)
                mask = th.as_tensor(np.array(episode['valid'][idx]), device=model.device)
                if mask.any():
                    actions = th.as_tensor(np.array(episode['actions'][idx]), device=model.device)
                    w = th.as_tensor(weights, device=model.device, dtype=th.float32)[actions][mask]
                    loss = -(dist.log_prob(actions)[mask] * w).sum() / w.sum()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    th.nn.utils.clip_grad_norm_(model.policy.parameters(), .5)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                    labels += int(mask.sum())
                if state is not None:
                    state = tuple(x.detach() for x in state)
        report = evaluate_imitation(model, data, split['validation'], sequence_length=protocol.sequence_length)
        row = dict(epoch=epoch+1, loss=float(np.mean(losses)), labels=labels, validation=report)
        with (directory / 'bc_metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row) + '\n')
        print(f'[BC {epoch+1}/{protocol.bc_epochs}] loss={row["loss"]:.4f} agreement={report["overall_accuracy"]:.3f}', flush=True)
    # Fixed final epoch, not repeated selection on the held-out test set.
    write_json(directory / 'bc_validation.json', report)
    return report
