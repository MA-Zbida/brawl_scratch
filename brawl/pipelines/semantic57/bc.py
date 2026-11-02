"""Mask-aware supervised warm start and episode-held-out agreement diagnostics."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch as th

from brawl.pipelines.semantic57.demos import DemoBatch, split_episodes, validate_rows


@th.no_grad()
def agreement(model, data: DemoBatch, indices: np.ndarray) -> dict:
    model.policy.set_training_mode(False)
    results = {}
    for family in ['all', *sorted(set(data.families[indices]))]:
        selected = indices if family == 'all' else indices[data.families[indices] == family]
        count, nll, correct, entropy = 0, 0., 0, 0.
        for start in range(0, len(selected), 512):
            rows = selected[start:start + 512]
            obs = th.as_tensor(data.obs[rows], device=model.device)
            actions = th.as_tensor(data.actions[rows], device=model.device)
            dist = model.policy.get_distribution(obs)
            nll -= dist.log_prob(actions).sum().item()
            entropy += dist.entropy().sum().item()
            correct += int((dist.get_actions(deterministic=True) == actions).sum())
            count += len(rows)
        if count:
            results[str(family)] = dict(samples=count, nll=nll/count, accuracy=correct/count, entropy=entropy/count)
    return results


def pretrain(model, data: DemoBatch, *, epochs=8, batch_size=256,
             learning_rate=3e-4, entropy_coef=.01, seed=42, output: Path) -> list[dict]:
    validate_rows(data.obs, data.actions)
    if min(epochs, batch_size, learning_rate) <= 0 or entropy_coef < 0:
        raise ValueError('Invalid BC hyperparameters')
    train, validation = split_episodes(data, seed=seed)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'bc_split.json').write_text(json.dumps({
        'seed': seed, 'train_episodes': np.unique(data.episodes[train]).tolist(),
        'validation_episodes': np.unique(data.episodes[validation]).tolist(),
        'interpretation': 'Teacher agreement, not live skill retention or mastery',
    }, indent=2), encoding='utf-8')
    rng = np.random.default_rng(seed)
    th.manual_seed(seed)
    # Separate optimizer: imitation Adam moments do not leak into PPO's optimizer.
    optimizer = th.optim.Adam(model.policy.parameters(), lr=learning_rate)
    rows = []
    for epoch in range(1, epochs + 1):
        model.policy.set_training_mode(True)
        order = rng.permutation(train)
        losses = []
        for start in range(0, len(order), batch_size):
            selected = order[start:start + batch_size]
            obs = th.as_tensor(data.obs[selected], device=model.device)
            actions = th.as_tensor(data.actions[selected], device=model.device)
            dist = model.policy.get_distribution(obs)
            loss = -dist.log_prob(actions).mean() - entropy_coef * dist.entropy().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.policy.parameters(), .5)
            optimizer.step()
            losses.append((loss.item(), len(selected)))
        metrics = agreement(model, data, validation)['all']
        row = dict(epoch=epoch, train_loss=sum(v*n for v, n in losses)/len(train),
                   validation_nll=metrics['nll'], validation_accuracy=metrics['accuracy'],
                   validation_entropy=metrics['entropy'])
        rows.append(row)
        print(f"[BC {epoch}/{epochs}] loss={row['train_loss']:.4f} "
              f"heldout_acc={metrics['accuracy']:.3f} entropy={metrics['entropy']:.3f}", flush=True)
        with (output / 'bc_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
    (output / 'bc_validation.json').write_text(json.dumps(agreement(model, data, validation), indent=2), encoding='utf-8')
    return rows
