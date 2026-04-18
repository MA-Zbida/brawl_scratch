from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.distributions import Distribution as TorchDistribution
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance


@dataclass
class _PPOSamples:
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class _TensorReplayStore:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(max(1, capacity))
        self.observations: Optional[th.Tensor] = None
        self.actions: Optional[th.Tensor] = None
        self.old_values: Optional[th.Tensor] = None
        self.old_log_prob: Optional[th.Tensor] = None
        self.advantages: Optional[th.Tensor] = None
        self.returns: Optional[th.Tensor] = None

    @property
    def size(self) -> int:
        if self.observations is None:
            return 0
        return int(self.observations.shape[0])

    @staticmethod
    def _flatten_time_env(array: np.ndarray, n_envs: int) -> np.ndarray:
        if array.ndim < 2:
            return array.reshape(-1)
        if int(array.shape[1]) == int(n_envs):
            return array.reshape((-1,) + array.shape[2:])
        return array.reshape((array.shape[0],) + array.shape[1:])

    def _append(self, key: str, value: th.Tensor) -> None:
        existing = getattr(self, key)
        value_cpu = value.detach().cpu()
        if existing is None:
            setattr(self, key, value_cpu)
            return
        setattr(self, key, th.cat((existing, value_cpu), dim=0))

    def _trim(self) -> None:
        extra = self.size - self.capacity
        if extra <= 0:
            return
        for key in (
            "observations",
            "actions",
            "old_values",
            "old_log_prob",
            "advantages",
            "returns",
        ):
            value = getattr(self, key)
            if value is not None:
                setattr(self, key, value[extra:])

    def add_from_rollout_buffer(self, rollout_buffer: Any) -> None:
        obs_raw = getattr(rollout_buffer, "observations", None)
        if obs_raw is None or isinstance(obs_raw, dict):
            return

        actions_raw = getattr(rollout_buffer, "actions", None)
        values_raw = getattr(rollout_buffer, "values", None)
        log_probs_raw = getattr(rollout_buffer, "log_probs", None)
        advantages_raw = getattr(rollout_buffer, "advantages", None)
        returns_raw = getattr(rollout_buffer, "returns", None)

        if any(v is None for v in (actions_raw, values_raw, log_probs_raw, advantages_raw, returns_raw)):
            return

        n_envs = int(getattr(rollout_buffer, "n_envs", 1))

        obs_np = np.asarray(obs_raw)
        actions_np = np.asarray(actions_raw)
        values_np = np.asarray(values_raw)
        log_probs_np = np.asarray(log_probs_raw)
        advantages_np = np.asarray(advantages_raw)
        returns_np = np.asarray(returns_raw)

        obs_flat = self._flatten_time_env(obs_np, n_envs)
        actions_flat = self._flatten_time_env(actions_np, n_envs)
        values_flat = self._flatten_time_env(values_np, n_envs).reshape(-1)
        log_probs_flat = self._flatten_time_env(log_probs_np, n_envs).reshape(-1)
        advantages_flat = self._flatten_time_env(advantages_np, n_envs).reshape(-1)
        returns_flat = self._flatten_time_env(returns_np, n_envs).reshape(-1)

        lengths = [
            int(obs_flat.shape[0]),
            int(actions_flat.shape[0]),
            int(values_flat.shape[0]),
            int(log_probs_flat.shape[0]),
            int(advantages_flat.shape[0]),
            int(returns_flat.shape[0]),
        ]
        n = int(min(lengths))
        if n <= 0:
            return

        obs_flat = obs_flat[:n]
        actions_flat = actions_flat[:n]
        values_flat = values_flat[:n]
        log_probs_flat = log_probs_flat[:n]
        advantages_flat = advantages_flat[:n]
        returns_flat = returns_flat[:n]

        obs_tensor = th.as_tensor(np.array(obs_flat, copy=True), dtype=th.float32)
        actions_tensor = th.as_tensor(np.array(actions_flat, copy=True))
        values_tensor = th.as_tensor(np.array(values_flat, copy=True), dtype=th.float32)
        log_probs_tensor = th.as_tensor(np.array(log_probs_flat, copy=True), dtype=th.float32)
        advantages_tensor = th.as_tensor(np.array(advantages_flat, copy=True), dtype=th.float32)
        returns_tensor = th.as_tensor(np.array(returns_flat, copy=True), dtype=th.float32)

        self._append("observations", obs_tensor)
        self._append("actions", actions_tensor)
        self._append("old_values", values_tensor)
        self._append("old_log_prob", log_probs_tensor)
        self._append("advantages", advantages_tensor)
        self._append("returns", returns_tensor)
        self._trim()

    def sample(self, batch_size: int, device: th.device, allow_oversample: bool = False) -> Optional[_PPOSamples]:
        if self.size <= 0:
            return None

        if allow_oversample:
            take = int(max(1, batch_size))
        else:
            take = int(max(1, min(batch_size, self.size)))
        indices = th.randint(0, self.size, (take,), device=th.device("cpu"))

        assert self.observations is not None
        assert self.actions is not None
        assert self.old_values is not None
        assert self.old_log_prob is not None
        assert self.advantages is not None
        assert self.returns is not None

        return _PPOSamples(
            observations=self.observations[indices].to(device),
            actions=self.actions[indices].to(device),
            old_values=self.old_values[indices].to(device),
            old_log_prob=self.old_log_prob[indices].to(device),
            advantages=self.advantages[indices].to(device),
            returns=self.returns[indices].to(device),
        )


class _BehaviorCloneStore:
    def __init__(self, npz_path: str, action_space: spaces.Space, expected_obs_dim: int) -> None:
        self.path = Path(npz_path)
        self.enabled = False
        self.disable_reason = ""
        self.observations: Optional[th.Tensor] = None
        self.actions: Optional[th.Tensor] = None

        if not self.path.exists():
            self.disable_reason = f"dataset not found: {self.path}"
            return

        try:
            with np.load(self.path, allow_pickle=False) as data:
                if "obs" not in data.files:
                    self.disable_reason = f"missing 'obs' in {self.path}"
                    return
                if "actions" not in data.files and "actions_discrete" not in data.files and "actions_multidiscrete" not in data.files:
                    self.disable_reason = f"missing actions arrays in {self.path}"
                    return

                obs_np = np.asarray(data["obs"], dtype=np.float32)
                if obs_np.ndim != 2:
                    self.disable_reason = f"expected obs shape [N, D], got {obs_np.shape}"
                    return
                if expected_obs_dim > 0 and int(obs_np.shape[1]) != expected_obs_dim:
                    self.disable_reason = (
                        f"obs dim mismatch for {self.path}: demos={int(obs_np.shape[1])}, model={expected_obs_dim}"
                    )
                    return

                actions_np: np.ndarray
                if isinstance(action_space, spaces.Discrete):
                    if "actions_discrete" in data.files:
                        actions_np = np.asarray(data["actions_discrete"], dtype=np.int64).reshape(-1)
                    else:
                        raw = np.asarray(data["actions"], dtype=np.int64)
                        if raw.ndim > 1:
                            raw = raw[:, 0]
                        actions_np = raw.reshape(-1)
                elif isinstance(action_space, spaces.MultiDiscrete):
                    if "actions_multidiscrete" in data.files:
                        raw = np.asarray(data["actions_multidiscrete"], dtype=np.int64)
                    else:
                        raw = np.asarray(data["actions"], dtype=np.int64)
                    if raw.ndim == 1:
                        raw = raw.reshape(-1, 1)
                    expected_dims = int(len(action_space.nvec))
                    if raw.ndim != 2 or int(raw.shape[1]) != expected_dims:
                        self.disable_reason = (
                            f"expected multidiscrete actions [N,{expected_dims}], got {raw.shape}"
                        )
                        return
                    actions_np = raw
                else:
                    raw = np.asarray(data["actions"], dtype=np.float32)
                    actions_np = raw.reshape(raw.shape[0], -1)

                n = int(min(obs_np.shape[0], actions_np.shape[0]))
                if n <= 0:
                    self.disable_reason = f"empty dataset after alignment: {self.path}"
                    return

                obs_np = obs_np[:n]
                actions_np = actions_np[:n]

                self.observations = th.as_tensor(np.array(obs_np, copy=True), dtype=th.float32)
                if isinstance(action_space, (spaces.Discrete, spaces.MultiDiscrete)):
                    self.actions = th.as_tensor(np.array(actions_np, copy=True), dtype=th.long)
                else:
                    self.actions = th.as_tensor(np.array(actions_np, copy=True), dtype=th.float32)
                self.enabled = True
        except Exception as exc:
            self.disable_reason = f"failed to load {self.path}: {exc}"

    @property
    def size(self) -> int:
        if not self.enabled or self.observations is None:
            return 0
        return int(self.observations.shape[0])

    def sample(self, batch_size: int, device: th.device) -> Optional[tuple[th.Tensor, th.Tensor]]:
        if self.size <= 0 or self.observations is None or self.actions is None:
            return None
        take = int(max(1, min(batch_size, self.size)))
        indices = th.randint(0, self.size, (take,), device=th.device("cpu"))
        return self.observations[indices].to(device), self.actions[indices].to(device)


class AnchoredReplayPPO(PPO):
    """PPO variant with fixed replay mixing, anchor KL and optional BC imitation loss."""

    def __init__(
        self,
        *args,
        replay_ratio: float = 0.30,
        replay_capacity: int = 262_144,
        replay_warmup_updates: int = 1,
        strict_replay_mix: bool = True,
        normalize_advantage_per_source: bool = True,
        anchor_kl_coef: float = 0.02,
        anchor_update_interval: int = 8,
        bc_loss_coef: float = 0.05,
        bc_batch_size: int = 128,
        bc_demos_path: Optional[str] = None,
        enable_pcgrad: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.replay_ratio = float(np.clip(replay_ratio, 0.0, 0.95))
        self.replay_capacity = int(max(1, replay_capacity))
        self.replay_warmup_updates = int(max(0, replay_warmup_updates))
        self.strict_replay_mix = bool(strict_replay_mix)
        self.normalize_advantage_per_source = bool(normalize_advantage_per_source)

        self.anchor_kl_coef = float(max(0.0, anchor_kl_coef))
        self.anchor_update_interval = int(max(1, anchor_update_interval))

        self.bc_loss_coef = float(max(0.0, bc_loss_coef))
        self.bc_batch_size = int(max(1, bc_batch_size))
        self.bc_demos_path = bc_demos_path

        self.enable_pcgrad = bool(enable_pcgrad)

        self._replay_store = _TensorReplayStore(self.replay_capacity)
        self._anchor_policy: Optional[Any] = None
        self._updates_since_anchor = 0
        self._train_calls = 0
        self._bc_warning_printed = False

        self._bc_store: Optional[_BehaviorCloneStore] = None
        obs_shape = getattr(self.observation_space, "shape", None)
        expected_obs_dim = int(obs_shape[0]) if obs_shape is not None and len(obs_shape) > 0 else -1
        if self.bc_loss_coef > 0.0 and self.bc_demos_path:
            self._bc_store = _BehaviorCloneStore(self.bc_demos_path, self.action_space, expected_obs_dim)

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["_replay_store", "_anchor_policy", "_bc_store"]

    @staticmethod
    def _normalize_advantages(values: th.Tensor) -> th.Tensor:
        if values.numel() <= 1:
            return values
        return (values - values.mean()) / (values.std() + 1e-8)

    @staticmethod
    def _to_batch(rollout_data: Any) -> _PPOSamples:
        return _PPOSamples(
            observations=rollout_data.observations,
            actions=rollout_data.actions,
            old_values=rollout_data.old_values.flatten(),
            old_log_prob=rollout_data.old_log_prob.flatten(),
            advantages=rollout_data.advantages.flatten(),
            returns=rollout_data.returns.flatten(),
        )

    def _refresh_anchor_policy(self) -> None:
        if self.anchor_kl_coef <= 0.0:
            return
        policy_cls = self.policy.__class__
        constructor_kwargs = self.policy._get_constructor_parameters()
        anchor = policy_cls(**constructor_kwargs)
        anchor.load_state_dict(self.policy.state_dict())
        anchor.to(self.device)
        anchor.set_training_mode(False)
        for parameter in anchor.parameters():
            parameter.requires_grad_(False)
        self._anchor_policy = anchor
        self._updates_since_anchor = 0

    def _maybe_init_anchor(self) -> None:
        if self.anchor_kl_coef <= 0.0:
            return
        if self._anchor_policy is None:
            self._refresh_anchor_policy()

    def _mix_rollout_with_replay(self, rollout_data: Any) -> tuple[_PPOSamples, float]:
        on_batch = self._to_batch(rollout_data)
        total = int(on_batch.observations.shape[0])

        if total <= 0:
            return on_batch, 0.0

        target_replay = int(round(total * self.replay_ratio))
        if target_replay <= 0:
            if self.normalize_advantage and self.normalize_advantage_per_source:
                on_batch.advantages = self._normalize_advantages(on_batch.advantages)
            return on_batch, 0.0

        replay_ready = self._replay_store.size > 0 and self._train_calls > self.replay_warmup_updates
        if not replay_ready:
            if self.normalize_advantage and self.normalize_advantage_per_source:
                on_batch.advantages = self._normalize_advantages(on_batch.advantages)
            return on_batch, 0.0

        replay_oversample = False
        if self.strict_replay_mix:
            replay_count = target_replay
            replay_oversample = replay_count > self._replay_store.size
        else:
            replay_count = min(target_replay, self._replay_store.size)

        if replay_count <= 0:
            if self.normalize_advantage and self.normalize_advantage_per_source:
                on_batch.advantages = self._normalize_advantages(on_batch.advantages)
            return on_batch, 0.0

        on_count = max(0, total - replay_count)

        permutation = th.randperm(total, device=on_batch.observations.device)
        on_idx = permutation[:on_count]

        on_obs = on_batch.observations[on_idx]
        on_actions = on_batch.actions[on_idx]
        on_old_values = on_batch.old_values[on_idx]
        on_old_log_prob = on_batch.old_log_prob[on_idx]
        on_advantages = on_batch.advantages[on_idx]
        on_returns = on_batch.returns[on_idx]

        replay_batch = self._replay_store.sample(
            replay_count,
            on_batch.observations.device,
            allow_oversample=replay_oversample,
        )
        if replay_batch is None:
            if self.normalize_advantage and self.normalize_advantage_per_source:
                on_batch.advantages = self._normalize_advantages(on_batch.advantages)
            return on_batch, 0.0

        replay_advantages = replay_batch.advantages

        if self.normalize_advantage and self.normalize_advantage_per_source:
            if on_advantages.numel() > 1:
                on_advantages = self._normalize_advantages(on_advantages)
            if replay_advantages.numel() > 1:
                replay_advantages = self._normalize_advantages(replay_advantages)

        mixed = _PPOSamples(
            observations=th.cat((on_obs, replay_batch.observations), dim=0),
            actions=th.cat((on_actions, replay_batch.actions), dim=0),
            old_values=th.cat((on_old_values, replay_batch.old_values), dim=0),
            old_log_prob=th.cat((on_old_log_prob, replay_batch.old_log_prob), dim=0),
            advantages=th.cat((on_advantages, replay_advantages), dim=0),
            returns=th.cat((on_returns, replay_batch.returns), dim=0),
        )

        effective_fraction = float(replay_batch.observations.shape[0] / max(1, mixed.observations.shape[0]))
        return mixed, effective_fraction

    @staticmethod
    def _mean_distribution_kl(anchor_distribution: Any, current_distribution: Any) -> th.Tensor:
        anchor_base = getattr(anchor_distribution, "distribution", None)
        current_base = getattr(current_distribution, "distribution", None)

        if isinstance(anchor_base, list) and isinstance(current_base, list):
            kl_terms = []
            for anchor_part, current_part in zip(anchor_base, current_base):
                kl_part = th.distributions.kl_divergence(anchor_part, current_part)
                if kl_part.ndim > 1:
                    kl_part = kl_part.sum(dim=1)
                kl_terms.append(kl_part)
            if not kl_terms:
                return th.zeros((), device=th.device("cpu"), dtype=th.float32)
            return th.stack(kl_terms, dim=1).sum(dim=1).mean()

        if anchor_base is None or current_base is None:
            return th.zeros((), device=th.device("cpu"), dtype=th.float32)

        anchor_distribution_t = anchor_base if isinstance(anchor_base, TorchDistribution) else None
        current_distribution_t = current_base if isinstance(current_base, TorchDistribution) else None
        if anchor_distribution_t is None or current_distribution_t is None:
            return th.zeros((), device=th.device("cpu"), dtype=th.float32)

        kl_values = th.distributions.kl_divergence(anchor_distribution_t, current_distribution_t)
        if kl_values.ndim > 1:
            kl_values = kl_values.sum(dim=1)
        return kl_values.mean()

    def _compute_anchor_kl_loss(self, observations: th.Tensor) -> th.Tensor:
        if self._anchor_policy is None or self.anchor_kl_coef <= 0.0:
            return th.zeros((), device=observations.device, dtype=th.float32)

        with th.no_grad():
            anchor_distribution = self._anchor_policy.get_distribution(observations)
        current_distribution = self.policy.get_distribution(observations)

        kl_loss = self._mean_distribution_kl(anchor_distribution, current_distribution)
        if kl_loss.device != observations.device:
            kl_loss = kl_loss.to(observations.device)
        return kl_loss

    def _compute_bc_loss(self) -> th.Tensor:
        if self._bc_store is None or self.bc_loss_coef <= 0.0:
            return th.zeros((), device=self.device, dtype=th.float32)

        if not self._bc_store.enabled:
            if not self._bc_warning_printed and self.verbose >= 1:
                print(f"[AnchoredReplayPPO] BC disabled: {self._bc_store.disable_reason}")
                self._bc_warning_printed = True
            return th.zeros((), device=self.device, dtype=th.float32)

        sampled = self._bc_store.sample(self.bc_batch_size, self.device)
        if sampled is None:
            return th.zeros((), device=self.device, dtype=th.float32)

        observations, actions = sampled
        distribution = self.policy.get_distribution(observations)

        eval_actions = actions
        if isinstance(self.action_space, spaces.Discrete):
            eval_actions = actions.long().flatten()
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            eval_actions = actions.long()

        log_prob = distribution.log_prob(eval_actions)
        return -th.mean(log_prob)

    def _apply_pcgrad_step(self, core_loss: th.Tensor, aux_loss: th.Tensor) -> bool:
        params = [parameter for parameter in self.policy.parameters() if parameter.requires_grad]
        if not params:
            return False

        core_grads = th.autograd.grad(core_loss, params, retain_graph=True, allow_unused=True)
        aux_grads = th.autograd.grad(aux_loss, params, retain_graph=False, allow_unused=True)

        merged_any = False
        self.policy.optimizer.zero_grad()

        for parameter, core_grad, aux_grad in zip(params, core_grads, aux_grads):
            if core_grad is None and aux_grad is None:
                continue

            if core_grad is None:
                merged_grad = aux_grad
            elif aux_grad is None:
                merged_grad = core_grad
            else:
                dot_product = th.sum(core_grad * aux_grad)
                projected_core = core_grad
                if dot_product.item() < 0.0:
                    denominator = th.sum(aux_grad * aux_grad).clamp_min(1e-12)
                    projected_core = core_grad - (dot_product / denominator) * aux_grad
                merged_grad = projected_core + aux_grad

            if merged_grad is None:
                continue

            parameter.grad = merged_grad.detach()
            merged_any = True

        if not merged_any:
            return False

        th.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        self.policy.optimizer.step()
        return True

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        clip_range_vf = None
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        self._train_calls += 1
        self._maybe_init_anchor()

        entropy_losses: list[float] = []
        pg_losses: list[float] = []
        value_losses: list[float] = []
        clip_fractions: list[float] = []
        approx_kl_divs: list[float] = []
        replay_fractions: list[float] = []
        anchor_kl_losses: list[float] = []
        bc_losses: list[float] = []
        aux_losses: list[float] = []

        pcgrad_steps = 0
        continue_training = True
        last_total_loss = th.zeros((), device=self.device, dtype=th.float32)

        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                batch, replay_fraction = self._mix_rollout_with_replay(rollout_data)
                replay_fractions.append(float(replay_fraction))

                actions = batch.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(batch.observations, actions)
                values = values.flatten()

                advantages = batch.advantages
                if self.normalize_advantage and not self.normalize_advantage_per_source and advantages.numel() > 1:
                    advantages = self._normalize_advantages(advantages)

                ratio = th.exp(log_prob - batch.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(float(policy_loss.item()))
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(float(clip_fraction))

                if clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = batch.old_values + th.clamp(values - batch.old_values, -clip_range_vf, clip_range_vf)

                value_loss = F.mse_loss(batch.returns, values_pred)
                value_losses.append(float(value_loss.item()))

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(float(entropy_loss.item()))

                core_loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                anchor_kl_loss = self._compute_anchor_kl_loss(batch.observations)
                bc_loss = self._compute_bc_loss()
                aux_loss = self.anchor_kl_coef * anchor_kl_loss + self.bc_loss_coef * bc_loss
                total_loss = core_loss + aux_loss
                last_total_loss = total_loss

                anchor_kl_losses.append(float(anchor_kl_loss.detach().item()))
                bc_losses.append(float(bc_loss.detach().item()))
                aux_losses.append(float(aux_loss.detach().item()))

                with th.no_grad():
                    log_ratio = log_prob - batch.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio)
                    approx_kl_divs.append(float(approx_kl_div.cpu().item()))

                if self.target_kl is not None and approx_kl_divs[-1] > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_divs[-1]:.2f}")
                    break

                used_pcgrad = False
                if self.enable_pcgrad and (self.anchor_kl_coef > 0.0 or self.bc_loss_coef > 0.0):
                    used_pcgrad = self._apply_pcgrad_step(core_loss, aux_loss)
                    if used_pcgrad:
                        pcgrad_steps += 1

                if not used_pcgrad:
                    self.policy.optimizer.zero_grad()
                    total_loss.backward()
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", float(np.mean(entropy_losses)) if entropy_losses else 0.0)
        self.logger.record("train/policy_gradient_loss", float(np.mean(pg_losses)) if pg_losses else 0.0)
        self.logger.record("train/value_loss", float(np.mean(value_losses)) if value_losses else 0.0)
        self.logger.record("train/approx_kl", float(np.mean(approx_kl_divs)) if approx_kl_divs else 0.0)
        self.logger.record("train/clip_fraction", float(np.mean(clip_fractions)) if clip_fractions else 0.0)
        self.logger.record("train/loss", float(last_total_loss.detach().item()))
        self.logger.record("train/explained_variance", float(explained_var))

        self.logger.record("train/replay_ratio_target", float(self.replay_ratio))
        self.logger.record("train/replay_fraction", float(np.mean(replay_fractions)) if replay_fractions else 0.0)
        self.logger.record("train/replay_buffer_size", int(self._replay_store.size))
        self.logger.record("train/anchor_kl_loss", float(np.mean(anchor_kl_losses)) if anchor_kl_losses else 0.0)
        self.logger.record("train/bc_loss", float(np.mean(bc_losses)) if bc_losses else 0.0)
        self.logger.record("train/aux_loss", float(np.mean(aux_losses)) if aux_losses else 0.0)
        self.logger.record("train/pcgrad_steps", int(pcgrad_steps))

        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

        self._replay_store.add_from_rollout_buffer(self.rollout_buffer)

        if self.anchor_kl_coef > 0.0:
            self._updates_since_anchor += 1
            if self._updates_since_anchor >= self.anchor_update_interval:
                self._refresh_anchor_policy()
