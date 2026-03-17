"""Intent-conditioned goal policy and PPO trainer for structured LLC learning."""

from __future__ import annotations

from typing import Optional
from typing import cast

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import explained_variance

from feature_extractor.memory.state_spec import StateSpec

INTENT_NAMES = ("navigate", "attack", "recover")
INTENT_NAVIGATE = 0
INTENT_ATTACK = 1
INTENT_RECOVER = 2


class IntentConditionedMlpExtractor(nn.Module):
    """Policy/value latent extractor with sampled discrete intent variable."""

    def __init__(self, feature_dim: int, intent_dim: int = 3):
        super().__init__()
        self.intent_dim = int(intent_dim)
        self.latent_dim_pi = 256
        self.latent_dim_vf = 256

        self.intent_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.intent_dim),
        )
        self.actor_net = nn.Sequential(
            nn.Linear(feature_dim + self.intent_dim, self.latent_dim_pi),
            nn.ReLU(),
        )
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, self.latent_dim_vf),
            nn.ReLU(),
        )

        self._intent_counts_rollout = np.zeros(self.intent_dim, dtype=np.float64)
        self._last_intent_index = INTENT_NAVIGATE

    def intent_logits(self, features: th.Tensor) -> th.Tensor:
        return self.intent_net(features)

    def _sample_intent(
        self,
        features: th.Tensor,
        deterministic: bool = False,
        track_intent: bool = False,
    ) -> tuple[th.Tensor, th.Tensor]:
        logits = self.intent_net(features)
        if deterministic:
            z = th.argmax(logits, dim=1)
        else:
            z = th.distributions.Categorical(logits=logits).sample()

        if z.numel() > 0:
            self._last_intent_index = int(z[0].detach().cpu().item())

        if track_intent and z.numel() > 0:
            counts = np.bincount(z.detach().cpu().numpy(), minlength=self.intent_dim).astype(np.float64)
            self._intent_counts_rollout += counts

        z_one_hot = F.one_hot(z, num_classes=self.intent_dim).to(features.dtype)
        return logits, z_one_hot

    def forward(
        self,
        features: th.Tensor,
        deterministic: bool = False,
        track_intent: bool = False,
    ) -> tuple[th.Tensor, th.Tensor]:
        _, z_one_hot = self._sample_intent(features, deterministic=deterministic, track_intent=track_intent)
        latent_pi = self.actor_net(th.cat([features, z_one_hot], dim=1))
        latent_vf = self.value_net(features)
        return latent_pi, latent_vf

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        latent_pi, _ = self.forward(features, deterministic=False, track_intent=False)
        return latent_pi

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)

    def consume_rollout_intent_counts(self) -> np.ndarray:
        counts = self._intent_counts_rollout.copy()
        self._intent_counts_rollout.fill(0.0)
        return counts

    @property
    def last_intent_index(self) -> int:
        return int(self._last_intent_index)


class GoalIntentActorCriticPolicy(ActorCriticPolicy):
    """Actor-critic policy with FiLM features + sampled discrete intent."""

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = IntentConditionedMlpExtractor(self.features_dim, intent_dim=3)

    def _latent(
        self,
        obs: th.Tensor,
        deterministic_intent: bool,
        track_intent: bool,
    ) -> tuple[th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(
                features,
                deterministic=deterministic_intent,
                track_intent=track_intent,
            )
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        return latent_pi, latent_vf

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        latent_pi, latent_vf = self._latent(
            obs,
            deterministic_intent=deterministic,
            track_intent=(not self.training),
        )
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        latent_pi, latent_vf = self._latent(obs, deterministic_intent=False, track_intent=False)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs: th.Tensor):
        latent_pi, _ = self._latent(obs, deterministic_intent=False, track_intent=(not self.training))
        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        _, latent_vf = self._latent(obs, deterministic_intent=True, track_intent=False)
        return self.value_net(latent_vf)

    def get_intent_logits(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs)
        if isinstance(features, tuple):
            pi_features, _ = features
            return self.mlp_extractor.intent_logits(pi_features)
        return self.mlp_extractor.intent_logits(features)

    def compute_heuristic_intent_targets(self, obs: th.Tensor) -> th.Tensor:
        if obs.ndim != 2:
            raise ValueError(f"Expected [batch, obs_dim] tensor, got shape={tuple(obs.shape)}")

        state_dim = int(getattr(self.features_extractor, "state_dim", obs.shape[1]))
        state = obs[:, :state_dim]

        offstage = state[:, StateSpec.index("player_is_offstage")] > 0.5
        strike = state[:, StateSpec.index("in_strike_range")] > 0.5
        frame_adv = state[:, StateSpec.index("frame_advantage_estimate")] > 0.0

        targets = th.full((state.shape[0],), INTENT_NAVIGATE, dtype=th.long, device=obs.device)
        targets = th.where(strike & frame_adv, th.full_like(targets, INTENT_ATTACK), targets)
        targets = th.where(offstage, th.full_like(targets, INTENT_RECOVER), targets)
        return targets

    @property
    def last_intent_index(self) -> int:
        return int(self.mlp_extractor.last_intent_index)

    def consume_rollout_intent_counts(self) -> np.ndarray:
        return self.mlp_extractor.consume_rollout_intent_counts()


class IntentPPO(PPO):
    """PPO + weak intent supervision: L_total = L_PPO + lambda * L_intent."""

    def __init__(self, *args, intent_loss_coef: float = 0.1, **kwargs):
        self.intent_loss_coef = float(intent_loss_coef)
        super().__init__(*args, **kwargs)

    def _denormalize_obs_tensor(self, obs: th.Tensor) -> th.Tensor:
        vecnorm = self.get_vec_normalize_env()
        if vecnorm is None or not getattr(vecnorm, "norm_obs", False):
            return obs

        obs_np = obs.detach().cpu().numpy()
        denorm_np = vecnorm.unnormalize_obs(obs_np)
        return th.as_tensor(denorm_np, device=obs.device, dtype=obs.dtype)

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining) if callable(self.clip_range) else float(self.clip_range)
        if self.clip_range_vf is not None:
            clip_range_vf = (
                self.clip_range_vf(self._current_progress_remaining)
                if callable(self.clip_range_vf)
                else float(self.clip_range_vf)
            )

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        intent_losses, intent_accs = [], []

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    clip_range_vf_val = float(clip_range_vf)
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf_val, clip_range_vf_val
                    )

                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                denorm_obs = self._denormalize_obs_tensor(rollout_data.observations)
                policy = cast(GoalIntentActorCriticPolicy, self.policy)
                intent_logits = policy.get_intent_logits(denorm_obs)
                intent_targets = policy.compute_heuristic_intent_targets(denorm_obs)
                intent_loss = F.cross_entropy(intent_logits, intent_targets)
                intent_losses.append(intent_loss.item())

                with th.no_grad():
                    intent_pred = th.argmax(intent_logits, dim=1)
                    intent_acc = (intent_pred == intent_targets).float().mean().item()
                    intent_accs.append(intent_acc)

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + self.intent_loss_coef * intent_loss
                )

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/intent_loss", np.mean(intent_losses) if len(intent_losses) else 0.0)
        self.logger.record("train/intent_acc", np.mean(intent_accs) if len(intent_accs) else 0.0)
        self.logger.record("train/intent_coef", self.intent_loss_coef)
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        if isinstance(self.policy, GoalIntentActorCriticPolicy):
            counts = self.policy.consume_rollout_intent_counts()
            total = float(np.sum(counts))
            if total > 0:
                for i, name in enumerate(INTENT_NAMES):
                    self.logger.record(f"rollout/intent_{name}_pct", float(100.0 * counts[i] / total))

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
