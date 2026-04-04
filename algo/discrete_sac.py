"""SAC-Discrete: Soft Actor-Critic for discrete action spaces.

Implements the algorithm from Christodoulou (2019) "Soft Actor-Critic for
Discrete Action Settings" on top of SB3's ``OffPolicyAlgorithm`` base.

Key differences from continuous SAC:
- Actor outputs a Categorical distribution (softmax over all actions).
- Critics output Q(s, a) for *all* actions simultaneously (no action input).
- Entropy and actor loss are computed as expectations over the full action
  distribution (no reparameterisation trick needed).

Usage::

    from algo.discrete_sac import DiscreteSAC
    model = DiscreteSAC("MlpPolicy", env, ...)
    model.learn(total_timesteps=100_000)
"""

from __future__ import annotations

from typing import Any, ClassVar, Optional, Union

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import polyak_update

from algo.discrete_sac_policy import DiscreteSACPolicy


class DiscreteSAC(OffPolicyAlgorithm):
    """SAC-Discrete (Christodoulou 2019) on SB3 OffPolicyAlgorithm base.

    Parameters
    ----------
    policy : str or type[DiscreteSACPolicy]
        "DiscreteSACPolicy" or the class itself.
    env : GymEnv
        Environment (Discrete action space required).
    learning_rate : float or Schedule
        Learning rate for actor, critic, and alpha optimisers.
    buffer_size : int
        Replay buffer capacity.
    learning_starts : int
        Number of random-action warmup steps before training begins.
    batch_size : int
        Mini-batch size sampled from the replay buffer.
    tau : float
        Polyak averaging coefficient for target network updates.
    gamma : float
        Discount factor.
    train_freq : int or tuple
        How many env steps between gradient updates.
    gradient_steps : int
        Number of gradient steps per ``train()`` call.  -1 means as many as
        steps collected.
    ent_coef : str or float
        Entropy coefficient.  ``"auto"`` enables automatic tuning toward
        ``target_entropy``.  ``"auto_0.1"`` sets the initial value.
    target_entropy : str or float
        Target entropy for auto-tuning.  ``"auto"`` uses
        ``-log(1/n_actions) * 0.98``.
    target_update_interval : int
        Polyak update frequency (in gradient steps).
    max_grad_norm : float
        Gradient clipping norm (0 = no clipping).
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "DiscreteSACPolicy": DiscreteSACPolicy,
    }

    policy: DiscreteSACPolicy

    def __init__(
        self,
        policy: Union[str, type[DiscreteSACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 100_000,
        learning_starts: int = 1_000,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_entropy: Union[str, float] = "auto",
        target_update_interval: int = 1,
        max_grad_norm: float = 0.0,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            supported_action_spaces=(spaces.Discrete,),
            support_multi_env=True,
        )

        self.ent_coef_init = ent_coef
        self.target_entropy = target_entropy  # type: ignore[assignment]
        self.target_update_interval = target_update_interval
        self.max_grad_norm = max_grad_norm

        # Set up during _setup_model
        self.log_ent_coef: Optional[th.Tensor] = None
        self.ent_coef_optimizer: Optional[th.optim.Optimizer] = None
        self.ent_coef_tensor: Optional[th.Tensor] = None

        if _init_setup_model:
            self._setup_model()

    # ------------------------------------------------------------------
    def _setup_model(self) -> None:
        super()._setup_model()

        n_actions = int(self.action_space.n)  # type: ignore[union-attr]

        # --- Target entropy ---
        if self.target_entropy == "auto":
            # 98% of maximum entropy for uniform Categorical(n_actions)
            self.target_entropy = float(-np.log(1.0 / n_actions) * 0.98)
        else:
            self.target_entropy = float(self.target_entropy)

        # --- Entropy coefficient (alpha) ---
        if isinstance(self.ent_coef_init, str) and self.ent_coef_init.startswith("auto"):
            init_value = 1.0
            if "_" in self.ent_coef_init:
                init_value = float(self.ent_coef_init.split("_")[1])
            self.log_ent_coef = th.log(
                th.ones(1, device=self.device) * init_value
            ).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam(
                [self.log_ent_coef], lr=self.lr_schedule(1)
            )
        else:
            self.ent_coef_tensor = th.tensor(
                float(self.ent_coef_init), device=self.device
            )

    # ------------------------------------------------------------------
    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)

        # Update learning rates
        actor_lr = self.lr_schedule(self._current_progress_remaining)
        for pg in self.policy.actor_optimizer.param_groups:
            pg["lr"] = actor_lr
        for pg in self.policy.critic_optimizer.param_groups:
            pg["lr"] = actor_lr
        if self.ent_coef_optimizer is not None:
            for pg in self.ent_coef_optimizer.param_groups:
                pg["lr"] = actor_lr

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )

            # Current entropy coefficient
            if self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
            else:
                ent_coef = self.ent_coef_tensor  # type: ignore[assignment]

            ent_coefs.append(ent_coef.item())

            # ============================================================
            # Critic update
            # ============================================================
            with th.no_grad():
                # Next-state action distribution from actor
                next_probs, next_log_probs = self.policy.get_action_dist(
                    replay_data.next_observations
                )
                # Target Q-values for all actions
                next_q1_tgt, next_q2_tgt = self.policy.q_values_target(
                    replay_data.next_observations
                )
                next_q_min = th.min(next_q1_tgt, next_q2_tgt)  # (B, n_actions)

                # V(s') = E_a[Q(s',a) - α log π(a|s')]
                next_v = (next_probs * (next_q_min - ent_coef * next_log_probs)).sum(
                    dim=1, keepdim=True
                )  # (B, 1)

                # TD target
                target_q = replay_data.rewards + (
                    1 - replay_data.dones
                ) * self.gamma * next_v

            # Current Q-values for all actions, then gather the taken action
            q1_all, q2_all = self.policy.q_values(replay_data.observations)
            actions = replay_data.actions.long()  # (B, 1)
            q1 = q1_all.gather(1, actions)  # (B, 1)
            q2 = q2_all.gather(1, actions)  # (B, 1)

            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

            self.policy.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.max_grad_norm > 0:
                th.nn.utils.clip_grad_norm_(
                    list(self.policy.critic_1.parameters())
                    + list(self.policy.critic_2.parameters())
                    + list(self.policy.critic_features_extractor.parameters()),
                    self.max_grad_norm,
                )
            self.policy.critic_optimizer.step()
            critic_losses.append(critic_loss.item())

            # ============================================================
            # Actor update
            # ============================================================
            probs, log_probs = self.policy.get_action_dist(replay_data.observations)

            # Q-values (detach critics to avoid affecting critic grads)
            with th.no_grad():
                q1_pi, q2_pi = self.policy.q_values(replay_data.observations)
            q_min_pi = th.min(q1_pi, q2_pi)

            # Actor loss: E_a[α log π(a|s) - Q(s,a)] (minimise)
            actor_loss = (probs * (ent_coef * log_probs - q_min_pi)).sum(dim=1).mean()

            self.policy.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.max_grad_norm > 0:
                th.nn.utils.clip_grad_norm_(
                    list(self.policy.actor.parameters())
                    + list(self.policy.actor_features_extractor.parameters()),
                    self.max_grad_norm,
                )
            self.policy.actor_optimizer.step()
            actor_losses.append(actor_loss.item())

            # ============================================================
            # Entropy coefficient update
            # ============================================================
            if self.ent_coef_optimizer is not None:
                # Entropy of current policy (detached)
                with th.no_grad():
                    entropy = -(probs * log_probs).sum(dim=1).mean()
                ent_coef_loss = -(
                    self.log_ent_coef * (self.target_entropy - entropy)  # type: ignore[operator]
                ).mean()

                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
                ent_coef_losses.append(ent_coef_loss.item())

            # ============================================================
            # Target network update
            # ============================================================
            if step % self.target_update_interval == 0:
                polyak_update(
                    self.policy.critic_1.parameters(),
                    self.policy.critic_1_target.parameters(),
                    self.tau,
                )
                polyak_update(
                    self.policy.critic_2.parameters(),
                    self.policy.critic_2_target.parameters(),
                    self.tau,
                )
                polyak_update(
                    self.policy.critic_features_extractor.parameters(),
                    self.policy.critic_target_features_extractor.parameters(),
                    self.tau,
                )

        self._n_updates += gradient_steps

        # Logging
        self.logger.record("train/ent_coef", float(np.mean(ent_coefs)))
        self.logger.record("train/actor_loss", float(np.mean(actor_losses)))
        self.logger.record("train/critic_loss", float(np.mean(critic_losses)))
        if len(ent_coef_losses) > 0:
            self.logger.record(
                "train/ent_coef_loss", float(np.mean(ent_coef_losses))
            )
