"""Round A model adapters. Import only AFTER capture_first in live entry points."""
from __future__ import annotations

import numpy as np
import torch as th
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from train.primitive_features import PrimitiveGeometry
from train.primitive_training import metadata as primitive_metadata
from train.checkpoint_roles import read_checkpoint_metadata, save_typed_checkpoint


class VisualStructured(BaseFeaturesExtractor):
    """Shared spatial encoder per RGB frame; temporal slots remain distinct."""
    def __init__(self, observation_space):
        super().__init__(observation_space, 147 + 4 * 64)
        self.geometry = PrimitiveGeometry(observation_space['structured'])
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=4, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 5)), nn.Flatten(), nn.Linear(480, 64), nn.ReLU())

    def forward(self, observations):
        frames = observations['frames']
        batch, _, height, width = frames.shape
        # SB3 performs uint8 -> [0,1] normalization exactly once before here.
        images = self.encoder(frames.reshape(batch * 4, 3, height, width)).reshape(batch, -1)
        return th.cat((self.geometry(observations['structured']), images), dim=1)


def build_model(candidate, env, **overrides):
    from train.primitive_training import build_model as baseline
    if candidate == 'mlp':
        model = baseline(env, **overrides)
    else:
        options = dict(n_steps=512, batch_size=256, n_epochs=4, learning_rate=1e-4,
            gamma=.999, gae_lambda=.95, clip_range=.1, ent_coef=.01,
            target_kl=.01, device='cpu', seed=42, verbose=0)
        policy = dict(net_arch={'pi': [128, 128], 'vf': [128, 128]})
        if candidate == 'recurrent':
            from sb3_contrib import RecurrentPPO
            policy.update(features_extractor_class=PrimitiveGeometry,
                          lstm_hidden_size=128, n_lstm_layers=1, enable_critic_lstm=True)
            cls, name = RecurrentPPO, 'MlpLstmPolicy'
        elif candidate == 'multi':
            policy.update(features_extractor_class=VisualStructured)
            cls, name = PPO, 'MultiInputPolicy'
        else:
            raise ValueError(f'Unknown candidate: {candidate}')
        options.update(overrides)
        options['policy_kwargs'] = policy
        model = cls(name, env, **options)
    model.round_a_candidate = candidate
    return model


class Predictor:
    def __init__(self, model):
        self.model, self.state = model, None
        self.first = True

    def predict(self, obs, *, start=False):
        action, self.state = self.model.predict(obs, state=self.state,
            episode_start=np.array([start or self.first]), deterministic=True)
        self.first = False
        return int(np.asarray(action).item())


def distribution(model, observations, state=None, *, start=False):
    tensor, _ = model.policy.obs_to_tensor(observations)
    if model.round_a_candidate != 'recurrent':
        return model.policy.get_distribution(tensor), None
    if state is None:
        layer = model.policy.lstm_actor
        state = tuple(th.zeros(layer.num_layers, 1, layer.hidden_size, device=model.device) for _ in range(2))
    starts = th.zeros(len(observations), device=model.device)
    starts[0] = float(start)
    return model.policy.get_distribution(tensor, state, starts)


def save_model(model, path, contract, **extra):
    if contract['candidate'] != model.round_a_candidate:
        raise ValueError('Model candidate differs from contract')
    validate_spaces(model, contract)
    payload = primitive_metadata() | dict(algorithm='round_a_v1', round_a=contract,
        environment_steps=int(model.num_timesteps), **extra)
    return save_typed_checkpoint(model, path, payload)


def load_model(path, candidate, env=None, *, device='cpu', **overrides):
    payload = read_checkpoint_metadata(path)
    contract = payload.get('round_a', {})
    if payload.get('algorithm') != 'round_a_v1' or contract.get('candidate') != candidate:
        raise ValueError('Wrong Round A candidate/contract; no implicit checkpoint conversion')
    for key, expected in primitive_metadata().items():
        if key != 'algorithm' and payload.get(key) != expected:
            raise ValueError(f'Primitive contract changed: {key}')
    if candidate == 'recurrent':
        from sb3_contrib import RecurrentPPO
        cls = RecurrentPPO
    else:
        cls = PPO
    model = cls.load(path, env=env, device=device, **overrides)
    if model.checkpoint_metadata != payload or model.round_a_candidate != candidate:
        raise ValueError('Embedded model contract differs from sidecar')
    validate_spaces(model, contract)
    return model


def validate_spaces(model, contract):
    import gymnasium as gym
    space = model.observation_space
    if not isinstance(model.action_space, gym.spaces.Discrete) or model.action_space.n != 27:
        raise ValueError('Invalid action space')
    if contract['candidate'] == 'multi':
        shape = (12, contract['image_height'], contract['image_width'])
        if (not isinstance(space, gym.spaces.Dict) or set(space.spaces) != {'structured', 'frames'}
                or space['frames'].shape != shape or space['frames'].dtype != np.uint8):
            raise ValueError('Visual observation space differs from checkpoint contract')
        space = space['structured']
    if not isinstance(space, gym.spaces.Box) or space.shape != (146,) or space.dtype != np.float32:
        raise ValueError('Structured observation space differs from checkpoint contract')
