"""Aligned pixel/structured observations and disk-backed episode demonstrations.

Frames are stored once, in world coordinates. Histories use the CURRENT mirror
flag, exactly as structured history does; mirroring each old frame separately
would introduce false motion when the fighters cross.
"""
from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import time

import gymnasium as gym
import numpy as np

from brawl.training.run_utils import sha256, write_json

OFFSETS = (0, 2, 4, 8)
DATA_VERSION = 'round_a_aligned_v1'


def stack_frames(frames, refs, mirrored):
    first = frames[int(refs[0])]
    height, width, channels = first.shape
    pixels = np.empty((len(refs) * channels, height, width), dtype=first.dtype)
    # Copy only the selected temporal slots directly into owned CHW storage.
    # Materializing all nine HWC history frames first wastes memory bandwidth.
    for slot, ref in enumerate(refs):
        frame = frames[int(ref)]
        if mirrored:
            frame = frame[:, ::-1, :]
        pixels[slot * channels:(slot + 1) * channels] = frame.transpose(2, 0, 1)
    return pixels


class ObservationAdapter(gym.Wrapper):
    """Aligned inputs; optional pixel work only for structured deployment arms."""
    def __init__(self, env, candidate, *, width, height, hz, clock=time.perf_counter, sleep=time.sleep,
                 prepare_images=True):
        super().__init__(env)
        if candidate not in ('mlp', 'recurrent', 'multi'):
            raise ValueError('Unknown candidate')
        if candidate == 'multi' and not prepare_images:
            raise ValueError('multi requires prepared images')
        self.prepare_images = prepare_images
        self.candidate, self.width, self.height = candidate, width, height
        self.clock, self.sleep, self.period = clock, sleep, 1 / hz if hz else 0.
        self.history = deque(maxlen=9)
        self.current = None
        self.last_decision = None
        self.intervals, self.overruns = [], 0
        self.cadence = {}
        self.recording = False
        self.frames, self.records, self.history_ids = [], [], deque(maxlen=9)
        if candidate == 'multi':
            self.observation_space = gym.spaces.Dict({
                'structured': env.observation_space,
                'frames': gym.spaces.Box(0, 255, (12, height, width), np.uint8)})

    def capture(self, obs):
        import cv2
        if np.asarray(obs).shape != (146,) or not np.isfinite(obs).all():
            raise RuntimeError('Nonfinite or malformed structured observation')
        frame = self.unwrapped._last_frame
        if frame is None:
            raise RuntimeError('Missing frame: cannot fabricate synchronized visual evidence')
        self.current_mirrored = bool(getattr(self.unwrapped, '_mirrored', False))
        if not self.prepare_images:
            # Raw capture still feeds YOLO, HUD and weapon sensing. Only the
            # unused CNN input is omitted; structured history is untouched.
            self.current = {'structured': np.asarray(obs, np.float32).copy()}
            return self.policy_obs()
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.history.append(frame.copy())
        mirrored = bool(getattr(self.unwrapped, '_mirrored', False))
        images = list(self.history)
        refs = [max(0, len(images)-1-offset) for offset in OFFSETS]
        pixels = stack_frames(images, refs, mirrored)
        self.current = {'structured': np.asarray(obs, np.float32).copy(), 'frames': pixels}
        self.current_mirrored = mirrored
        if self.recording:
            self.frames.append(frame.copy())
            self.history_ids.append(len(self.frames)-1)
        return self.policy_obs()

    def policy_obs(self):
        if self.candidate == 'multi':
            return {key: value.copy() for key, value in self.current.items()}
        return self.current['structured'].copy()

    def begin_recording(self):
        if not self.prepare_images:
            raise RuntimeError('Aligned collection requires prepared images')
        self.recording = True
        self.frames, self.records = [], []
        self.history_ids.clear()
        # Include the observation already visible to the next teacher decision.
        for frame in self.history:
            self.frames.append(frame.copy())
            self.history_ids.append(len(self.frames)-1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.current is None:
            self.history.clear()
            return self.capture(obs), info
        # MatchTrainingEnv's reporting reset returns the same observation. Do
        # not insert a duplicate frame into temporal history at this boundary.
        self.current['structured'] = np.asarray(obs, np.float32).copy()
        return self.policy_obs(), info

    def step(self, action):
        now = self.clock()
        if self.last_decision is not None:
            remaining = self.period - (now - self.last_decision)
            if remaining > 0:
                self.sleep(remaining)
            elapsed = self.clock() - self.last_decision
            self.intervals.append(elapsed)
            self.overruns += int(bool(self.period and elapsed > self.period * 1.1))
            try:
                usage = self.env.get_wrapper_attr('usage')
            except AttributeError:
                usage = 'all'
            stats = self.cadence.setdefault(usage, dict(intervals=0, overruns=0, seconds=0.))
            stats['intervals'] += 1
            stats['seconds'] += elapsed
            stats['overruns'] += int(bool(self.period and elapsed > self.period * 1.1))
            if len(self.intervals) > 10000:
                del self.intervals[:5000]
        self.last_decision = self.clock()
        if self.recording:
            ids = list(self.history_ids)
            self.records.append((self.current['structured'].copy(), int(action),
                [ids[max(0, len(ids)-1-offset)] for offset in OFFSETS], self.current_mirrored))
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.capture(obs), reward, terminated, truncated, info

    def release_inputs(self):
        self.env.get_wrapper_attr('release_inputs')()
        self.last_decision = None

    def cadence_report(self):
        return {usage: dict(values, measured_hz=values['intervals']/max(values['seconds'], 1e-6),
                            overrun_fraction=values['overruns']/max(values['intervals'], 1))
                for usage, values in self.cadence.items()}


class EpisodeWriter:
    def __init__(self, directory, *, width, height, extra=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if (self.directory / 'dataset.json').exists():
            raise FileExistsError('Dataset is immutable')
        self.manifest = dict(version=DATA_VERSION, image_width=width, image_height=height,
            offsets=list(OFFSETS), color='RGB', episodes=[], **(extra or {}))

    def append(self, *, obs, actions, valid, families, drill, frames, frame_refs, mirrored):
        name = f'episode_{len(self.manifest["episodes"]):06d}'
        directory = self.directory / name
        directory.mkdir(exist_ok=False)
        arrays = dict(obs=obs, actions=actions, valid=valid, families=families,
                      frames=frames, frame_refs=frame_refs, mirrored=mirrored)
        for key, value in arrays.items():
            np.save(directory / f'{key}.npy', value, allow_pickle=False)
        self.manifest['episodes'].append(dict(path=name, drill=drill,
            rows=len(actions), labels=int(np.asarray(valid).sum()),
            hashes={key: sha256(directory / f'{key}.npy') for key in arrays}))

    def finish(self):
        write_json(self.directory / 'dataset.json', self.manifest)


class Dataset:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / 'dataset.json').read_text())
        if (self.manifest['version'] != DATA_VERSION or self.manifest['offsets'] != list(OFFSETS)
                or self.manifest.get('color') != 'RGB'):
            raise ValueError('Incompatible aligned dataset')
        if not self.manifest['episodes']:
            raise ValueError('Empty dataset')
        for entry in self.manifest['episodes']:
            if set(entry['hashes']) != {'obs', 'actions', 'valid', 'families', 'frames', 'frame_refs', 'mirrored'}:
                raise ValueError('Missing array hashes')
            folder = (self.directory / entry['path']).resolve()
            if folder.parent != self.directory.resolve():
                raise ValueError('Episode path outside dataset')
            for key, digest in entry['hashes'].items():
                if key not in ('obs', 'actions', 'valid', 'families', 'frames', 'frame_refs', 'mirrored'):
                    raise ValueError('Invalid array name')
                if sha256(folder / f'{key}.npy') != digest:
                    raise ValueError('Dataset hash mismatch')
            episode = self.episode(entry)
            n = entry['rows']
            if episode['obs'].shape != (n, 146) or not np.isfinite(episode['obs']).all():
                raise ValueError('Invalid structured observations')
            if episode['frames'].shape[1:] != (self.manifest['image_height'], self.manifest['image_width'], 3):
                raise ValueError('Invalid frame dimensions')
            if (episode['frames'].dtype != np.uint8 or episode['frame_refs'].shape != (n, 4)
                    or not np.issubdtype(episode['frame_refs'].dtype, np.integer)):
                raise ValueError('Invalid frame contract')
            if np.any(episode['frame_refs'] < 0) or np.any(episode['frame_refs'] >= len(episode['frames'])):
                raise ValueError('Frame references outside episode')
            if (np.any(episode['frame_refs'][:, 1:] > episode['frame_refs'][:, :1])
                    or np.any(np.diff(episode['frame_refs'][:, 0]) < 0)):
                raise ValueError('Noncausal frame references')
            if (episode['actions'].shape != (n,) or not np.issubdtype(episode['actions'].dtype, np.integer)
                    or np.any((episode['actions'] < 0) | (episode['actions'] >= 27))):
                raise ValueError('Invalid actions')
            if episode['valid'].shape != (n,) or episode['valid'].dtype != bool or not episode['valid'].any():
                raise ValueError('Invalid label mask')
            if episode['families'].shape != (n,) or episode['mirrored'].shape != (n,) or episode['mirrored'].dtype != bool:
                raise ValueError('Invalid family/mirror metadata')

    def episode(self, entry):
        return {key: np.load(self.directory / entry['path'] / f'{key}.npy', mmap_mode='r', allow_pickle=False)
                for key in ('obs', 'actions', 'valid', 'families', 'frames', 'frame_refs', 'mirrored')}

    def split(self, seed=42):
        rng = np.random.default_rng(seed)
        result = dict(train=[], validation=[], test=[])
        for drill in sorted({e['drill'] for e in self.manifest['episodes']}):
            ids = [i for i, entry in enumerate(self.manifest['episodes']) if entry['drill'] == drill]
            if len(ids) < 5:
                raise ValueError('At least five accepted episodes per drill for train/validation/test')
            rng.shuffle(ids)
            count = max(1, int(np.ceil(.2 * len(ids))))
            result['test'].extend(ids[:count])
            result['validation'].extend(ids[count:2*count])
            result['train'].extend(ids[2*count:])
        return result

    def observations(self, episode, indices, candidate):
        structured = np.array(episode['obs'][indices], copy=True)
        if candidate != 'multi':
            return structured
        pixels = np.stack([stack_frames(episode['frames'], episode['frame_refs'][i], episode['mirrored'][i])
                           for i in indices])
        return dict(structured=structured, frames=pixels)
