"""Hot-path optimizations must preserve pixels and leave live methods untouched."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize('mirrored', [False, True])
def test_stack_reads_only_requested_frames_and_owns_result(mirrored):
    from train.round_a_data import stack_frames
    frames = np.random.default_rng(42).integers(0, 256, (9, 12, 20, 3), dtype=np.uint8)
    refs = [8, 6, 4, 0]
    class History:
        def __getitem__(self, index):
            assert isinstance(index, int) and index in refs
            return frames[index]
    selected = frames[refs]
    if mirrored:
        selected = selected[:, :, ::-1, :]
    expected = selected.transpose(0, 3, 1, 2).reshape(12, 12, 20).copy()
    result = stack_frames(History(), refs, mirrored)
    np.testing.assert_array_equal(result, expected)
    assert result.flags.c_contiguous
    frames[:] = 0
    np.testing.assert_array_equal(result, expected)


def test_timing_restores_methods_and_saves_on_exception(tmp_path):
    from train.round_a_timing import TimingProbe
    ticks = iter([0., .002, .01, .014])
    class Target:
        def call(self, x):
            if x < 0:
                raise ValueError('original failure')
            return x + 1
    target = Target()
    output = tmp_path / 'timing.json'
    with pytest.raises(ValueError, match='original failure'):
        with TimingProbe(output, clock=lambda: next(ticks), max_samples=1) as probe:
            probe.wrap('example', target, 'call')
            assert target.call(2) == 3
            target.call(-1)
    assert 'call' not in vars(target)
    report = json.loads(output.read_text())
    row = report['stages']['example']
    assert row['calls'] == 2
    assert row['samples'] == 1
    assert row['mean_ms'] == pytest.approx(3.)
    assert row['p95_ms'] == pytest.approx(4.)


def test_timing_preserves_instance_override_and_ignores_missing(tmp_path):
    from train.round_a_timing import TimingProbe
    original = lambda: 7
    target = SimpleNamespace(call=original)
    with TimingProbe(tmp_path / 'timing.json') as probe:
        probe.wrap('absent', target, 'missing')
        probe.wrap('present', target, 'call')
        assert target.call() == 7
    assert target.call is original
    assert not hasattr(target, 'missing')
    with pytest.raises(ValueError):
        TimingProbe(tmp_path / 'bad.json', max_samples=0)


def test_timing_cli_is_opt_in():
    from train.round_a import parser
    assert not parser().parse_args(['--stage', 'train']).profile_timing
    assert parser().parse_args(['--stage', 'train', '--profile-timing']).profile_timing


def test_live_profile_includes_wrappers_and_replay(tmp_path):
    import gymnasium as gym
    from train.round_a_timing import qualification_timing
    from train.round_a_data import ObservationAdapter
    from train.match_env import MatchTrainingEnv, ReplayGymWrapper
    from world_model.replay_log import WorldReplayWriter
    class Base(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, (146,), np.float32)
        action_space = gym.spaces.Discrete(27)
        _last_frame = np.zeros((36, 64, 3), np.uint8)
        def reset(self, **kwargs): return np.zeros(146, np.float32), {}
        def step(self, action): return np.zeros(146, np.float32), 0., False, False, {}
        def _release_all_inputs(self): pass
    match = MatchTrainingEnv(Base(), mode='training_room', steps_path=tmp_path / 'steps.csv')
    replay = ReplayGymWrapper(match, WorldReplayWriter(tmp_path / 'world'))
    adapter = ObservationAdapter(replay, 'mlp', width=64, height=36, hz=0)
    model = SimpleNamespace(predict=lambda obs: 0)
    with qualification_timing(model, adapter, tmp_path / 'live.json'):
        obs, _ = adapter.reset()
        adapter.step(model.predict(obs))
    adapter.close()
    rows = json.loads((tmp_path / 'live.json').read_text())['stages']
    for key in ('policy_predict', 'image_history', 'base_step', 'match_step', 'csv_write', 'replay_append'):
        assert rows[key]['calls'] >= 1
    assert 'step' not in vars(match)


def test_offline_benchmark_checks_exact_history_and_budgets():
    from tools.benchmark_round_a_overhead import benchmark
    report = benchmark(iterations=2, repeats=1)
    assert report['pixel_equivalence'] is True
    assert len(report['measurements']) == 7
    assert report['structured_equivalence'] is True
    assert all(row['samples'] == 2 for row in report['measurements'])
    with pytest.raises(ValueError):
        benchmark(iterations=0)


def test_benchmark_cli_saves_new_result_only(tmp_path):
    from tools.benchmark_round_a_overhead import main
    path = tmp_path / 'benchmark.json'
    args = ['--iterations', '1', '--repeats', '1', '--output', str(path)]
    main(args)
    assert json.loads(path.read_text())['pixel_equivalence']
    with pytest.raises(FileExistsError):
        main(args)
