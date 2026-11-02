"""Offline hot-path microbenchmarks: no capture, keyboard input, or training.

Synthetic full-resolution pixels measure copying/resizing, not detector accuracy
or achievable live Hz. An optional existing MLP checkpoint measures CPU predict.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import cv2
import gymnasium as gym
import numpy as np

from brawl.pipelines.round_a.data import ObservationAdapter, stack_frames
from brawl.training.run_utils import sha256, write_json


def benchmark(*, iterations: int = 300, repeats: int = 3, checkpoint: Path | None = None) -> dict:
    if not 1 <= iterations <= 5000 or not 1 <= repeats <= 5:
        raise ValueError('Budget: 1..5000 iterations and 1..5 repeats')
    rng = np.random.default_rng(42)
    frame = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    history = list(rng.integers(0, 256, (9, 144, 256, 3), dtype=np.uint8))
    refs = [8, 6, 4, 0]

    def legacy(mirrored):
        image = np.asarray(history)[refs]
        if mirrored:
            image = image[:, :, ::-1, :]
        return image.transpose(0, 3, 1, 2).reshape(12, 144, 256).copy()

    for mirrored in (False, True):
        np.testing.assert_array_equal(legacy(mirrored), stack_frames(history, refs, mirrored))
    class PixelSource(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, (146,), np.float32)
        action_space = gym.spaces.Discrete(27)
        _last_frame = frame
    obs = np.arange(146, dtype=np.float32)
    shared = ObservationAdapter(PixelSource(), 'mlp', width=256, height=144, hz=0)
    deployment = ObservationAdapter(PixelSource(), 'mlp', width=256, height=144, hz=0, prepare_images=False)
    np.testing.assert_array_equal(shared.capture(obs), deployment.capture(obs))
    variants = [
        ('history_legacy', lambda: legacy(False)),
        ('history_direct', lambda: stack_frames(history, refs, False)),
        ('mirrored_legacy', lambda: legacy(True)),
        ('mirrored_direct', lambda: stack_frames(history, refs, True)),
        ('resize_area', lambda: cv2.resize(frame, (256, 144), interpolation=cv2.INTER_AREA)),
        ('adapter_shared_pixels', lambda: shared.capture(obs)),
        ('adapter_structured_only', lambda: deployment.capture(obs)),
    ]
    report = dict(pixel_equivalence=True, structured_equivalence=True, numpy=np.__version__, opencv=cv2.__version__,
        opencv_threads=cv2.getNumThreads(), frame_shape=list(frame.shape),
        iterations=iterations, repeats=repeats, measurements=[],
        scope='Offline synthetic pixels; excludes capture, YOLO, classifier, input and replay. Not live Hz.')
    if checkpoint is not None:
        import torch
        from brawl.pipelines.round_a.models import load_model, Predictor
        torch.set_num_threads(1)  # Match the Round A entry point, including CPU MLP updates.
        predictor = Predictor(load_model(checkpoint, 'mlp', device='cpu'))
        obs = np.zeros(146, np.float32)
        variants.append(('mlp_predict', lambda: predictor.predict(obs)))
        report.update(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                      torch=torch.__version__, torch_threads=torch.get_num_threads())
    for repeat in range(repeats):
        # Reverse ordering to expose drift from warming/thermal load.
        for name, operation in variants[::1 if repeat % 2 == 0 else -1]:
            for _ in range(30):
                operation()
            samples = []
            for _ in range(iterations):
                began = time.perf_counter()
                operation()
                samples.append(1000 * (time.perf_counter() - began))
            report['measurements'].append(dict(variant=name, repeat=repeat, samples=len(samples),
                p50_ms=float(np.median(samples)), p95_ms=float(np.quantile(samples, .95))))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, help='Optional typed Round A MLP checkpoint (read-only)')
    parser.add_argument('--output', type=Path, required=True, help='New JSON result file; never overwrite')
    parser.add_argument('--iterations', type=int, default=300)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    report = benchmark(iterations=args.iterations, repeats=args.repeats, checkpoint=args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    for row in report['measurements']:
        print(f'{row["variant"]:20} repeat={row["repeat"]} p50={row["p50_ms"]:.3f}ms p95={row["p95_ms"]:.3f}ms')
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
