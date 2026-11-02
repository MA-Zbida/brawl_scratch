"""Frozen final-policy diagnostics, with opt-in synchronized perception video."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Keep the same DXGI-before-CUDA contract as the live training entry point.
import capture_first  # noqa: E402,F401

import argparse
import csv
import hashlib
import json
import math
import time
import uuid
from contextlib import ExitStack
from typing import Any, Callable, Sequence

import numpy as np
import torch

from action_space import ACTION_DIM, Action
from feature_extractor.memory.state_spec import StateSpec
from train.match_env import MatchTrainingEnv


class CaptureProbe:
    """Own each retrieved image; retrieval time is NOT sensor exposure time."""

    def __init__(self, provider: Any, clock: Callable = time.perf_counter):
        self.provider = provider
        self.clock = clock
        self.capture_id = -1
        self.frame = None
        self.returned_at = math.nan
        self.last_valid_at = math.nan
        self.repeated_pixels = False

    def get_frame(self):
        frame = self.provider.get_frame()
        self.returned_at = self.clock()
        if frame is None:
            self.frame = None
            self.repeated_pixels = False
            return None
        owned = np.array(frame, copy=True)
        self.repeated_pixels = self.frame is not None and np.array_equal(owned, self.frame)
        self.capture_id += 1
        self.last_valid_at = self.returned_at
        self.frame = owned
        return owned

    def __getattr__(self, name):
        return getattr(self.provider, name)


class InputProbe:
    """Record API commands, not a claim that the game accepted those inputs."""

    def __init__(self, controller: Any, clock: Callable = time.perf_counter):
        self.controller = controller
        self.clock = clock
        self.calls: list[dict] = []

    def _call(self, method: str, keys: list[str], *args):
        event = dict(method=method, keys=keys, started_s=self.clock())
        self.calls.append(event)
        result = getattr(self.controller, method)(*args)
        event["returned_s"] = self.clock()
        return result

    def set_pressed(self, keys):
        return self._call("set_pressed", sorted(map(str, keys)), keys)

    def key_down(self, key):
        return self._call("key_down", [str(key)], key)

    def key_up(self, key):
        return self._call("key_up", [str(key)], key)

    def reset(self):
        return self._call("reset", [])

    def __getattr__(self, name):
        return getattr(self.controller, name)


class VideoSink:
    """One movie frame per valid capture; CSV timestamps retain variable cadence."""

    def __init__(self, path: Path, fps: float, max_width: int):
        self.path, self.fps, self.max_width = path, fps, max_width
        self.writer = None
        self.count = 0
        self.size = None

    def write(self, frame: np.ndarray, lines: list[str]) -> int:
        import cv2

        scale = min(1., self.max_width / frame.shape[1])
        width = max(2, int(frame.shape[1] * scale) // 2 * 2)
        height = max(2, int(frame.shape[0] * scale) // 2 * 2)
        canvas = np.zeros((height + 96, width, 3), dtype=np.uint8)
        canvas[96:] = cv2.resize(frame, (width, height))
        for i, line in enumerate(lines):
            cv2.putText(canvas, line, (8, 20 + 23 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, .48, (240, 240, 240), 1, cv2.LINE_AA)
        size = (width, height + 96)
        if self.writer is None:
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"),
                                          self.fps, size)
            if not self.writer.isOpened():
                raise RuntimeError(f"Cannot open video encoder: {self.path}")
            self.size = size
        if size != self.size:
            raise ValueError("Capture dimensions changed during recording")
        self.writer.write(canvas)
        index = self.count
        self.count += 1
        return index

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None


class WeaponCropSink:
    """Save exact classifier input patches, never pseudo-labels from predictions."""

    def __init__(self, directory: Path, stream: Any, capture: CaptureProbe):
        self.directory, self.stream, self.capture = directory, stream, capture
        (directory / 'unlabelled').mkdir(exist_ok=False)
        self.count = 0

    def __call__(self, name, box, patch, prediction) -> None:
        import cv2
        from dataclasses import asdict

        relative = Path('unlabelled') / f'capture_{self.capture.capture_id:06d}_{name}_{self.count:06d}.png'
        # Lossless/native resolution before model resize. MP4 pixels cannot
        # reproduce this input, even if the video and prediction are aligned.
        if not cv2.imwrite(str(self.directory / relative), patch):
            raise OSError(f'Could not save classifier crop: {relative}')
        self.stream.write(json.dumps(dict(
            observed_capture_id=self.capture.capture_id, fighter=name,
            box=box, path=relative.as_posix(), label=None,
            prediction=asdict(prediction), shape=list(patch.shape),
        )) + '\n')
        self.stream.flush()
        self.count += 1


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _state_columns(obs: np.ndarray, prefix: str) -> dict:
    names = StateSpec.FEATURES
    return {f"{prefix}_{names[i] if i < len(names) else 'history_' + str(i)}": float(v)
            for i, v in enumerate(np.asarray(obs).reshape(-1))}


def record_diagnostic(
    model: Any, base: Any, directory: Path, *, windows: int = 2,
    seconds: float = 120., max_steps: int = 10000, video: bool = False,
    video_fps: float = 30., video_width: int = 1280, stochastic: bool = False,
    weapon_crops: bool = False,
    seed: int = 42, clock: Callable = time.perf_counter,
) -> dict:
    """Own/close base; never update model weights or silently reset the arena."""
    directory = Path(directory)
    capture = CaptureProbe(base.frame_provider, clock)
    inputs = InputProbe(base.input_controller, clock)
    base.frame_provider, base.input_controller = capture, inputs
    env = MatchTrainingEnv(base, mode="training_room", max_steps=max_steps,
                           matches_path=directory / "windows.csv")
    env.usage = "evaluation"
    summary = dict(status="error", steps=0, completed_windows=0, video_frames=0)
    origin = clock()
    sink = VideoSink(directory / "diagnostic.mp4", video_fps, video_width) if video else None
    try:
        with ExitStack() as stack:
            if weapon_crops:
                classifier = getattr(base, 'weapon_classifier', None)
                if classifier is None:
                    raise ValueError('Weapon crops require an enabled visual classifier')
                crops_dir = directory / 'weapon_crops'
                crops_dir.mkdir(exist_ok=False)
                crop_stream = stack.enter_context((crops_dir / 'crops.jsonl').open('x', encoding='utf-8'))
                crop_sink = WeaponCropSink(crops_dir, crop_stream, capture)
                old_observer = getattr(classifier, 'prediction_observer', None)
                stack.callback(setattr, classifier, 'prediction_observer', old_observer)
                classifier.prediction_observer = crop_sink
                contract = dict(
                    input_width=classifier.width, input_height=classifier.height,
                    crop_pad=classifier.pad, classes=classifier.classes,
                    color_order='BGR', scale='float32 / 255', resize='INTER_AREA',
                    device=str(classifier.device),
                    label_contract='null until manually inspected; prediction is not ground truth',
                )
                (crops_dir / 'contract.json').write_text(json.dumps(contract, indent=2), encoding='utf-8')
            stream = stack.enter_context((directory / "transitions.csv").open("x", newline="", encoding="utf-8"))
            events = stack.enter_context((directory / "events.jsonl").open("x", encoding="utf-8"))
            writer = None
            model.policy.set_training_mode(False)
            obs, _ = env.reset(seed=seed)
            for window in range(1, windows + 1):
                started = clock()
                for step in range(1, max_steps + 1):
                    policy_capture_id = capture.capture_id
                    # A failed grab returns cached state in BrawlDeepEnv; keep
                    # the timestamp of that state's last successful retrieval.
                    policy_frame_time = capture.last_valid_at
                    inputs.calls.clear()
                    decision_started = clock()
                    with torch.no_grad():
                        tensor, _ = model.policy.obs_to_tensor(obs)
                        dist = model.policy.get_distribution(tensor).distribution
                        probs = dist.probs.detach().cpu().numpy().reshape(-1)
                        action = int(dist.sample().item()) if stochastic else int(probs.argmax())
                    decision_finished = clock()
                    # A window boundary is bookkeeping, not a physical game reset.
                    env.force_boundary = clock() - started >= seconds
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    returned = clock()
                    first_call = inputs.calls[0]["started_s"] if inputs.calls else math.nan
                    record_started = clock()
                    video_frame = -1
                    if sink is not None and capture.frame is not None:
                        video_frame = sink.write(capture.frame, [
                            f"window {window} step {step} | elapsed {returned-origin:.3f}s | video frame {sink.count}",
                            f"policy capture {policy_capture_id} -> returned capture {capture.capture_id} | requested {Action(action).name}",
                            f"reported {Action(int(info.get('effective_action', action))).name} | reward {reward:+.3f} | pickup p={probs[26]:.4f}",
                            "Image = returned perception frame; NOT necessarily the result of the requested action.",
                        ])
                    row = dict(
                        window=window, step=step, interaction=summary["steps"],
                        wall_s=returned-origin, policy_capture_id=policy_capture_id,
                        observed_capture_id=capture.capture_id, frame_valid=int(capture.frame is not None),
                        repeated_pixels=int(capture.repeated_pixels), video_frame=video_frame,
                        video_time_s=video_frame/video_fps if video_frame >= 0 else math.nan,
                        policy_frame_returned_s=policy_frame_time,
                        frame_returned_s=capture.returned_at, decision_started_s=decision_started,
                        decision_finished_s=decision_finished, first_input_call_s=first_call,
                        env_returned_s=returned,
                        decision_frame_age_s=decision_started-policy_frame_time,
                        policy_s=decision_finished-decision_started,
                        env_s=returned-decision_finished,
                        video_encode_s=clock()-record_started,
                        action_requested=action, action_reported=int(info.get("effective_action", action)),
                        reward=reward, terminated=int(terminated), truncated=int(truncated),
                        **{f"p_{i}": float(p) for i, p in enumerate(probs)},
                        **{f"reward_{k}": v for k, v in info["reward_breakdown"].items()},
                        **_state_columns(obs, "pre"), **_state_columns(next_obs, "post"),
                    )
                    for key in (
                        "canon_mirrored", "damage_valid", "op_delta_damage", "self_delta_damage",
                        "op_stock_lost_step", "self_stock_lost_step", "weapon_visible_this_frame",
                        "weapon_pickup_action", "weapon_drop_action", "player_weapon_visual_valid",
                        "player_weapon_visual_armed", "player_weapon_visual_confidence",
                        "player_weapon_visual_fresh", "player_weapon_visual_disarm_event",
                    ):
                        row[key] = info.get(key, math.nan)
                    events.write(json.dumps(dict(
                        interaction=summary["steps"], policy_capture_id=policy_capture_id,
                        observed_capture_id=capture.capture_id, input_calls=inputs.calls,
                        detections=info.get("detections", []),
                        hud=info.get("hud_diagnostics", {}),
                        input_taps_pressed=info.get("input_taps_pressed", []),
                    ), default=_json_default) + "\n")
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    summary["steps"] += 1
                    obs = next_obs
                    if step % 30 == 0:
                        stream.flush()
                        events.flush()
                    if terminated or truncated:
                        summary["completed_windows"] += 1
                        print(f"Window {window}/{windows}: {step} steps, {returned-started:.1f}s, "
                              f"stocks {env.last_summary['stocks_taken']:.0f}-{env.last_summary['stocks_lost']:.0f}")
                        if window < windows:
                            obs, _ = env.reset()
                        break
            summary["status"] = "complete"
    except KeyboardInterrupt:
        summary["status"] = "interrupted"
        print("Interrupted; retaining completed diagnostic rows.")
    finally:
        try:
            env.close()
        finally:
            if sink is not None:
                summary["video_frames"] = sink.count
                sink.close()
            summary["wall_seconds"] = clock() - origin
            (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args(argv: Sequence[str] | None = None):
    from train.final_config import add_weapon_state_argument
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", default="outputs/final_diagnostics")
    parser.add_argument("--windows", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=120.)
    parser.add_argument("--max-steps", type=int, default=10000, help="Safety cap per window")
    parser.add_argument("--video", action="store_true", help="Opt in to MP4 encoding")
    parser.add_argument("--weapon-crops", action="store_true",
                        help="Opt in to native lossless classifier crops; diagnostic disk/latency cost only")
    parser.add_argument("--video-fps", type=float, default=30., help="Playback FPS, not an environment speed limit")
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delay", type=float, default=5.)
    add_weapon_state_argument(parser)
    args = parser.parse_args(argv)
    if args.weapon_crops and args.weapon_state_source != 'visual_fusion':
        parser.error('--weapon-crops requires --weapon-state-source visual_fusion')
    if (args.windows < 1 or args.max_steps < 1 or args.video_width < 2 or
            not math.isfinite(args.seconds) or args.seconds <= 0 or
            not math.isfinite(args.video_fps) or args.video_fps <= 0 or
            not math.isfinite(args.delay) or args.delay < 0):
        parser.error("Window counts, durations and video settings must be valid positive values")
    return args


def main() -> None:
    args = parse_args()
    from dataclasses import asdict
    from train.final_policy import load_final_model
    from train.final_config import FinalAgentConfig
    from train.final_agent import _make_env
    from reward.match_objective import MatchObjective
    from train.checkpoint_roles import read_checkpoint_metadata
    from env import RUNTIME_CONTRACT

    model = load_final_model(args.model, device=args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    directory = Path(args.out_dir) / f"{Path(args.model).stem}_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"
    directory.mkdir(parents=True, exist_ok=False)
    with Path(args.model).open("rb") as handle:
        checkpoint_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    cfg = FinalAgentConfig(init_from=None, resume=args.model, device=args.device,
                           weapon_state_source=args.weapon_state_source,
                           weapon_classifier_checkpoint=args.weapon_classifier_checkpoint)
    manifest = dict(
        schema_version=1, **vars(args), **RUNTIME_CONTRACT, checkpoint_sha256=checkpoint_hash,
        checkpoint_metadata=read_checkpoint_metadata(args.model),
        training_env_config=asdict(cfg), reward_version=MatchObjective.VERSION,
        mode="training_room", deterministic=not args.stochastic,
        clock="time.perf_counter; timestamps share a process clock, not video playback time",
        capture_contract="capture_id counts retrievals; returned_at is not a hardware timestamp; repeated_pixels is pixel equality only",
        alignment="pre_* and policy_capture_id feed policy; post_* and observed_capture_id describe returned perception; do not assume post_* follows action",
        video_contract="One encoded frame per valid retrieval; resized, lossy BGR MP4 with header; video_frame is zero-based; timestamps in CSV are authoritative",
        action_contract="action_reported is env effective_action; input_calls are controller API calls, not verified game acceptance",
    )
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    base = _make_env(cfg)
    print(f"Diagnostic directory: {directory}")
    print(f"Frozen policy, {'stochastic' if args.stochastic else 'deterministic'}, training-room stock exchanges.")
    print(f"Weapon state source: {cfg.weapon_state_source}")
    try:
        manifest['env_config'] = asdict(base.config)
        if getattr(base, 'weapon_classifier', None) is not None:
            with Path(base.config.weapon_classifier_checkpoint).open('rb') as handle:
                manifest['weapon_classifier_sha256'] = hashlib.file_digest(handle, 'sha256').hexdigest()
        (directory / 'manifest.json').write_text(
            json.dumps(manifest, indent=2, default=_json_default), encoding='utf-8')
        print(f"Switch to Wu Shang vs attacking Easy Mordex. Starting in {args.delay:g}s.")
        time.sleep(args.delay)
    except BaseException:
        base.close()
        raise
    record_diagnostic(model, base, directory, windows=args.windows, seconds=args.seconds,
                      max_steps=args.max_steps, video=args.video, video_fps=args.video_fps,
                      video_width=args.video_width, stochastic=args.stochastic, seed=args.seed,
                      weapon_crops=args.weapon_crops)
    print(f"Share this diagnostic directory: {directory}")


if __name__ == "__main__":
    main()
