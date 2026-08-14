#!/usr/bin/env python
from __future__ import annotations




# Acquire the screen duplicator before torch loads. On hybrid graphics importing
# torch moves the process to the discrete GPU, and DXGI duplication of the
# integrated-GPU display output then becomes impossible. Must stay above every
# import that pulls in torch (ultralytics, stable_baselines3, env, ...).
#
# The sys.path bootstrap has to come first: running this file directly puts its
# own directory on sys.path, not the repo root, so capture_first is not
# importable without it. sys and pathlib are safe -- neither loads torch.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import capture_first  # noqa: E402,F401  (import order is load-bearing)

import argparse
import ctypes
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import keyboard
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env import BrawlDeepEnv, EnvConfig, NullInputController
from feature_extractor.memory.state_spec import StateSpec
from train.curriculum_config import PHASES, build_phase_spec
from action_space import ACTION_DIM, Action, components, describe
from train.heuristic_teachers import HeuristicTeacher, RecoverySetupController
from train.llc_stage_common import StageGoalEnv
from world_model import WorldReplayRecorder, WorldReplayWriter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect expert demos for BC (all curriculum phases)")
    p.add_argument("--phase", type=str, default=PHASES[0], choices=list(PHASES))
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument(
        "--max-collection-attempts",
        type=int,
        default=0,
        help="Max attempted episodes to gather target accepted demos (0 = auto)",
    )
    p.add_argument("--max-episode-steps", type=int, default=100)
    p.add_argument(
        "--min-episode-steps",
        type=int,
        default=4,
        help="Reject successful demonstrations with fewer recorded transitions",
    )
    p.add_argument("--delay", type=float, default=3.0)
    p.add_argument("--output", type=str, default="")
    p.add_argument(
        "--teacher",
        choices=("heuristic",),
        default="heuristic",
        help=(
            "Only the scripted teacher is supported. Manual recording was dropped with "
            "the move to a single 27-action space: a human cannot select one canonical "
            "action per frame, and inferring intent from raw key state cannot recover "
            "which of several actions sharing those keys was meant."
        ),
    )
    p.add_argument("--move-mouse-to-goal", action="store_true", default=True)
    p.add_argument("--no-move-mouse-to-goal", dest="move_mouse_to_goal", action="store_false")
    p.add_argument("--death-penalty", type=float, default=0.0)
    p.add_argument("--end-episode-on-first-hit", dest="end_episode_on_first_hit", action="store_true")
    p.add_argument("--no-end-episode-on-first-hit", dest="end_episode_on_first_hit", action="store_false")
    p.set_defaults(end_episode_on_first_hit=None)
    p.add_argument("--hit-damage-threshold", type=float, default=1e-3)
    p.add_argument(
        "--enforce-recovery-sequence",
        dest="enforce_recovery_sequence",
        action="store_true",
        help="Only keep episodes that complete a configured sequential recovery objective",
    )
    p.add_argument(
        "--no-enforce-recovery-sequence",
        dest="enforce_recovery_sequence",
        action="store_false",
    )
    p.set_defaults(enforce_recovery_sequence=True)
    p.add_argument(
        "--weapon-hold-steps",
        type=int,
        default=20,
        help="For weapon_acquisition demos, accept only after pickup is held for this many consecutive steps.",
    )
    p.add_argument(
        "--weapon-reset-max-steps",
        type=int,
        default=30,
        help="For heuristic weapon_acquisition demos, max warmup steps used to drop a held weapon before recording.",
    )
    p.add_argument(
        "--weapon-reset-retry-interval",
        type=int,
        default=8,
        help="During weapon reset warmup, retry NUM5 every N steps while still armed.",
    )
    p.add_argument(
        "--weapon-drop-grace-steps",
        type=int,
        default=3,
        help="After pickup, reject the weapon episode only after this many consecutive unarmed observations.",
    )
    p.add_argument(
        "--world-replay",
        type=str,
        default="data/world_replay",
        help="Directory for the persistent transition log. Every stepped transition is "
             "recorded here, including episodes this collector rejects.",
    )
    p.add_argument(
        "--no-world-replay",
        dest="world_replay",
        action="store_const",
        const="",
        help="Disable the persistent transition log for this run.",
    )
    return p.parse_args()


_COMBAT_PHASES = frozenset({"combat_execution", "all_skills_llc"})
_WEAPON_PHASE = "weapon_acquisition"
_WEAPON_IDLE_ACTION = int(Action.NOOP)
_WEAPON_DROP_ACTION = int(Action.PICKUP)
_GOAL_RESAMPLE_RETRIES = 8

_DEFAULT_MAX_STEPS = {
    # Locomotion / weapon: short episodes suffice.
    "default": 100,
    # Combat phases need longer episodes to produce enough hit opportunities.
    "combat": 600,
}


def _build_env(args: argparse.Namespace) -> StageGoalEnv:
    spec = build_phase_spec(
        phase=args.phase,
        death_penalty=float(args.death_penalty),
        terminate_on_death=True,
    )
    # Demo collection should never inject random reset actions.
    spec = replace(spec, reset_perturb_steps=0)

    is_combat = args.phase in _COMBAT_PHASES

    # Auto-pick max_episode_steps when the user didn't explicitly set it.
    max_ep_steps = int(args.max_episode_steps)
    if max_ep_steps == _DEFAULT_MAX_STEPS["default"] and is_combat:
        max_ep_steps = _DEFAULT_MAX_STEPS["combat"]

    config = EnvConfig(
        terminate_on_stock_out=False,
        max_episode_steps=max_ep_steps,
        yolo_infer_every_n_steps=1,
        action_repeat_steps=1,
        action_repeat_min_steps=1,
        action_repeat_max_steps=1,
        tap_latch_steps=1,
    )
    input_controller = NullInputController() if str(args.teacher) == "manual" else None
    base = BrawlDeepEnv(config=config, input_controller=input_controller)
    env = StageGoalEnv(base, spec)

    replay_root = str(getattr(args, "world_replay", "") or "").strip()
    if replay_root:
        # Wrap outside StageGoalEnv so the logged observation is the one the policy
        # actually consumes, goal channels included.
        #
        # This records every transition the env produces, including episodes the
        # collector goes on to reject. That is deliberate: a failed recovery is
        # exactly as informative about physics as a successful one, and demo
        # acceptance is a statement about teacher quality, not about dynamics.
        writer = WorldReplayWriter(
            replay_root,
            phase=str(args.phase),
            metadata={"source": "collect_bc_locomotion_demos", "teacher": str(args.teacher)},
        )
        print(f"World replay log: {writer.dir}")
        return WorldReplayRecorder(env, writer)  # type: ignore[return-value]
    return env


def _screen_size_from_env(env: StageGoalEnv) -> tuple[int, int]:
    base = env.unwrapped
    frame = getattr(base, "_last_frame", None)
    if frame is not None and hasattr(frame, "shape") and len(frame.shape) >= 2:
        h, w = int(frame.shape[0]), int(frame.shape[1])
        if w > 0 and h > 0:
            return w, h

    user32 = ctypes.windll.user32
    return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))


def _set_mouse_to_goal(env: StageGoalEnv, goal_target: np.ndarray, goal_mask: np.ndarray | None = None) -> None:
    if goal_target is None or goal_target.shape[0] < 2:
        return

    goal_xy = np.asarray(goal_target[:2], dtype=np.float32)
    feature_names = list(env.stage_spec.feature_names or [])
    if "player_x" in feature_names and "player_y" in feature_names:
        x_idx = feature_names.index("player_x")
        y_idx = feature_names.index("player_y")
        mask = (
            np.asarray(goal_mask, dtype=np.float32).reshape(-1)
            if goal_mask is not None
            else np.asarray(env.stage_spec.mask, dtype=np.float32).reshape(-1)
        )
        if (
            x_idx < goal_target.shape[0]
            and y_idx < goal_target.shape[0]
            and (x_idx >= mask.shape[0] or y_idx >= mask.shape[0] or mask[x_idx] > 0.0 or mask[y_idx] > 0.0)
        ):
            goal_xy = np.asarray([goal_target[x_idx], goal_target[y_idx]], dtype=np.float32)
        else:
            return

    user32 = ctypes.windll.user32
    try:
        user32.SetProcessDPIAware()
    except Exception:
        pass

    w, h = _screen_size_from_env(env)
    x = int(np.clip(float(goal_xy[0]), 0.0, 1.0) * max(1, w - 1))
    y = int(np.clip(float(goal_xy[1]), 0.0, 1.0) * max(1, h - 1))
    user32.SetCursorPos(x, y)


def _describe_goal(env: StageGoalEnv, goal_target: np.ndarray, goal_mask: np.ndarray | None = None) -> str:
    target = np.asarray(goal_target, dtype=np.float32).reshape(-1)
    mask = (
        np.asarray(goal_mask, dtype=np.float32).reshape(-1)
        if goal_mask is not None
        else np.asarray(env.stage_spec.mask, dtype=np.float32).reshape(-1)
    )
    names = list(env.stage_spec.feature_names or [])
    active: list[str] = []
    for idx, value in enumerate(target.tolist()):
        weight = float(mask[idx]) if idx < mask.shape[0] else 0.0
        if weight <= 0.0:
            continue
        name = names[idx] if idx < len(names) else f"g{idx}"
        active.append(f"{name}={float(value):.3f}")
        if len(active) >= 4:
            break
    if active:
        return "goal(" + ", ".join(active) + ")"
    if target.shape[0] >= 2:
        return f"goal0/1=({float(target[0]):.3f}, {float(target[1]):.3f})"
    return "goal(n/a)"


def _phase_uses_player_xy_goal(spec) -> bool:
    feature_names = list(spec.feature_names or [])
    if not feature_names:
        return False

    idx = {name: i for i, name in enumerate(feature_names)}
    x_idx = idx.get("player_x")
    y_idx = idx.get("player_y")
    if x_idx is None or y_idx is None:
        return False

    mask = np.asarray(spec.mask, dtype=np.float32).reshape(-1)
    return bool(mask[x_idx] > 0.0 or mask[y_idx] > 0.0)


def _resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output.strip():
        out = Path(args.output)
    else:
        out = Path("train/models") / f"{args.phase}_demos.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _obs_has_weapon(obs: np.ndarray) -> bool:
    try:
        return bool(StateSpec.get(np.asarray(obs, dtype=np.float32), "player_has_weapon") > 0.5)
    except Exception:
        return False


@dataclass
class CollectionRejections:
    """Rejection counters that are persisted with the demonstration archive."""

    episodes_rejected_trivial: int = 0

    def reject_trivial(self) -> None:
        self.episodes_rejected_trivial += 1

    def as_metadata(self) -> dict[str, np.ndarray]:
        return {
            "episodes_rejected_trivial": np.asarray(
                [self.episodes_rejected_trivial],
                dtype=np.int64,
            )
        }


@dataclass
class RecoveryArming:
    """Keep spawn/onstage and death/respawn transitions out of recovery demos."""

    required: bool
    armed: bool = False

    def should_record(self, obs: np.ndarray, info: dict) -> bool:
        if not self.required:
            return True
        if not self.armed:
            offstage = bool(
                StateSpec.get(np.asarray(obs, dtype=np.float32), "player_is_offstage")
                > 0.5
            )
            # Position memory intentionally survives short detector gaps, including
            # death/respawn. An offstage bit alone can therefore describe a stale
            # pre-death location while the wrapper has no active goal. Fail closed
            # so every recorded row has a real, controllable recovery state.
            self.armed = bool(offstage and _recording_state_ready(info))
        return self.armed


def _recording_state_ready(info: dict) -> bool:
    """Return whether a transition has both a real goal and a controllable player."""
    goal_active = float(info.get("goal_active", 0.0)) > 0.5
    player_exists = float(info.get("player_exists", 0.0)) > 0.5
    respawn_complete = (
        float(info.get("player_respawn_timer", float("inf"))) <= 1e-6
    )
    return bool(goal_active and player_exists and respawn_complete)


def wait_for_recording_ready(
    env: StageGoalEnv,
    obs: np.ndarray,
    info: dict,
    *,
    max_steps: int,
) -> tuple[np.ndarray, dict, bool, int]:
    """Discard reset/respawn frames until the wrapper publishes an active goal."""
    warmup_steps = 0
    while not _recording_state_ready(info) and warmup_steps < max(0, int(max_steps)):
        # The all-zero target is an inactive sentinel, not a training goal. NOOP is
        # safest while the player cannot act, and none of these transitions belong
        # in the behaviour-cloning archive.
        obs, _reward, terminated, truncated, info = env.step(int(Action.NOOP))
        warmup_steps += 1
        if terminated or truncated:
            break
    return obs, info, _recording_state_ready(info), warmup_steps


def prepare_nontrivial_goal(
    env: StageGoalEnv,
    obs: np.ndarray,
    info: dict,
    *,
    max_retries: int = _GOAL_RESAMPLE_RETRIES,
) -> tuple[np.ndarray, dict, bool, int]:
    """Resample a reset-time target until the current state is outside success."""
    retries = 0
    while env.goal_error(obs, info) < float(env.stage_spec.success_threshold):
        if retries >= max(0, int(max_retries)):
            return obs, info, False, retries
        obs, info = env.resample_goal(obs, info)
        retries += 1
    return obs, info, True, retries


def _prepare_weapon_episode_start(
    env: StageGoalEnv,
    args: argparse.Namespace,
    obs: np.ndarray,
    info: dict,
) -> tuple[np.ndarray, dict, bool, int]:
    """Make heuristic weapon demos start unarmed; warmup steps are not recorded."""
    if not _obs_has_weapon(obs):
        return obs, info, True, 0

    if str(args.teacher) != "heuristic":
        return obs, info, False, 0

    max_steps = max(0, int(args.weapon_reset_max_steps))
    retry_interval = max(1, int(args.weapon_reset_retry_interval))
    warmup_steps = 0

    for step_idx in range(max_steps):
        action = _WEAPON_DROP_ACTION if step_idx % retry_interval == 0 else _WEAPON_IDLE_ACTION
        obs, _reward, _terminated, _truncated, info = env.step(action)
        warmup_steps += 1
        if not _obs_has_weapon(obs):
            obs, info = env.reset()
            return obs, info, not _obs_has_weapon(obs), warmup_steps

    return obs, info, not _obs_has_weapon(obs), warmup_steps


def resolve_allowed_actions(spec) -> list[int]:
    """Actions this phase may execute, as indices into the 27-action space.

    StageGoalEnv is what actually enforces the restriction; the collector derives it
    only to report a run's constraints. Kept at module scope so it can be tested
    without standing up a live environment -- the previous version read
    `spec.allowed_attack_actions`, a field that no longer exists, and nothing caught
    it because no test ever ran the collector past argument parsing.
    """
    if spec.allowed_actions is not None:
        return sorted({int(v) for v in spec.allowed_actions})

    return [
        a for a in range(ACTION_DIM)
        if not (spec.disable_attack and (components(a).light or components(a).heavy))
        and not (spec.disable_dodge and components(a).dodge)
        and not (spec.disable_jump and components(a).jump)
    ]


def main() -> None:
    args = parse_args()
    out_path = _resolve_output_path(args)
    hit_damage_threshold = float(max(0.0, args.hit_damage_threshold))
    phase_lower = str(args.phase).strip().lower()
    is_damage = phase_lower in _COMBAT_PHASES
    is_weapon_acquisition = phase_lower == _WEAPON_PHASE
    enforce_recovery_sequence = bool(args.enforce_recovery_sequence and phase_lower == "recovery_mastery")
    min_episode_steps = max(1, int(args.min_episode_steps))
    weapon_hold_steps_required = max(1, int(args.weapon_hold_steps))
    weapon_drop_grace_steps = max(0, int(args.weapon_drop_grace_steps))

    target_episodes = max(1, int(args.episodes))
    max_collection_attempts = int(args.max_collection_attempts)
    if max_collection_attempts <= 0:
        if enforce_recovery_sequence:
            max_collection_attempts = max(target_episodes, target_episodes * 5)
        else:
            # Only success episodes are accepted; budget for some max-step timeouts.
            max_collection_attempts = max(target_episodes, target_episodes * 3)

    # Default: never end on first hit; BC episodes should mirror PPO episodes,
    # which continue after hits until goal progress / max steps.
    if args.end_episode_on_first_hit is None:
        end_episode_on_first_hit = False
    else:
        end_episode_on_first_hit = bool(args.end_episode_on_first_hit)

    env = _build_env(args)
    spec = env.stage_spec
    allowed_actions = resolve_allowed_actions(spec)

    # Movement is the one phase whose goal is an absolute screen position, so the
    # cursor is the only way to see where the target actually is while collecting.
    # Forced on regardless of the flag: a movement run without it is not reviewable,
    # and a bad movement dataset is invisible until BC has already learned it.
    mouse_guidance_required = phase_lower == "movement_fluency"
    mouse_guidance_enabled = bool(args.move_mouse_to_goal or mouse_guidance_required)
    if mouse_guidance_enabled and not _phase_uses_player_xy_goal(spec):
        if mouse_guidance_required:
            raise RuntimeError(
                "movement_fluency requires mouse guidance, but its goal does not use "
                "player_x/player_y. The phase spec and the collector disagree."
            )
        mouse_guidance_enabled = False


    print("=" * 68)
    print(f"BC DEMO COLLECTION - {args.phase.upper()}")
    print(f"Action space: Discrete({ACTION_DIM}) -- canonical TOWARD/AWAY")
    print(f"Allowed this phase ({len(allowed_actions)}): {', '.join(describe(a) for a in allowed_actions[:8])}{' ...' if len(allowed_actions) > 8 else ''}")
    print("Action source: scripted teacher")
    print("Input mode: the teacher drives Brawlhalla through env key injection")
    print(f"Target episodes (accepted): {target_episodes} | Output: {out_path}")
    print(f"Collection attempt budget: {max_collection_attempts}")
    eff_max_steps = int(args.max_episode_steps)
    if eff_max_steps == _DEFAULT_MAX_STEPS["default"] and is_damage:
        eff_max_steps = _DEFAULT_MAX_STEPS["combat"]
    print(f"Max episode steps: {eff_max_steps}")
    print("Goal resampling: on success; otherwise a new goal comes with the next episode reset")
    if mouse_guidance_enabled:
        print("Mouse guidance: enabled (cursor moves to target_x,target_y)")
    elif args.move_mouse_to_goal:
        print("Mouse guidance: disabled (phase goals are relational/non-XY)")
    if end_episode_on_first_hit:
        print(f"Episode end action: stop on first hit (op_delta_damage >= {hit_damage_threshold:.4f})")
    if enforce_recovery_sequence:
        print("Recovery quality gate: ON (keep only offstage->onstage sequence-complete episodes)")
    elif phase_lower == "recovery_mastery":
        print(
            "WARNING: recovery sequence enforcement is disabled. The resulting archive "
            "can contain spawn-success episodes that never demonstrate recovery."
        )
    print(f"Minimum recorded episode steps: {min_episode_steps}")
    if is_weapon_acquisition:
        print(
            "Weapon quality gate: start unarmed -> pickup transition -> "
            f"hold for {weapon_hold_steps_required} consecutive steps "
            f"(drop grace={weapon_drop_grace_steps})"
        )
        if args.teacher == "heuristic":
            print("Weapon reset warmup: if an episode starts armed, tap NUM5 before recording the next episode")
    print("Press Ctrl+C to stop and save partial data.")
    print("=" * 68)
    print(f"Starting in {args.delay:.1f}s...")
    time.sleep(max(0.0, float(args.delay)))

    obs_buf: list[np.ndarray] = []
    act_multi_buf: list[np.ndarray] = []
    done_buf: list[bool] = []
    goal_xy_buf: list[np.ndarray] = []
    goal_target_buf: list[np.ndarray] = []
    goal_mask_buf: list[np.ndarray] = []
    goal_type_buf: list[str] = []
    goal_type_index_buf: list[int] = []
    op_ko_buf: list[bool] = []
    sequential_phase_buf: list[int] = []
    sequential_step1_transition_buf: list[bool] = []
    sequential_step1_completed_buf: list[bool] = []
    sequential_terminal_success_buf: list[bool] = []

    step_total = 0
    episodes_collected = 0
    episodes_attempted = 0
    episodes_rejected_recovery = 0
    episodes_rejected_weapon = 0
    rejections = CollectionRejections()
    weapon_reset_failures = 0
    episodes_with_hit = 0
    episodes_ended_on_first_hit = 0

    try:
        while episodes_collected < target_episodes and episodes_attempted < max_collection_attempts:
            attempt_id = episodes_attempted + 1
            obs, info = env.reset()
            obs, info, recording_ready, readiness_steps = wait_for_recording_ready(
                env,
                obs,
                info,
                max_steps=eff_max_steps,
            )
            step_total += readiness_steps
            if not recording_ready:
                episodes_attempted += 1
                print(
                    f"Attempt {attempt_id}/{max_collection_attempts} -> accepted=0 "
                    f"accepted_total={episodes_collected}/{target_episodes} "
                    f"steps=0 reason=goal_never_activated "
                    f"warmup_steps={readiness_steps}"
                )
                continue

            if is_weapon_acquisition:
                obs, info, weapon_start_ready, weapon_warmup_steps = _prepare_weapon_episode_start(env, args, obs, info)
                step_total += weapon_warmup_steps
                if not weapon_start_ready:
                    episodes_attempted += 1
                    episodes_rejected_weapon += 1
                    weapon_reset_failures += 1
                    print(
                        f"Attempt {attempt_id}/{max_collection_attempts} -> accepted=0 "
                        f"accepted_total={episodes_collected}/{target_episodes} "
                        f"steps=0 reason=weapon_reset_failed warmup_steps={weapon_warmup_steps}"
                    )
                    continue
                # Dropping a weapon resets the wrapped environment; that reset may
                # itself land inside a death/respawn window and clear the goal.
                obs, info, recording_ready, readiness_steps = wait_for_recording_ready(
                    env,
                    obs,
                    info,
                    max_steps=eff_max_steps,
                )
                step_total += readiness_steps
                if not recording_ready:
                    episodes_attempted += 1
                    episodes_rejected_weapon += 1
                    print(
                        f"Attempt {attempt_id}/{max_collection_attempts} -> accepted=0 "
                        f"accepted_total={episodes_collected}/{target_episodes} "
                        f"steps=0 reason=goal_never_activated_after_weapon_reset "
                        f"warmup_steps={readiness_steps}"
                    )
                    continue

            teacher = HeuristicTeacher()
            recovery_setup = RecoverySetupController()

            # Recovery has a later recording boundary: the episode starts only once
            # the player is offstage. Its arming gate is therefore the non-vacuity
            # check; resampling the fixed ledge target at spawn would reject forever.
            if phase_lower != "recovery_mastery":
                obs, info, goal_ready, goal_retries = prepare_nontrivial_goal(
                    env,
                    obs,
                    info,
                )
                if not goal_ready:
                    episodes_attempted += 1
                    rejections.reject_trivial()
                    print(
                        f"Attempt {attempt_id}/{max_collection_attempts} -> accepted=0 "
                        f"accepted_total={episodes_collected}/{target_episodes} "
                        f"steps=0 reason=goal_satisfied_at_reset "
                        f"resample_retries={goal_retries}"
                    )
                    continue

            done = False
            ep_steps = 0
            ep_had_hit = False
            end_reason = "unknown"
            ep_step1_transition_seen = False
            ep_terminal_success_seen = False
            recovery_arming = RecoveryArming(required=enforce_recovery_sequence)
            recovery_arm_transition_pending = False
            ep_weapon_started_unarmed = bool(not _obs_has_weapon(obs))
            ep_weapon_prev_has_weapon = bool(_obs_has_weapon(obs))
            ep_weapon_pickup_seen = False
            ep_weapon_hold_steps = 0
            ep_weapon_lost_steps_after_pickup = 0
            ep_weapon_hold_success = False
            ep_weapon_dropped_after_pickup = False

            ep_obs_buf: list[np.ndarray] = []
            ep_act_multi_buf: list[np.ndarray] = []
            ep_done_buf: list[bool] = []
            ep_goal_xy_buf: list[np.ndarray] = []
            ep_goal_target_buf: list[np.ndarray] = []
            ep_goal_mask_buf: list[np.ndarray] = []
            ep_goal_type_buf: list[str] = []
            ep_goal_type_index_buf: list[int] = []
            ep_op_ko_buf: list[bool] = []
            ep_seq_phase_buf: list[int] = []
            ep_seq_step1_transition_buf: list[bool] = []
            ep_seq_step1_completed_buf: list[bool] = []
            ep_seq_terminal_success_buf: list[bool] = []

            goal_target = np.asarray(info.get("goal_target", np.zeros(2, dtype=np.float32)), dtype=np.float32)
            goal_mask = np.asarray(info.get("goal_mask", np.zeros_like(goal_target)), dtype=np.float32)

            if enforce_recovery_sequence:
                recovery_arming.should_record(obs, info)
                if recovery_arming.armed:
                    ep_step1_transition_seen = True
                    recovery_arm_transition_pending = True

                # Ignore reset-time goal success while arming. The ledge target is
                # intentionally already satisfied onstage, and these transitions are
                # never written to the archive.
                while not recovery_arming.armed and ep_steps < eff_max_steps:
                    action_id = recovery_setup.action(obs)
                    next_obs, _reward, terminated, _truncated, info = env.step(action_id)
                    obs = next_obs
                    ep_steps += 1
                    step_total += 1
                    if recovery_arming.should_record(obs, info):
                        ep_step1_transition_seen = True
                        recovery_arm_transition_pending = True
                        break
                    if terminated:
                        end_reason = "terminated_before_offstage"
                        break

                if not recovery_arming.armed:
                    episodes_attempted += 1
                    episodes_rejected_recovery += 1
                    print(
                        f"Attempt {attempt_id}/{max_collection_attempts} -> accepted=0 "
                        f"accepted_total={episodes_collected}/{target_episodes} "
                        f"steps=0 reason=never_offstage arming_steps={ep_steps}"
                    )
                    continue

                goal_target = np.asarray(info.get("goal_target", goal_target), dtype=np.float32)
                goal_mask = np.asarray(info.get("goal_mask", goal_mask), dtype=np.float32)
            if mouse_guidance_enabled:
                _set_mouse_to_goal(env, goal_target, goal_mask)

            while not done:
                # The teacher already holds position without re-pressing pickup once
                # armed. Overriding it with a forced NOOP here used to fill the hold
                # window with idle frames, and since only successful episodes are
                # saved, those became a behaviour-cloning anchor that taught the agent
                # to freeze after every pickup.
                action = teacher.action(args.phase, obs, info)
                action_id = int(action)

                ep_obs_buf.append(np.asarray(obs, dtype=np.float32).copy())
                ep_act_multi_buf.append(int(action_id))
                ep_goal_xy_buf.append(np.asarray(goal_target[:2], dtype=np.float32).copy())
                ep_goal_target_buf.append(np.asarray(goal_target, dtype=np.float32).copy())
                ep_goal_mask_buf.append(np.asarray(goal_mask, dtype=np.float32).copy())
                ep_goal_type_buf.append(str(info.get("goal_type", "unknown")))
                ep_goal_type_index_buf.append(int(info.get("goal_type_index", -1)))

                next_obs, _reward, terminated, truncated, info = env.step(action_id)
                op_delta_damage = float(max(0.0, info.get("op_delta_damage", 0.0)))
                op_ko_event = float(info.get("op_stock_lost_step", 0.0)) > 0.0
                ep_op_ko_buf.append(op_ko_event)
                hit_event = op_delta_damage >= hit_damage_threshold
                if hit_event:
                    ep_had_hit = True

                if is_weapon_acquisition:
                    curr_has_weapon = _obs_has_weapon(next_obs)
                    pickup_now = bool(ep_weapon_started_unarmed and curr_has_weapon and not ep_weapon_prev_has_weapon)
                    if pickup_now:
                        ep_weapon_pickup_seen = True
                    if ep_weapon_pickup_seen:
                        if curr_has_weapon:
                            ep_weapon_hold_steps += 1
                            ep_weapon_lost_steps_after_pickup = 0
                        else:
                            ep_weapon_hold_steps = 0
                            ep_weapon_lost_steps_after_pickup += 1
                            if ep_weapon_lost_steps_after_pickup > weapon_drop_grace_steps:
                                ep_weapon_dropped_after_pickup = True
                    ep_weapon_hold_success = bool(ep_weapon_pickup_seen and ep_weapon_hold_steps >= weapon_hold_steps_required)
                    ep_weapon_prev_has_weapon = curr_has_weapon

                seq_phase = int(info.get("sequential_phase", 0))
                seq_step1_transition = bool(float(info.get("sequential_step1_transition", 0.0)) > 0.5)
                seq_step1_completed = bool(float(info.get("sequential_step1_completed", 0.0)) > 0.5)
                seq_terminal_success = bool(
                    float(info.get("terminal_success", 0.0)) > 0.5
                    and seq_phase == 2
                )
                if enforce_recovery_sequence:
                    player_onstage = bool(
                        StateSpec.get(np.asarray(next_obs, dtype=np.float32), "player_is_offstage")
                        <= 0.5
                    )
                    seq_phase = 2
                    seq_step1_transition = recovery_arm_transition_pending
                    seq_step1_completed = True
                    seq_terminal_success = bool(
                        player_onstage
                        and float(info.get("goal_success", 0.0)) > 0.5
                    )
                    recovery_arm_transition_pending = False
                ep_seq_phase_buf.append(seq_phase)
                ep_seq_step1_transition_buf.append(seq_step1_transition)
                ep_seq_step1_completed_buf.append(seq_step1_completed)
                ep_seq_terminal_success_buf.append(seq_terminal_success)
                ep_step1_transition_seen = bool(ep_step1_transition_seen or seq_step1_transition)
                ep_terminal_success_seen = bool(ep_terminal_success_seen or seq_terminal_success)

                terminated_by_hit = bool(end_episode_on_first_hit and hit_event)
                if is_weapon_acquisition:
                    done = bool(terminated or ep_weapon_hold_success or ep_weapon_dropped_after_pickup)
                elif enforce_recovery_sequence:
                    # Goal success near the ledge is common while still offstage. It
                    # becomes terminal only after the return to stage is observed.
                    done = bool(terminated or seq_terminal_success)
                else:
                    done = bool(terminated or truncated or terminated_by_hit)
                ep_done_buf.append(done)

                if is_weapon_acquisition and ep_weapon_hold_success:
                    end_reason = "weapon_hold_success"
                elif is_weapon_acquisition and ep_weapon_dropped_after_pickup:
                    end_reason = "weapon_dropped_after_pickup"
                elif terminated_by_hit:
                    end_reason = "first_hit"
                elif terminated:
                    end_reason = "env_terminated"
                elif truncated:
                    end_reason = "env_truncated"

                if bool(info.get("goal_new_sampled", False)):
                    goal_target = np.asarray(info.get("goal_target", goal_target), dtype=np.float32)
                    goal_mask = np.asarray(info.get("goal_mask", goal_mask), dtype=np.float32)
                    if mouse_guidance_enabled:
                        _set_mouse_to_goal(env, goal_target, goal_mask)

                obs = next_obs
                ep_steps += 1
                step_total += 1

                if step_total % 100 == 0:
                    if is_damage:
                        # Show combat-relevant info for damage phases.
                        op_dd = float(info.get("op_delta_damage", 0.0))
                        g_err = float(info.get("goal_error", 0.0))
                        print(
                            f"steps={step_total} accepted={episodes_collected}/{target_episodes} "
                            f"attempt={attempt_id}/{max_collection_attempts} ep_steps={ep_steps} "
                            f"op_delta_dmg={op_dd:.4f} goal_err={g_err:.3f}"
                        )
                    else:
                        print(
                            f"steps={step_total} accepted={episodes_collected}/{target_episodes} "
                            f"attempt={attempt_id}/{max_collection_attempts} ep_steps={ep_steps} "
                            f"{_describe_goal(env, goal_target, goal_mask)}"
                        )

                if ep_steps >= int(args.max_episode_steps):
                    if not done and len(ep_done_buf) > 0:
                        done = True
                        ep_done_buf[-1] = True
                        end_reason = "collector_step_cap"
                    break

            episodes_attempted += 1
            if ep_had_hit:
                episodes_with_hit += 1
            if end_reason == "first_hit":
                episodes_ended_on_first_hit += 1

            accept_episode = bool(float(info.get("goal_success", 0.0)) > 0.5)
            if is_weapon_acquisition:
                accept_episode = bool(
                    ep_weapon_started_unarmed
                    and ep_weapon_pickup_seen
                    and ep_weapon_hold_success
                    and not ep_weapon_dropped_after_pickup
                )
                if not accept_episode:
                    episodes_rejected_weapon += 1
            if enforce_recovery_sequence:
                accept_episode = bool(accept_episode and ep_step1_transition_seen and ep_terminal_success_seen)
                if not accept_episode:
                    episodes_rejected_recovery += 1

            recorded_steps = len(ep_obs_buf)
            rejected_as_trivial = recorded_steps < min_episode_steps
            if rejected_as_trivial:
                accept_episode = False
                rejections.reject_trivial()

            if accept_episode and len(ep_obs_buf) > 0:
                obs_buf.extend(ep_obs_buf)
                act_multi_buf.extend(ep_act_multi_buf)
                done_buf.extend(ep_done_buf)
                goal_xy_buf.extend(ep_goal_xy_buf)
                goal_target_buf.extend(ep_goal_target_buf)
                goal_mask_buf.extend(ep_goal_mask_buf)
                goal_type_buf.extend(ep_goal_type_buf)
                goal_type_index_buf.extend(ep_goal_type_index_buf)
                op_ko_buf.extend(ep_op_ko_buf)
                sequential_phase_buf.extend(ep_seq_phase_buf)
                sequential_step1_transition_buf.extend(ep_seq_step1_transition_buf)
                sequential_step1_completed_buf.extend(ep_seq_step1_completed_buf)
                sequential_terminal_success_buf.extend(ep_seq_terminal_success_buf)
                episodes_collected += 1

            print(
                f"Attempt {attempt_id}/{max_collection_attempts} -> "
                f"accepted={int(accept_episode)} "
                f"accepted_total={episodes_collected}/{target_episodes} "
                f"steps={recorded_steps} env_steps={ep_steps} reason={end_reason} "
                f"had_hit={int(ep_had_hit)} trivial={int(rejected_as_trivial)}"
                + (
                    f" seq(step1={int(ep_step1_transition_seen)}, step2={int(ep_terminal_success_seen)})"
                    if enforce_recovery_sequence
                    else ""
                )
                + (
                    f" weapon(start_unarmed={int(ep_weapon_started_unarmed)}, "
                    f"pickup={int(ep_weapon_pickup_seen)}, "
                    f"hold={ep_weapon_hold_steps}/{weapon_hold_steps_required}, "
                    f"dropped={int(ep_weapon_dropped_after_pickup)})"
                    if is_weapon_acquisition
                    else ""
                )
            )
        if episodes_collected < target_episodes:
            print(
                f"Collection stopped with {episodes_collected}/{target_episodes} accepted episodes "
                f"after {episodes_attempted}/{max_collection_attempts} attempts."
            )

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving partial dataset...")
    finally:
        env.close()

    if episodes_attempted > 0 and end_episode_on_first_hit:
        hit_ratio = float(episodes_with_hit) / float(max(1, episodes_attempted))
        first_hit_end_ratio = float(episodes_ended_on_first_hit) / float(max(1, episodes_attempted))
        print(
            f"Hit summary: hit_episodes={episodes_with_hit}/{episodes_attempted} ({hit_ratio:.2%}), "
            f"ended_on_first_hit={episodes_ended_on_first_hit}/{episodes_attempted} ({first_hit_end_ratio:.2%})"
        )

    if enforce_recovery_sequence:
        print(
            "Recovery sequence summary: "
            f"accepted={episodes_collected}/{target_episodes}, "
            f"attempted={episodes_attempted}/{max_collection_attempts}, "
            f"rejected={episodes_rejected_recovery}"
        )
    if is_weapon_acquisition:
        print(
            "Weapon acquisition summary: "
            f"accepted={episodes_collected}/{target_episodes}, "
            f"attempted={episodes_attempted}/{max_collection_attempts}, "
            f"rejected={episodes_rejected_weapon}, "
            f"reset_failures={weapon_reset_failures}, "
            f"hold_steps_required={weapon_hold_steps_required}, "
            f"drop_grace_steps={weapon_drop_grace_steps}"
        )
    print(
        "Trivial episode summary: "
        f"rejected={rejections.episodes_rejected_trivial}, "
        f"minimum_recorded_steps={min_episode_steps}"
    )

    if len(obs_buf) == 0:
        print("No samples collected; nothing to save.")
        return

    lengths = {
        "obs": len(obs_buf),
        "actions_multi": len(act_multi_buf),
        "dones": len(done_buf),
        "goal_xy": len(goal_xy_buf),
        "goal_target": len(goal_target_buf),
        "goal_mask": len(goal_mask_buf),
        "goal_type": len(goal_type_buf),
        "goal_type_index": len(goal_type_index_buf),
        "op_ko_events": len(op_ko_buf),
        "sequential_phase": len(sequential_phase_buf),
        "sequential_step1_transition": len(sequential_step1_transition_buf),
        "sequential_step1_completed": len(sequential_step1_completed_buf),
        "sequential_terminal_success": len(sequential_terminal_success_buf),
    }
    aligned_n = min(lengths.values())
    if aligned_n <= 0:
        print("No aligned samples to save after buffer consistency check.")
        return
    if any(v != aligned_n for v in lengths.values()):
        print(f"Buffer mismatch detected ({lengths}); trimming all arrays to {aligned_n} samples.")
        obs_buf = obs_buf[:aligned_n]
        act_multi_buf = act_multi_buf[:aligned_n]
        done_buf = done_buf[:aligned_n]
        goal_xy_buf = goal_xy_buf[:aligned_n]
        goal_target_buf = goal_target_buf[:aligned_n]
        goal_mask_buf = goal_mask_buf[:aligned_n]
        goal_type_buf = goal_type_buf[:aligned_n]
        goal_type_index_buf = goal_type_index_buf[:aligned_n]
        op_ko_buf = op_ko_buf[:aligned_n]
        sequential_phase_buf = sequential_phase_buf[:aligned_n]
        sequential_step1_transition_buf = sequential_step1_transition_buf[:aligned_n]
        sequential_step1_completed_buf = sequential_step1_completed_buf[:aligned_n]
        sequential_terminal_success_buf = sequential_terminal_success_buf[:aligned_n]
        if len(done_buf) > 0:
            done_buf[-1] = True

    obs_arr = np.stack(obs_buf).astype(np.float32)
    act_multi_arr = np.asarray(act_multi_buf, dtype=np.int64)
    act_arr = act_multi_arr
    action_encoding = "discrete27"
    done_arr = np.asarray(done_buf, dtype=bool)
    goal_xy_arr = np.stack(goal_xy_buf).astype(np.float32)
    goal_target_arr = np.stack(goal_target_buf).astype(np.float32)
    goal_mask_arr = np.stack(goal_mask_buf).astype(np.float32)
    goal_type_arr = np.asarray(goal_type_buf, dtype="<U32")
    goal_type_index_arr = np.asarray(goal_type_index_buf, dtype=np.int64)
    op_ko_arr = np.asarray(op_ko_buf, dtype=bool)
    sequential_phase_arr = np.asarray(sequential_phase_buf, dtype=np.int64)
    sequential_step1_transition_arr = np.asarray(sequential_step1_transition_buf, dtype=bool)
    sequential_step1_completed_arr = np.asarray(sequential_step1_completed_buf, dtype=bool)
    sequential_terminal_success_arr = np.asarray(sequential_terminal_success_buf, dtype=bool)

    np.savez_compressed(
        str(out_path),
        obs=obs_arr,
        actions=act_arr,
        actions_discrete=act_multi_arr,
        action_encoding=np.asarray([action_encoding]),
        dones=done_arr,
        goal_xy=goal_xy_arr,
        goal_target=goal_target_arr,
        goal_mask=goal_mask_arr,
        goal_type=goal_type_arr,
        goal_type_index=goal_type_index_arr,
        op_ko_events=op_ko_arr,
        sequential_phase=sequential_phase_arr,
        sequential_step1_transition=sequential_step1_transition_arr,
        sequential_step1_completed=sequential_step1_completed_arr,
        sequential_terminal_success=sequential_terminal_success_arr,
        recovery_sequence_enforced=np.asarray([1 if enforce_recovery_sequence else 0], dtype=np.int64),
        episodes_target=np.asarray([target_episodes], dtype=np.int64),
        episodes_attempted=np.asarray([episodes_attempted], dtype=np.int64),
        episodes_collected=np.asarray([episodes_collected], dtype=np.int64),
        episodes_rejected=np.asarray([episodes_rejected_recovery], dtype=np.int64),
        episodes_rejected_weapon=np.asarray([episodes_rejected_weapon], dtype=np.int64),
        **rejections.as_metadata(),
        weapon_reset_failures=np.asarray([weapon_reset_failures], dtype=np.int64),
        weapon_hold_steps_required=np.asarray([weapon_hold_steps_required], dtype=np.int64),
        weapon_drop_grace_steps=np.asarray([weapon_drop_grace_steps], dtype=np.int64),
        teacher=np.asarray([args.teacher]),
        phase=np.asarray([args.phase]),
    )

    print(f"Saved {args.phase} demos")
    print(f"  path   : {out_path}")
    print(f"  obs    : {obs_arr.shape}")
    print(f"  actions(saved): {act_arr.shape} [{action_encoding}]")
    print(f"  action range: {int(act_multi_arr.min())}..{int(act_multi_arr.max())} of {ACTION_DIM}")
    if enforce_recovery_sequence:
        step1_count = int(sequential_step1_transition_arr.sum())
        step2_term_count = int(sequential_terminal_success_arr.sum())
        print("  recovery_seq: enforced=1")
        print(f"  recovery_seq step1_transitions: {step1_count}")
        print(f"  recovery_seq terminal_step2_success: {step2_term_count}")


if __name__ == "__main__":
    main()
