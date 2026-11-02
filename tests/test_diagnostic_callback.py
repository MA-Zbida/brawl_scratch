from __future__ import annotations

import numpy as np

from brawl.control.action_space import Action
from brawl.pipelines.llc.stage_common import DiagnosticCallback, StageDashboardCallback


def test_diagnostic_callback_reports_discrete_action_mix(capsys) -> None:
    callback = DiagnosticCallback(report_every=4)
    actions = (
        Action.NOOP,
        Action.MOVE_TOWARD,
        Action.JUMP_TOWARD,
        Action.MOVE_AWAY,
    )

    for step, action in enumerate(actions, start=1):
        callback.num_timesteps = step
        callback.locals = {
            "infos": [{"goal_error": 0.2, "goal_success": 0.0}],
            "rewards": np.asarray([0.1], dtype=np.float32),
            "actions": np.asarray([int(action)], dtype=np.int64),
        }
        assert callback._on_step() is True

    output = capsys.readouterr().out
    assert "hdir=[toward:50% away:25% neutral:25%]" in output
    assert "noop:25%" in output


def test_diagnostic_callback_prints_goal_contract_from_info(capsys) -> None:
    callback = DiagnosticCallback(report_every=1)
    goal_target = np.asarray([0.2, 0.7], dtype=np.float32)
    goal_mask = np.asarray([1.0, 1.0], dtype=np.float32)
    callback.num_timesteps = 1
    callback.locals = {
        "infos": [
            {
                "goal_error": 0.1,
                "goal_success": 0.0,
                "goal_target": goal_target,
                "goal_target_world": goal_target,
                "goal_mask": goal_mask,
                "raw_goal_feats": np.asarray([0.25, 0.6], dtype=np.float32),
                "stage_feature_names": ["player_x", "player_y"],
                "goal_success_threshold": 0.04,
                "goal_new_sampled": False,
                "canon_mirrored": 0.0,
            }
        ],
        "rewards": np.asarray([0.1], dtype=np.float32),
        "actions": np.asarray([int(Action.NOOP)], dtype=np.int64),
        # This intentionally contains plausible but wrong goal-like values. The
        # diagnostic must use the authoritative info contract, not fixed slices.
        "new_obs": np.full((1, 168), -0.9, dtype=np.float32),
    }

    assert callback._on_step() is True

    output = capsys.readouterr().out
    assert "goal_target=[0.200 0.700]" in output
    assert "mask=[1.000 1.000]" in output
    assert "player_xy=[0.250 0.600]" in output
    assert "target_xy=[0.200 0.700]" in output
    assert "threshold=0.040" in output
    assert "in_step_resample=0" in output
    assert "goal_target=[-0.9" not in output


def test_dashboard_counts_one_categorical_action_per_step(tmp_path) -> None:
    callback = StageDashboardCallback(
        save_dir=tmp_path,
        model_name="test",
        enable_csv=False,
        enable_plot=False,
    )

    assert callback._cur_ep_action_counts.shape == (27,)

    callback.locals = {
        "infos": [{"stage_action": int(Action.LIGHT_TOWARD)}],
        "rewards": np.asarray([0.0], dtype=np.float32),
        "dones": np.asarray([False]),
    }
    assert callback._on_step() is True

    assert callback._cur_ep_action_counts.sum() == 1
    assert callback._cur_ep_action_counts[int(Action.LIGHT_TOWARD)] == 1
    assert callback._action_entropy(callback._cur_ep_action_counts) == 0.0


def test_dashboard_categorical_entropy_is_normalized() -> None:
    uniform_counts = np.ones((27,), dtype=np.int64)

    assert StageDashboardCallback._action_entropy(uniform_counts) == 1.0


def test_dashboard_csv_includes_weapon_perception_provenance(tmp_path) -> None:
    callback = StageDashboardCallback(
        save_dir=tmp_path,
        model_name="weapon",
        enable_csv=True,
        enable_plot=False,
    )
    callback._on_training_start()
    callback.locals = {
        "infos": [{
            "stage_action": int(Action.PICKUP),
            "weapon_visible_this_frame": 0.0,
            "weapon_pickup_action": 0.0,
            "player_weapon_visual_valid": 1.0,
            "player_weapon_visual_armed": 0.0,
            "player_weapon_visual_confidence": 0.98,
            "player_weapon_visual_fresh": 1.0,
            "weapon_visual_confirmed": 0.0,
            "weapon_success_blocked_visual": 1.0,
        }],
        "rewards": np.asarray([0.0], dtype=np.float32),
        "dones": np.asarray([False]),
    }
    callback._on_step()
    callback._on_training_end()

    header = (tmp_path / "weapon_steps.csv").read_text(encoding="utf-8").splitlines()[0]
    for field in (
        "weapon_visible_this_frame",
        "weapon_pickup_action",
        "player_weapon_visual_valid",
        "player_weapon_visual_armed",
        "player_weapon_visual_confidence",
        "player_weapon_visual_fresh",
        "weapon_visual_confirmed",
        "weapon_success_blocked_visual",
    ):
        assert field in header
