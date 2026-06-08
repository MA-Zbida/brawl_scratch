from __future__ import annotations

import numpy as np

from tools.plot_llc_diagnostics import (
    _active_error_matrix,
    _active_rows_for_features,
    _feature_matrix,
    _json_array,
    _moving_average,
)


def test_plot_helpers_parse_logged_goal_error_arrays() -> None:
    arr = _json_array("[0.1, 0.2, 0.3]", expected=5)
    assert arr.shape == (5,)
    assert np.allclose(arr[:3], [0.1, 0.2, 0.3])
    assert np.allclose(arr[3:], [0.0, 0.0])

    matrix = _active_error_matrix([
        {"active_feature_errors": "[0.1,0.0,0.2]"},
        {"active_feature_errors": "[0.0,0.3,0.0]"},
    ])
    assert matrix.shape[0] == 2
    assert matrix.shape[1] >= 11


def test_moving_average_uses_valid_window() -> None:
    x, y = _moving_average(np.asarray([1.0, 2.0, 3.0], dtype=np.float32), 2)
    assert np.allclose(x, [1.0, 2.0])
    assert np.allclose(y, [1.5, 2.5])


def test_feature_matrix_and_active_rows_parse_goal_columns() -> None:
    rows = [
        {
            "raw_goal_feats": "[0.1,0.2,0.3,0.4]",
            "goal_mask": "[0,0,1,1]",
        },
        {
            "raw_goal_feats": "[0.5,0.6,0.7,0.8]",
            "goal_mask": "[1,1,0,0]",
        },
    ]

    raw = _feature_matrix(rows, "raw_goal_feats")
    mask = _feature_matrix(rows, "goal_mask")
    active_movement = _active_rows_for_features(mask, ("player_x", "player_y"))
    active_recovery = _active_rows_for_features(mask, ("signed_dx_to_ledge", "dy_to_ledge"))

    assert raw.shape == mask.shape
    assert raw.shape[0] == 2
    assert active_movement.tolist() == [True, False]
    assert active_recovery.tolist() == [False, True]
