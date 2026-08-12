"""Regression guard for `evaluate.py`'s policy-output conversion.

The `Discrete(27)` migration left `_to_env_action` demanding four components.
Nothing caught it because no test exercised the top-level evaluator, so every
`python evaluate.py` run would have raised on its first step -- after loading a
checkpoint and opening the game.
"""

from __future__ import annotations

import numpy as np
import pytest

from action_space import ACTION_DIM, Action
from evaluate import _to_env_action


@pytest.mark.parametrize(
    "prediction",
    [
        np.asarray(int(Action.LIGHT_TOWARD), dtype=np.int64),      # 0-d, unbatched predict
        np.asarray([int(Action.LIGHT_TOWARD)], dtype=np.int64),    # (1,), vec-env predict
        np.asarray([[int(Action.LIGHT_TOWARD)]], dtype=np.int64),  # (1,1)
        int(Action.LIGHT_TOWARD),                                  # plain int
    ],
)
def test_accepts_every_shape_predict_returns(prediction) -> None:
    action = _to_env_action(prediction)
    assert action == int(Action.LIGHT_TOWARD)
    assert isinstance(action, int)


def test_rejects_legacy_multidiscrete_prediction() -> None:
    """A 4-vector means the checkpoint predates the migration -- say so, don't guess."""
    with pytest.raises(ValueError, match="predates the Discrete migration"):
        _to_env_action(np.asarray([1, 0, 0, 2], dtype=np.int64))


def test_rejects_action_outside_the_space() -> None:
    with pytest.raises(ValueError, match="outside"):
        _to_env_action(np.asarray([ACTION_DIM], dtype=np.int64))
