"""Indicator geometry calibration.

The Kaggle notebook's first attempt at this produced
``INDICATOR_MAX_MATCH_SCORE = 0.0`` -- a constant that rejects every match -- because it
derived thresholds from min/max of a set contaminated by its own mis-pairings. These tests
pin the two properties that prevent a repeat: ambiguous frames are excluded rather than
guessed, and thresholds never come from extremes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.calibrate_indicator_geometry import (
    CHARACTER_ID,
    INDICATOR_ID,
    calibrate,
    derive_constants,
    score,
    threshold_quality,
)


def write_frame(tmp_path: Path, name: str, rows: list[tuple[int, float, float, float, float]]) -> Path:
    p = tmp_path / f"{name}.txt"
    p.write_text("\n".join(f"{c} {cx} {cy} {w} {h}" for c, cx, cy, w, h in rows), encoding="utf-8")
    return p


def clean_frame(cx: float, ind_cy: float = 0.53, char_cy: float = 0.60):
    """One indicator directly above one character."""
    return [
        (INDICATOR_ID, cx, ind_cy, 0.02, 0.02),
        (CHARACTER_ID, cx, char_cy, 0.05, 0.09),
    ]


# ── pairing ─────────────────────────────────────────────────────────────────

def test_pairs_indicator_with_the_character_below_it(tmp_path):
    files = [write_frame(tmp_path, f"f{i}", clean_frame(0.30 + 0.01 * i)) for i in range(20)]
    cal = calibrate(files)

    assert cal.n_usable == 20
    assert cal.n_ambiguous == 0
    assert np.allclose(cal.dx, 0.0, atol=1e-6)
    assert cal.dy == pytest.approx(0.07, abs=1e-6)


def test_far_apart_characters_are_unambiguous(tmp_path):
    """A distant second character is not a real competitor."""
    rows = clean_frame(0.30) + [(CHARACTER_ID, 0.85, 0.60, 0.05, 0.09)]
    cal = calibrate([write_frame(tmp_path, "f", rows)])

    assert cal.n_usable == 1
    assert cal.false_scores.size == 1
    assert cal.true_scores[0] < cal.false_scores[0]


def test_ambiguous_frames_are_excluded_not_guessed(tmp_path):
    """Two characters nearly equidistant: guessing here is what corrupted the tails."""
    rows = [
        (INDICATOR_ID, 0.50, 0.53, 0.02, 0.02),
        (CHARACTER_ID, 0.49, 0.60, 0.05, 0.09),
        (CHARACTER_ID, 0.51, 0.60, 0.05, 0.09),
    ]
    cal = calibrate([write_frame(tmp_path, "f", rows)])

    assert cal.n_usable == 0
    assert cal.n_ambiguous == 1
    assert cal.true_scores.size == 0, "an ambiguous frame must not enter the true set"


def test_character_above_the_indicator_is_not_a_candidate(tmp_path):
    rows = [
        (INDICATOR_ID, 0.50, 0.53, 0.02, 0.02),
        (CHARACTER_ID, 0.50, 0.40, 0.05, 0.09),   # above
    ]
    cal = calibrate([write_frame(tmp_path, "f", rows)])
    assert cal.n_usable == 0
    assert cal.n_no_candidate == 1


def test_frames_with_multiple_indicators_are_skipped(tmp_path):
    rows = [
        (INDICATOR_ID, 0.30, 0.53, 0.02, 0.02),
        (INDICATOR_ID, 0.70, 0.53, 0.02, 0.02),
        (CHARACTER_ID, 0.30, 0.60, 0.05, 0.09),
    ]
    cal = calibrate([write_frame(tmp_path, "f", rows)])
    assert cal.n_multi_indicator == 1
    assert cal.n_usable == 0


# ── threshold derivation ────────────────────────────────────────────────────

def test_threshold_is_positive_and_accepts_real_pairs(tmp_path):
    """The regression that motivated this file: a zero threshold rejects everything."""
    files = [write_frame(tmp_path, f"f{i}", clean_frame(0.30 + 0.005 * i)) for i in range(50)]
    cal = calibrate(files)
    consts = derive_constants(cal)

    assert consts["INDICATOR_MAX_MATCH_SCORE"] > 0.0
    typical = float(np.median(cal.true_scores))
    assert consts["INDICATOR_MAX_MATCH_SCORE"] > typical, "threshold must accept a typical pair"


def test_threshold_survives_a_single_contaminating_outlier(tmp_path):
    """One bad frame must not move the threshold, which is what min/max allowed."""
    files = [write_frame(tmp_path, f"f{i}", clean_frame(0.30 + 0.005 * i)) for i in range(60)]
    files.append(write_frame(tmp_path, "outlier", [
        (INDICATOR_ID, 0.10, 0.53, 0.02, 0.02),
        (CHARACTER_ID, 0.45, 0.70, 0.05, 0.09),   # wildly offset
    ]))

    consts = derive_constants(calibrate(files))
    clean_only = derive_constants(calibrate(files[:-1]))

    ratio = consts["INDICATOR_MAX_MATCH_SCORE"] / clean_only["INDICATOR_MAX_MATCH_SCORE"]
    assert 0.5 < ratio < 2.0, f"one outlier moved the threshold by {ratio:.2f}x"


def test_threshold_separates_true_from_far_false_pairs(tmp_path):
    files = []
    for i in range(40):
        rows = clean_frame(0.25 + 0.005 * i) + [(CHARACTER_ID, 0.90, 0.60, 0.05, 0.09)]
        files.append(write_frame(tmp_path, f"f{i}", rows))

    cal = calibrate(files)
    consts = derive_constants(cal)
    q = threshold_quality(cal, consts["INDICATOR_MAX_MATCH_SCORE"])

    assert q["true_accept_rate"] == 1.0
    assert q["false_admit_rate"] == 0.0


def test_characters_above_the_indicator_are_not_scored_as_false_pairs(tmp_path):
    """They are rejected by ABOVE_TOLERANCE, not by the score.

    Scoring them would fabricate a near-zero 'false' pair -- max(0, dy) zeroes the
    vertical term when dy < 0 -- which is what previously dragged the threshold below
    the true median.
    """
    rows = clean_frame(0.50) + [(CHARACTER_ID, 0.50, 0.30, 0.05, 0.09)]   # directly above
    cal = calibrate([write_frame(tmp_path, "f", rows)])

    assert cal.n_usable == 1
    assert cal.n_gate_rejected == 1
    assert cal.false_scores.size == 0, "an above-indicator character is not a competitor"


def test_threshold_always_accepts_the_true_distribution(tmp_path):
    """The regression: a threshold under the true median cripples identity."""
    files = []
    for i in range(60):
        rows = clean_frame(0.30 + 0.004 * i) + [(CHARACTER_ID, 0.30 + 0.004 * i, 0.35, 0.05, 0.09)]
        files.append(write_frame(tmp_path, f"f{i}", rows))

    cal = calibrate(files)
    consts = derive_constants(cal)

    assert consts["INDICATOR_MAX_MATCH_SCORE"] > float(np.median(cal.true_scores))
    assert threshold_quality(cal, consts["INDICATOR_MAX_MATCH_SCORE"])["true_accept_rate"] >= 0.98


def test_empty_input_raises_rather_than_emitting_a_bad_constant(tmp_path):
    """Silently emitting a broken constant is worse than failing."""
    cal = calibrate([write_frame(tmp_path, "f", [(CHARACTER_ID, 0.5, 0.6, 0.05, 0.09)])])
    with pytest.raises(ValueError):
        derive_constants(cal)


def test_score_formula_matches_detection_schema():
    from feature_extractor.memory.detection_schema import (
        INDICATOR_HORIZONTAL_WEIGHT,
        INDICATOR_VERTICAL_WEIGHT,
    )
    dx, dy = 0.03, 0.05
    expected = INDICATOR_HORIZONTAL_WEIGHT * abs(dx) + INDICATOR_VERTICAL_WEIGHT * max(0.0, dy)
    assert score(dx, dy) == pytest.approx(expected)


def test_cli_margin_default_matches_the_function_default():
    """Guards the drift that produced a 2.5x threshold.

    The CLI carried a stale default from an earlier p95-based design, silently
    overriding the function's own default on every command-line run.
    """
    import inspect

    from tools.calibrate_indicator_geometry import parse_args

    fn_default = inspect.signature(derive_constants).parameters["margin"].default
    cli_default = parse_args(["--labels", "."]).margin
    assert cli_default == fn_default


def test_margin_above_the_default_admits_far_more_wrong_pairs(tmp_path):
    """Why the margin is small: the trade is heavily asymmetric."""
    files = []
    for i in range(60):
        cx = 0.30 + 0.004 * i
        rows = clean_frame(cx) + [(CHARACTER_ID, cx + 0.09, 0.62, 0.05, 0.09)]
        files.append(write_frame(tmp_path, f"f{i}", rows))
    cal = calibrate(files)

    tight = threshold_quality(cal, derive_constants(cal, margin=1.15)["INDICATOR_MAX_MATCH_SCORE"])
    loose = threshold_quality(cal, derive_constants(cal, margin=2.5)["INDICATOR_MAX_MATCH_SCORE"])

    assert tight["true_accept_rate"] >= 0.98
    assert loose["false_admit_rate"] >= tight["false_admit_rate"]


def test_character_box_extent_is_measured(tmp_path):
    """FighterState.height is a foot-offset calibration; measure it, don't inherit it."""
    files = [
        write_frame(tmp_path, f"f{i}", [
            (INDICATOR_ID, 0.40, 0.53, 0.02, 0.02),
            (CHARACTER_ID, 0.40, 0.60, 0.06, 0.14),
        ])
        for i in range(20)
    ]
    cal = calibrate(files)

    assert cal.char_h.size == 20
    assert float(np.median(cal.char_h)) == pytest.approx(0.14)
    assert float(np.median(cal.char_h)) / 2.0 == pytest.approx(0.07)
