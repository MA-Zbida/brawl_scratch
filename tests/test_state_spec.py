"""Smoke tests for StateSpec, Memory.to_vector(), and observation consistency.

Run with:
    python tests/test_state_spec.py
    # or
    python -m pytest tests/test_state_spec.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def test_state_spec_dim():
    """StateSpec.dim() matches the expected 51."""
    from feature_extractor.memory.state_spec import StateSpec
    assert StateSpec.dim() == 51, f"Expected 51, got {StateSpec.dim()}"
    print("[PASS] StateSpec.dim() == 51")


def test_state_spec_index_consistency():
    """Every feature name has a unique index in [0, dim)."""
    from feature_extractor.memory.state_spec import StateSpec
    seen_indices = set()
    for name in StateSpec.FEATURES:
        idx = StateSpec.index(name)
        assert 0 <= idx < StateSpec.dim(), f"{name} index {idx} out of range"
        assert idx not in seen_indices, f"Duplicate index {idx} for {name}"
        seen_indices.add(idx)
    assert len(seen_indices) == StateSpec.dim()
    print(f"[PASS] All {StateSpec.dim()} feature indices are unique and valid")


def test_memory_to_vector_shape():
    """Memory.to_vector() produces a vector matching StateSpec.dim()."""
    from feature_extractor.memory.structured_memory import Memory
    from feature_extractor.memory.state_spec import StateSpec
    m = Memory()
    vec = m.to_vector()
    assert vec.shape == (StateSpec.dim(),), f"Expected ({StateSpec.dim()},), got {vec.shape}"
    assert vec.dtype == np.float32, f"Expected float32, got {vec.dtype}"
    print(f"[PASS] Memory.to_vector() shape = ({StateSpec.dim()},), dtype = float32")


def test_state_spec_get():
    """StateSpec.get() returns correct values from a known state."""
    from feature_extractor.memory.structured_memory import Memory
    from feature_extractor.memory.state_spec import StateSpec
    m = Memory()
    m.player.x = 0.42
    m.player.y = 0.58
    m.player.grounded = True
    m.player.weapon_state = 1.0
    m.opponent.x = 0.65
    m.opponent.y = 0.70
    m.opponent.exists = True

    vec = m.to_vector()

    assert abs(StateSpec.get(vec, "player_x") - 0.42) < 1e-5
    assert abs(StateSpec.get(vec, "player_y") - 0.58) < 1e-5
    assert StateSpec.get(vec, "player_grounded") == 1.0
    assert StateSpec.get(vec, "player_has_weapon") == 1.0
    assert abs(StateSpec.get(vec, "opponent_x") - 0.65) < 1e-5
    assert abs(StateSpec.get(vec, "opponent_y") - 0.70) < 1e-5
    assert StateSpec.get(vec, "opponent_exists") == 1.0
    print("[PASS] StateSpec.get() returns correct values for known state")


def test_memory_buffer_reuse():
    """to_vector() reuses the pre-allocated buffer (same memory address)."""
    from feature_extractor.memory.structured_memory import Memory
    m = Memory()
    v1 = m.to_vector()
    v1_data = v1.ctypes.data
    m.player.x = 0.99
    v2 = m.to_vector()
    v2_data = v2.ctypes.data
    assert v1_data == v2_data, "Buffer not reused — to_vector() is allocating new arrays"
    assert abs(v2[0] - 0.99) < 1e-5, "Buffer not updated correctly"
    print("[PASS] Memory.to_vector() reuses pre-allocated buffer")


def test_state_spec_validate_vector():
    """StateSpec.validate_vector() accepts correct sizes and rejects wrong ones."""
    from feature_extractor.memory.state_spec import StateSpec
    good = np.zeros(StateSpec.dim(), dtype=np.float32)
    StateSpec.validate_vector(good)  # should not raise

    bad = np.zeros(40, dtype=np.float32)
    try:
        StateSpec.validate_vector(bad)
        assert False, "Should have raised ValueError for wrong dim"
    except ValueError:
        pass
    print("[PASS] StateSpec.validate_vector() correctly validates dimensions")


def test_get_multi():
    """StateSpec.get_multi() returns multiple features as array."""
    from feature_extractor.memory.structured_memory import Memory
    from feature_extractor.memory.state_spec import StateSpec
    m = Memory()
    m.player.x = 0.3
    m.player.y = 0.7
    vec = m.to_vector()
    result = StateSpec.get_multi(vec, "player_x", "player_y")
    assert result.shape == (2,)
    assert abs(result[0] - 0.3) < 1e-5
    assert abs(result[1] - 0.7) < 1e-5
    print("[PASS] StateSpec.get_multi() returns correct multi-feature array")


def test_goal_spec():
    """Continuous goal specification is available and consistent."""
    from hierarchical.goals import GOAL_DIM, GOAL_HIGH, GOAL_LOW, GOAL_NAMES
    assert GOAL_DIM == 6
    assert len(GOAL_NAMES) == GOAL_DIM
    assert GOAL_LOW.shape == (GOAL_DIM,)
    assert GOAL_HIGH.shape == (GOAL_DIM,)
    assert np.all(GOAL_LOW < GOAL_HIGH)
    print("[PASS] Goal spec (6D continuous) is valid")


def test_hitstun_tracking():
    """Memory hitstun and airborne tracking works correctly."""
    from feature_extractor.memory.structured_memory import Memory
    from feature_extractor.memory.state_spec import StateSpec
    m = Memory()

    # Simulate player not grounded for several frames
    m.player.x = 0.50
    m.player.y = 0.30
    m.player.grounded = False
    m.player.on_edge = False
    for _ in range(30):
        m.update_on_ground()

    vec = m.to_vector()
    airborne = StateSpec.get(vec, "player_airborne_time")
    assert airborne > 0.0, f"Expected positive airborne time, got {airborne}"
    assert airborne <= 1.0, f"Airborne time should be normalised, got {airborne}"

    # Simulate getting hit
    m.just_got_hit = 1.0
    m.player.damage_percent = 0.5
    m.update_hitstun(0.0)  # trigger hitstun
    vec = m.to_vector()
    hitstun = StateSpec.get(vec, "player_hitstun")
    assert hitstun > 0.0, f"Expected positive hitstun, got {hitstun}"
    print("[PASS] Hitstun and airborne tracking work correctly")


# ── Run all tests ────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_state_spec_dim,
        test_state_spec_index_consistency,
        test_memory_to_vector_shape,
        test_state_spec_get,
        test_memory_buffer_reuse,
        test_state_spec_validate_vector,
        test_get_multi,
        test_goal_spec,
        test_hitstun_tracking,
    ]
    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test_fn.__name__}: {e}")
            failed += 1
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed > 0:
        sys.exit(1)
