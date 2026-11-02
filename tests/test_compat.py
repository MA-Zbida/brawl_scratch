"""Checkpoints saved before the restructure still load (brawl/compat.py)."""
from __future__ import annotations

import importlib
import importlib.util
import pickle

import pytest

from brawl.compat import LEGACY_MODULES, legacy_module_aliases
from brawl.training import run_utils

# What SB3/cloudpickle wrote for a function or class reference before the move.
LEGACY_PICKLE = b"ctrain.run_utils\nsha256\n."


def test_old_module_names_resolve_to_the_current_modules_inside_the_block():
    with legacy_module_aliases():
        assert importlib.import_module("train.run_utils") is run_utils
        assert pickle.loads(LEGACY_PICKLE) is run_utils.sha256


def test_old_module_names_fail_outside_the_block():
    with legacy_module_aliases():
        importlib.import_module("train.run_utils")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("train.run_utils")
    with pytest.raises(ModuleNotFoundError):
        pickle.loads(LEGACY_PICKLE)


def test_aliasing_keeps_the_real_module_spec():
    with legacy_module_aliases():
        importlib.import_module("train.run_utils")
    assert run_utils.__spec__.name == "brawl.training.run_utils"


@pytest.mark.parametrize("target", sorted(set(LEGACY_MODULES.values())))
def test_every_alias_target_exists(target):
    assert importlib.util.find_spec(target) is not None, target
