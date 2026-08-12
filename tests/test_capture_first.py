"""Screen-duplicator acquisition ordering.

Measured on the target hardware: DXGI duplication succeeds in a bare process and
fails after `import torch` — loading the NVIDIA CUDA libraries moves the process to
the discrete GPU, which does not own the display output.

That makes import order load-bearing in a way nothing in the source shows. These
tests keep the ordering enforced, because the failure mode is a bare COMError about
"feature levels" that reads like a driver problem and costs hours to trace back to an
import statement.
"""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent

#: Anything that transitively pulls in torch.
_HEAVY_IMPORT = re.compile(
    r"^\s*(import torch|from torch|import ultralytics|from ultralytics"
    r"|import stable_baselines3|from stable_baselines3"
    r"|from env import|from feature_extractor|from train\.|from algo)",
    re.M,
)

#: Entry points that construct a live environment.
LIVE_ENTRY_POINTS = [
    "tools/debug_observation_overlay.py",
    "train/collect_bc_locomotion_demos.py",
    "train/pretrain_bc_locomotion.py",
    "train/train_curriculum.py",
    "train/evaluate_retention.py",
    "evaluate.py",
]


@pytest.mark.parametrize("rel", LIVE_ENTRY_POINTS)
def test_entry_point_actually_launches(rel):
    """Run the script. Do not merely parse it.

    Two bugs shipped past parse-only checks: `import capture_first` placed above
    `from __future__` (a SyntaxError), and then the same import failing with
    ModuleNotFoundError because launching `python tools/x.py` puts *tools/* on
    sys.path rather than the repo root.

    Both are invisible to `ast.parse`, which is why this spawns the interpreter the
    way a user would. Missing third-party packages are tolerated — this is checking
    the bootstrap, not the dependency set — but anything wrong with capture_first,
    the path setup, or syntax is a failure.
    """
    import subprocess

    proc = subprocess.run(
        [sys.executable, rel, "--help"],
        cwd=REPO, capture_output=True, text=True, timeout=300,
    )
    combined = proc.stdout + proc.stderr

    assert "No module named 'capture_first'" not in combined, (
        f"{rel}: capture_first is not importable when the script is launched directly. "
        "The sys.path bootstrap must precede it."
    )
    assert "SyntaxError" not in combined, f"{rel} does not parse:\n{combined[-800:]}"
    for marker in ("IndentationError", "NameError: name '_sys'", "NameError: name '_Path'"):
        assert marker not in combined, f"{rel}: {marker}\n{combined[-800:]}"


@pytest.mark.parametrize("rel", LIVE_ENTRY_POINTS)
def test_entry_point_is_valid_python(rel):
    import ast

    source = (REPO / rel).read_text(encoding="utf-8")
    ast.parse(source, filename=rel)


@pytest.mark.parametrize("rel", LIVE_ENTRY_POINTS)
def test_future_import_stays_first(rel):
    """`from __future__` must precede every other statement, including capture_first."""
    import ast

    source = (REPO / rel).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=rel)

    future = [
        n for n in tree.body
        if isinstance(n, ast.ImportFrom) and n.module == "__future__"
    ]
    if not future:
        return

    preceding = [
        n for n in tree.body
        if n.lineno < future[0].lineno
        and not (isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant))
    ]
    assert not preceding, (
        f"{rel}: statements before `from __future__` "
        f"(lines {[n.lineno for n in preceding]}) — this is a SyntaxError"
    )


@pytest.mark.parametrize("rel", LIVE_ENTRY_POINTS)
def test_capture_first_precedes_every_torch_pulling_import(rel):
    source = (REPO / rel).read_text(encoding="utf-8")

    idx = source.find("import capture_first")
    assert idx >= 0, f"{rel} builds a live env but never imports capture_first"

    heavy = [m.start() for m in _HEAVY_IMPORT.finditer(source)]
    if heavy:
        assert idx < min(heavy), (
            f"{rel}: capture_first must be imported before anything that loads torch; "
            "otherwise the duplicator can no longer be acquired"
        )


def test_env_imports_capture_first_before_the_detector():
    """env.py pulls in Extract -> ultralytics -> torch at module scope."""
    source = (REPO / "env.py").read_text(encoding="utf-8")
    assert source.index("import capture_first") < source.index(
        "from feature_extractor.yolo.extract import Extract"
    )


def _fresh_module(dxcam_stub, torch_loaded: bool):
    """Reload capture_first against a stub adapter set.

    `torch` in sys.modules is never touched. Removing and re-importing the real torch
    re-runs its C++ operator registration, which raises and poisons every later test
    in the process -- the flag is set directly instead, which is all `status()` reads.
    """
    sys.modules.pop("capture_first", None)
    sys.modules["dxcam"] = dxcam_stub
    import capture_first

    capture_first.acquire(force=True)
    capture_first._torch_already_loaded = torch_loaded
    return capture_first


@pytest.fixture(autouse=True)
def _restore_modules():
    """Leave sys.modules exactly as found, so ordering between tests cannot matter."""
    saved_dxcam = sys.modules.get("dxcam")
    saved_capture = sys.modules.get("capture_first")
    yield
    for name, saved in (("dxcam", saved_dxcam), ("capture_first", saved_capture)):
        if saved is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = saved


def _stub(working: tuple[int, int] | None):
    class Cam:
        pass

    mod = types.ModuleType("dxcam")
    mod.device_info = lambda: "Device[0]:<Intel>\nDevice[1]:<NVIDIA>"
    mod.output_info = lambda: "Device[0] Output[0]: Res:(1920,1080)"

    def create(device_idx=0, output_idx=0, output_color="BGR"):
        if working is not None and (device_idx, output_idx) == working:
            return Cam()
        raise OSError("The specified device interface or feature level is not supported")

    mod.create = create
    return mod


def test_camera_ownership_transfers_once():
    """The env adopts the pre-acquired camera; a second caller must not get it too."""
    cf = _fresh_module(_stub((0, 0)), torch_loaded=False)
    assert cf.is_available()

    first = cf.take_camera()
    assert first is not None
    assert cf.take_camera() is None
    assert not cf.is_available()


def test_probes_across_adapters():
    cf = _fresh_module(_stub((1, 0)), torch_loaded=False)
    assert cf.is_available()
    assert "device 1, output 0" in cf.status()


def test_failure_after_torch_names_the_import_as_the_cause():
    """The diagnosis has to appear in the message, not just in a doc somewhere."""
    cf = _fresh_module(_stub(None), torch_loaded=True)

    assert not cf.is_available()
    status = cf.status()
    assert "torch was already imported" in status
    assert "capture_first" in status


def test_failure_without_torch_does_not_blame_torch():
    cf = _fresh_module(_stub(None), torch_loaded=False)
    assert not cf.is_available()
    assert "torch was already imported" not in cf.status()


def test_module_is_inert_under_pytest():
    """Test runs must not grab a real duplicator from the desktop."""
    source = (REPO / "capture_first.py").read_text(encoding="utf-8")
    assert '"pytest" not in sys.modules' in source
