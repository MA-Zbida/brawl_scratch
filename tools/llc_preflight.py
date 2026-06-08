#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass
class Check:
    name: str
    status: str
    detail: str
    fix: str = ""


REQUIRED_MODULES: tuple[tuple[str, str, str], ...] = (
    ("numpy", "numpy", "arrays, demos, metrics"),
    ("gymnasium", "gymnasium", "environment API"),
    ("stable_baselines3", "stable-baselines3", "PPO training"),
    ("torch", "torch", "BC/PPO neural nets"),
    ("ultralytics", "ultralytics", "YOLO perception"),
    ("cv2", "opencv-python", "overlay/video/perception preprocessing"),
    ("keyboard", "keyboard", "manual demo controls"),
    ("dxcam", "dxcam", "Brawlhalla frame capture"),
    ("pydirectinput", "pydirectinput", "Brawlhalla keyboard control"),
)

OPTIONAL_MODULES: tuple[tuple[str, str, str], ...] = (
    ("matplotlib", "matplotlib", "diagnostic plots"),
    ("pytest", "pytest", "local test suite"),
)

REQUIRED_FILES: tuple[str, ...] = (
    "requirements-llc.txt",
    "env.py",
    "config.py",
    "feature_extractor/yolo/best.pt",
    "feature_extractor/yolo/best.onnx",
    "feature_extractor/yolo/best.engine",
    "train/curriculum_config.py",
    "train/train_curriculum.py",
    "train/collect_bc_locomotion_demos.py",
)

INSTALL_REQUIREMENTS = "python -m pip install -r requirements-llc.txt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preflight checks before live Brawlhalla LLC training")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "auto"])
    p.add_argument("--models-dir", type=str, default="train/models")
    p.add_argument("--outputs-dir", type=str, default="outputs")
    p.add_argument("--strict-warnings", action="store_true", help="Treat warnings as failures")
    p.add_argument("--skip-cuda-import", action="store_true", help="Do not import torch to query CUDA")
    return p.parse_args()


def _module_version(module: str, distribution: str) -> str:
    candidates = [distribution, module]
    if module == "cv2":
        candidates = ["opencv-python", "opencv-contrib-python", "cv2"]
    for name in candidates:
        try:
            return importlib.metadata.version(name)
        except Exception:
            continue
    return "installed"


def check_module(module: str, distribution: str, purpose: str, *, required: bool) -> Check:
    spec = importlib.util.find_spec(module)
    if spec is None:
        status = "FAIL" if required else "WARN"
        return Check(
            name=f"module:{module}",
            status=status,
            detail=f"missing ({purpose})",
            fix=f"Run: {INSTALL_REQUIREMENTS} (missing package: {distribution})",
        )
    version = _module_version(module, distribution)
    return Check(
        name=f"module:{module}",
        status="PASS",
        detail=f"{version} ({purpose})",
    )


def check_python() -> Check:
    version = ".".join(str(part) for part in sys.version_info[:3])
    if sys.version_info < (3, 10):
        return Check("python", "FAIL", f"Python {version}", "Use Python 3.10+ for the RL stack.")
    return Check("python", "PASS", f"Python {version}")


def check_platform() -> Check:
    system = platform.system()
    if system != "Windows":
        return Check(
            "platform",
            "WARN",
            f"{system}; live Brawlhalla capture/control is Windows-oriented",
            "Run live training on Windows with Brawlhalla focused.",
        )
    return Check("platform", "PASS", f"{system} {platform.release()}")


def check_cuda(args: argparse.Namespace) -> Check:
    device = str(args.device).strip().lower()
    if device == "cpu":
        return Check("cuda", "PASS", "CPU mode selected")
    if bool(args.skip_cuda_import):
        return Check("cuda", "WARN", "CUDA query skipped", "Run without --skip-cuda-import before long training.")
    if importlib.util.find_spec("torch") is None:
        status = "FAIL" if device == "cuda" else "WARN"
        return Check("cuda", status, "torch missing; cannot query CUDA", f"Run: {INSTALL_REQUIREMENTS}, or use --device cpu.")
    try:
        import torch  # type: ignore

        available = bool(torch.cuda.is_available())
        if available:
            name = str(torch.cuda.get_device_name(0)) if torch.cuda.device_count() > 0 else "CUDA device"
            return Check("cuda", "PASS", f"available: {name}")
        status = "FAIL" if device == "cuda" else "WARN"
        return Check("cuda", status, "torch installed but CUDA is not available", "Use --device cpu or install a CUDA-enabled torch.")
    except Exception as exc:
        status = "FAIL" if device == "cuda" else "WARN"
        return Check("cuda", status, f"could not query CUDA: {exc}", "Verify torch/CUDA install.")


def check_required_files() -> list[Check]:
    checks: list[Check] = []
    for raw in REQUIRED_FILES:
        path = Path(raw)
        if path.exists():
            size = path.stat().st_size if path.is_file() else 0
            checks.append(Check(f"file:{raw}", "PASS", f"found ({size} bytes)" if size else "found"))
        else:
            checks.append(Check(f"file:{raw}", "FAIL", "missing", "Pull main branch or restore the project asset."))
    return checks


def check_dir(path: str, purpose: str) -> Check:
    p = Path(path)
    if p.exists() and p.is_dir():
        return Check(f"dir:{path}", "PASS", f"found ({purpose})")
    if p.exists() and not p.is_dir():
        return Check(f"dir:{path}", "FAIL", "path exists but is not a directory", "Rename/remove the file and create the directory.")
    return Check(f"dir:{path}", "WARN", f"missing ({purpose})", f"Create directory before training: {path}")


def check_ui_config() -> list[Check]:
    try:
        from config import PLATFORM_BOUNDS, UI_REGIONS
    except Exception as exc:
        return [Check("config:import", "FAIL", f"could not import config.py: {exc}", "Fix config.py before live perception.")]

    checks: list[Check] = []
    required_regions = ("stock", "op", "agent")
    missing = [key for key in required_regions if key not in UI_REGIONS]
    if missing:
        checks.append(Check("config:UI_REGIONS", "FAIL", "missing: " + ", ".join(missing), "Calibrate UI_REGIONS in config.py."))
    else:
        checks.append(Check("config:UI_REGIONS", "PASS", str(UI_REGIONS)))

    try:
        x_min = float(PLATFORM_BOUNDS["x_min"])
        x_max = float(PLATFORM_BOUNDS["x_max"])
        y_min = float(PLATFORM_BOUNDS["y_min"])
        y_max = float(PLATFORM_BOUNDS["y_max"])
        valid = 0.0 <= x_min < x_max <= 1.0 and 0.0 <= y_min < y_max <= 1.0
    except Exception:
        valid = False
    if valid:
        checks.append(Check("config:PLATFORM_BOUNDS", "PASS", str(PLATFORM_BOUNDS)))
    else:
        checks.append(Check("config:PLATFORM_BOUNDS", "FAIL", str(PLATFORM_BOUNDS), "Use normalized bounds with min < max in [0,1]."))
    return checks


def build_checks(args: argparse.Namespace) -> list[Check]:
    checks: list[Check] = [check_python(), check_platform()]
    checks.extend(check_module(*item, required=True) for item in REQUIRED_MODULES)
    checks.extend(check_module(*item, required=False) for item in OPTIONAL_MODULES)
    checks.append(check_cuda(args))
    checks.extend(check_required_files())
    checks.append(check_dir(str(args.models_dir), "models, demos, eval CSVs"))
    checks.append(check_dir(str(args.outputs_dir), "run protocol and reports"))
    checks.extend(check_ui_config())
    return checks


def summarize(checks: list[Check], *, strict_warnings: bool = False) -> tuple[str, int]:
    failures = [check for check in checks if check.status == "FAIL"]
    warnings = [check for check in checks if check.status == "WARN"]
    status = "PASS"
    exit_code = 0
    if failures or (strict_warnings and warnings):
        status = "FAIL"
        exit_code = 2
    elif warnings:
        status = "WARN"
        exit_code = 1

    width = max(len(check.name) for check in checks) if checks else 10
    lines: list[str] = []
    lines.append(f"LLC PREFLIGHT: {status}")
    lines.append(f"{'check'.ljust(width)}  status  detail")
    lines.append("-" * (width + 50))
    for check in checks:
        lines.append(f"{check.name.ljust(width)}  {check.status:6s}  {check.detail}")
        if check.status != "PASS" and check.fix:
            lines.append(f"{' '.ljust(width)}          fix: {check.fix}")

    if status == "PASS":
        lines.append("")
        lines.append("Next: run the perception overlay, then collect demos.")
    elif failures:
        lines.append("")
        lines.append("Stop: fix FAIL rows before live training.")
    else:
        lines.append("")
        lines.append("Check WARN rows before long training; use --strict-warnings to fail on them.")
    return "\n".join(lines), exit_code


def main() -> None:
    args = parse_args()
    text, exit_code = summarize(build_checks(args), strict_warnings=bool(args.strict_warnings))
    print(text)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
