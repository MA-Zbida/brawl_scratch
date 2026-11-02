"""Repository locations.

Every default path in the package is anchored here rather than at the current working
directory, so commands behave the same from any folder.

Layout (see README.md):

- ``runs/``     training outputs, git-ignored. ``runs/legacy/<era>/`` holds work from
  before the research hypothesis; new experiments write ``runs/<experiment>/...``.
- ``results/``  committed summaries and figures (``results/legacy/<era>/`` likewise).
- ``data/``     datasets, demos, world replay and frozen input models, git-ignored.
- ``assets/``   committed inputs: YOLO weights and reference screenshots.
- ``configs/``  experiment and protocol files.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

RUNS = REPO_ROOT / "runs"
LEGACY_RUNS = RUNS / "legacy"
RESULTS = REPO_ROOT / "results"
LEGACY_RESULTS = RESULTS / "legacy"
DATA = REPO_ROOT / "data"
ASSETS = REPO_ROOT / "assets"
CONFIGS = REPO_ROOT / "configs"
LEGACY_CONFIGS = CONFIGS / "legacy"
RESEARCH = REPO_ROOT / "research"


def source_files() -> list[Path]:
    """Python sources that define a run's behavior, for provenance hashes."""
    files = [*(REPO_ROOT / "brawl").rglob("*.py"), *(REPO_ROOT / "scripts").rglob("*.py")]
    return sorted(p for p in files if "__pycache__" not in p.parts)
