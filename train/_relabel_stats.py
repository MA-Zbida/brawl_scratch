from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.pretrain_bc_locomotion import _build_goal_relabel_dataset, _load_dataset, _parse_horizons
from train.curriculum_config import build_phase_spec

obs, act, dones = _load_dataset("train/models/locomotion_demos.npz", "locomotion")
spec = build_phase_spec("locomotion", death_penalty=1.0, terminate_on_death=True)
horizons = _parse_horizons("1,2,4,8")
robs, ract, eps, rel = _build_goal_relabel_dataset(
    obs=obs,
    actions=act,
    dones=dones,
    spec=spec,
    horizons=horizons,
    max_relabels_per_step=0,
    include_original_samples=True,
)
print("raw", obs.shape[0], "episodes", eps, "relabeled_only", rel, "final", robs.shape[0], "multiplier", round(robs.shape[0] / obs.shape[0], 2))
