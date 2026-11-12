"""YAML configs, command-line overrides and experiment grids (PADP's config_io, for this project).

An experiment file looks like::

    name: s1_ppo
    base: ../base.yaml          # defaults, relative to this file
    seeds: [0, 1, 2, 3, 4]
    overrides:                  # applied to every variant
      env: {difficulty: medium}
    grid:                       # variants = cartesian product of factor levels
      - name: regime
        key: env.regime         # each level sets this key...
        levels: {realtime: realtime, oracle: stepped}
      - name: start             # ...or, without `key`, each level is a dict of dotted keys
        levels:
          bc: {train.init: runs/bc/medium_v4/policy.zip}
    compare: {factor: regime, baseline: realtime, treatment: oracle}

Variant names join the level names, e.g. ``realtime-bc``. A run lives in
``runs/<experiment>/<variant>/seed<seed>/``. Settings are plain nested dictionaries; a key that
configs/base.yaml does not have is an error, so a typing mistake cannot pass as a setting.
"""
from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from brawl.paths import CONFIGS, RUNS

SECTIONS = ("env", "train")


def load_yaml(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def set_dotted(config: dict[str, Any], dotted: str, value: Any) -> None:
    *parents, leaf = dotted.split(".")
    node = config
    for key in parents:
        node = node.setdefault(key, {})
    node[leaf] = value


def parse_sets(items: Optional[list[str]]) -> dict[str, Any]:
    """``["train.learning_rate=1e-4", "env.k=2"]`` -> nested dict (values parsed as YAML)."""
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        set_dotted(out, key.strip(), yaml.safe_load(raw))
    return out


def check(config: dict[str, Any], known: dict[str, Any]) -> None:
    """Every setting must exist in the defaults."""
    for section in config:
        unknown = set(config[section]) - set(known.get(section, {})) if section in known else {section}
        if unknown:
            raise ValueError(f"unknown setting(s) in {section!r}: {sorted(unknown)} (not in configs/base.yaml)")


@dataclass
class Experiment:
    name: str
    description: str
    config: dict[str, Any]
    factors: list[tuple[str, dict[str, dict[str, Any]]]] = field(default_factory=list)  # (name, level -> {dotted key: value})
    seeds: list[int] = field(default_factory=lambda: [0])
    compare: Optional[dict[str, str]] = None
    path: Optional[Path] = None

    def variants(self) -> list[tuple[str, dict[str, Any]]]:
        if not self.factors:
            return [("default", {})]
        out = []
        for combo in itertools.product(*[list(levels.items()) for _, levels in self.factors]):
            settings: dict[str, Any] = {}
            for _, assignments in combo:
                settings.update(assignments)
            out.append(("-".join(level for level, _ in combo), settings))
        return out

    def resolve(self, variant: str, seed: int, sets: Optional[list[str]] = None) -> dict[str, Any]:
        lookup = dict(self.variants())
        if variant not in lookup:
            raise ValueError(f"unknown variant {variant!r}; choose from {sorted(lookup)}")
        config = copy.deepcopy(self.config)
        for dotted, value in lookup[variant].items():
            set_dotted(config, dotted, copy.deepcopy(value))
        config = deep_merge(config, parse_sets(sets))
        set_dotted(config, "train.seed", seed)
        check(config, resolve_config(CONFIGS / "base.yaml"))
        return config

    def run_dir(self, variant: str, seed: int, runs_root: Path = RUNS) -> Path:
        return Path(runs_root) / self.name / variant / f"seed{seed}"


def resolve_config(path: Path) -> dict[str, Any]:
    """The settings of a file, following its ``base:`` chain (base first, then overrides)."""
    path = Path(path).resolve()
    raw = load_yaml(path)
    config: dict[str, Any] = resolve_config(path.parent / raw["base"]) if "base" in raw else {}
    config = deep_merge(config, {k: raw[k] for k in SECTIONS if k in raw})
    return deep_merge(config, raw.get("overrides", {}))


def load_experiment(path: Path) -> Experiment:
    """Load an experiment file, or a plain config (one default variant, named after the file)."""
    path = Path(path)
    raw = load_yaml(path)
    factors = []
    for spec in raw.get("grid", []) or []:
        levels = spec["levels"]
        if not isinstance(levels, dict):
            levels = {str(v): v for v in levels}
        if "key" in spec:
            levels = {str(name): {spec["key"]: value} for name, value in levels.items()}
        for name in levels:
            if "-" in str(name):
                raise ValueError(f"level name {name!r} must not contain '-' (used to join variant names)")
        factors.append((spec["name"], {str(k): dict(v) for k, v in levels.items()}))
    return Experiment(name=raw.get("name", path.stem), description=raw.get("description", ""),
                      config=resolve_config(path), factors=factors, seeds=list(raw.get("seeds", [0])),
                      compare=raw.get("compare"), path=path)
