"""Load checkpoints saved before the ``brawl`` package existed.

SB3 checkpoints pickle classes and functions by module path, for example
``train.round_a_models.VisualStructured`` or ``train.primitive_training.PrimitiveGeometry``.
Those modules moved into ``brawl`` (see research/LAB_NOTEBOOK.md, 2026-11-02 restructure).

``legacy_module_aliases()`` makes the old names importable for the duration of a load, each
one resolving to the module object that replaced it, so a restored policy uses the current
classes. It is deliberately scoped: outside the ``with`` block an old import still fails,
so a missed rewrite cannot hide behind the aliases.
"""
from __future__ import annotations

import contextlib
import importlib
import importlib.abc
import importlib.util
import sys
import types
from typing import Any, Iterator

#: Old module path -> current module path.
LEGACY_MODULES: dict[str, str] = {
    'action_space':                               'brawl.control.action_space',
    'algo':                                       'brawl.algo',
    'algo.anchored_replay_ppo':                   'brawl.algo.anchored_replay_ppo',
    'analysis':                                   'brawl.analysis',
    'analysis.io':                                'brawl.analysis.io',
    'analysis.paper_evidence':                    'brawl.analysis.paper_evidence',
    'analysis.plot_final_training':               'brawl.analysis.plot_final_training',
    'analysis.plot_learning_curves':              'brawl.analysis.plot_learning_curves',
    'analysis.plot_retention_matrix':             'brawl.analysis.plot_retention_matrix',
    'analysis.plot_round_a':                      'brawl.analysis.plot_round_a',
    'analysis.style':                             'brawl.analysis.style',
    'capture':                                    'brawl.capture',
    'capture.dxgi':                               'brawl.capture.dxgi',
    'capture.gdi':                                'brawl.capture.gdi',
    'capture_first':                              'brawl.capture_first',
    'config':                                     'brawl.calibration',
    'control':                                    'brawl.control',
    'control.input':                              'brawl.control.input',
    'control.semantic_actions':                   'brawl.control.semantic_actions',
    'control.semantic_executor':                  'brawl.control.semantic_executor',
    'control.taps':                               'brawl.control.taps',
    'env':                                        'brawl.env.live',
    'evaluate':                                   'brawl.pipelines.llc.evaluate',
    'feature_extractor.film_extractor':           'brawl.pipelines.llc.film_extractor',
    'feature_extractor.final_extractor':          'brawl.pipelines.final_agent.extractor',
    'feature_extractor.memory':                   'brawl.state',
    'feature_extractor.memory.canonicalize':      'brawl.state.canonicalize',
    'feature_extractor.memory.detection_schema':  'brawl.state.detection_schema',
    'feature_extractor.memory.state_spec':        'brawl.state.state_spec',
    'feature_extractor.memory.structured_memory': 'brawl.state.structured_memory',
    'feature_extractor.memory.utils':             'brawl.state.utils',
    'feature_extractor.yolo':                     'brawl.perception.yolo',
    'feature_extractor.yolo.extract':             'brawl.perception.yolo',
    'hierarchical.goal_conditioning':             'brawl.pipelines.llc.goal_conditioning',
    'hierarchical.goals':                         'brawl.pipelines.llc.goals',
    'hierarchical.llc_env':                       'brawl.pipelines.llc.llc_env',
    'perception.async_pipeline':                  'brawl.perception.async_pipeline',
    'perception.weapon_classifier':               'brawl.perception.weapon_classifier',
    'perception.weapon_evidence':                 'brawl.perception.weapon_evidence',
    'reward':                                     'brawl.reward',
    'reward.extract_rgb':                         'brawl.reward.extract_rgb',
    'reward.match_objective':                     'brawl.reward.match_objective',
    'reward.providers':                           'brawl.reward.providers',
    'reward.rgb_to_dmg':                          'brawl.reward.rgb_to_dmg',
    'reward.stock':                               'brawl.reward.stock',
    'reward.ui_probe':                            'brawl.reward.ui_probe',
    'train.checkpoint_roles':                     'brawl.training.checkpoints',
    'train.collect_bc_locomotion_demos':          'brawl.pipelines.llc.collect_demos',
    'train.collect_heuristic_curriculum_demos':   'brawl.pipelines.llc.collect_heuristic_demos',
    'train.compare_movement_checkpoints':         'brawl.pipelines.llc.compare_checkpoints',
    'train.curriculum_config':                    'brawl.pipelines.llc.curriculum_config',
    'train.curriculum_goals':                     'brawl.training.curriculum_goals',
    'train.diagnostic_timing':                    'brawl.pipelines.round_a.diagnostic_timing',
    'train.evaluate_retention':                   'brawl.pipelines.llc.evaluate_retention',
    'train.experiment_config':                    'brawl.training.experiment_config',
    'train.final_agent':                          'brawl.pipelines.final_agent.run',
    'train.final_campaign':                       'brawl.pipelines.final_agent.campaign',
    'train.final_config':                         'brawl.pipelines.final_agent.config',
    'train.final_policy':                         'brawl.pipelines.final_agent.policy',
    'train.heuristic_teachers':                   'brawl.training.teachers',
    'train.llc_stage_common':                     'brawl.pipelines.llc.stage_common',
    'train.match_env':                            'brawl.env.match',
    'train.phase_action_policy':                  'brawl.pipelines.llc.phase_action_policy',
    'train.phase_registry':                       'brawl.pipelines.llc.phase_registry',
    'train.pipeline_cli':                         'brawl.training.cli',
    'train.pipeline_runner':                      'brawl.training.runner',
    'train.policy_transfer':                      'brawl.pipelines.final_agent.policy_transfer',
    'train.pretrain_bc_locomotion':               'brawl.pipelines.llc.pretrain_bc',
    'train.primitive_env':                        'brawl.pipelines.primitive27.env',
    'train.primitive_features':                   'brawl.pipelines.primitive27.features',
    'train.primitive_pipeline':                   'brawl.pipelines.primitive27.run',
    'train.primitive_retention':                  'brawl.training.primitive_retention',
    'train.primitive_training':                   'brawl.pipelines.primitive27.training',
    'train.retention':                            'brawl.pipelines.llc.retention',
    'train.round_a':                              'brawl.pipelines.round_a.run',
    'train.round_a_bc':                           'brawl.pipelines.round_a.bc',
    'train.round_a_collection':                   'brawl.pipelines.round_a.collection',
    'train.round_a_config':                       'brawl.pipelines.round_a.config',
    'train.round_a_data':                         'brawl.pipelines.round_a.data',
    'train.round_a_models':                       'brawl.pipelines.round_a.models',
    'train.round_a_runtime':                      'brawl.pipelines.round_a.runtime',
    'train.round_a_timing':                       'brawl.pipelines.round_a.timing',
    'train.round_a_video':                        'brawl.pipelines.round_a.video',
    'train.run_match':                            'brawl.pipelines.final_agent.run_match',
    'train.run_utils':                            'brawl.training.run_utils',
    'train.semantic_bc':                          'brawl.pipelines.semantic57.bc',
    'train.semantic_collect':                     'brawl.pipelines.semantic57.collect',
    'train.semantic_contract':                    'brawl.pipelines.semantic57.contract',
    'train.semantic_demos':                       'brawl.pipelines.semantic57.demos',
    'train.semantic_env':                         'brawl.pipelines.semantic57.env',
    'train.semantic_pipeline':                    'brawl.pipelines.semantic57.run',
    'train.semantic_policy':                      'brawl.pipelines.semantic57.policy',
    'train.semantic_teacher':                     'brawl.pipelines.semantic57.teacher',
    'train.train_config':                         'brawl.pipelines.llc.train_config',
    'train.train_curriculum':                     'brawl.pipelines.llc.train_curriculum',
    'train.train_weapon_classifier':              'brawl.pipelines.weapon_classifier.train',
    'train.transfer_skill_checkpoint':            'brawl.pipelines.final_agent.transfer_skill_checkpoint',
    'world_model':                                'brawl.world_model',
    'world_model.replay_log':                     'brawl.world_model.replay_log',
}

# Old packages with no single successor (their modules moved to different places).
_SYNTHETIC_PACKAGES = sorted({prefix for name in LEGACY_MODULES
                              for prefix in (".".join(name.split(".")[:i]) for i in range(1, name.count(".") + 1))
                              if prefix not in LEGACY_MODULES})


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target: str | None):
        self.target = target
        self._spec = None

    def create_module(self, spec):
        if self.target is None:
            module = types.ModuleType(spec.name)
            module.__path__ = []
            return module
        module = importlib.import_module(self.target)
        self._spec = module.__spec__
        return module

    def exec_module(self, module) -> None:
        if self.target is not None:
            # The import system stamps the alias spec on the module; keep the real one.
            module.__spec__ = self._spec


class _LegacyFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in LEGACY_MODULES:
            return importlib.util.spec_from_loader(fullname, _AliasLoader(LEGACY_MODULES[fullname]))
        if fullname in _SYNTHETIC_PACKAGES:
            return importlib.util.spec_from_loader(fullname, _AliasLoader(None), is_package=True)
        return None


@contextlib.contextmanager
def legacy_module_aliases() -> Iterator[None]:
    """Resolve pre-restructure module names while the block runs."""
    finder = _LegacyFinder()
    before = set(sys.modules)
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
        for name in set(sys.modules) - before:
            if name in LEGACY_MODULES or name in _SYNTHETIC_PACKAGES:
                del sys.modules[name]


def load_sb3(cls: Any, *args: Any, **kwargs: Any) -> Any:
    """``cls.load(*args, **kwargs)`` that also accepts checkpoints saved before the restructure."""
    with legacy_module_aliases():
        return cls.load(*args, **kwargs)
