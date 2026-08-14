"""Legacy-compatible goal-conditioning utilities for skill pretraining.

There is no runtime hierarchy. Current training uses ``StageGoalEnv`` and
``StageGoalFiLMExtractor``; these exports remain only for older goal-conditioned
checkpoints and research utilities.
"""

from hierarchical.goals import GOAL_DIM, GOAL_NAMES, GOAL_LOW, GOAL_HIGH, GoalSampler
from hierarchical.goal_conditioning import GoalConditionedModulationExtractor
from hierarchical.llc_env import LLCEnv

__all__ = [
    "GOAL_DIM",
    "GOAL_NAMES",
    "GOAL_LOW",
    "GOAL_HIGH",
    "GoalSampler",
    "GoalConditionedModulationExtractor",
    "LLCEnv",
]
