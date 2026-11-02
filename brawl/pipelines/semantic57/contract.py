"""Serialization contract shared by new demos, policies, and live execution."""
from __future__ import annotations

from dataclasses import asdict

from brawl.control.semantic_actions import ACTIONS, contract, require_contract
from brawl.control.semantic_executor import SemanticExecutor, ExecutionTiming
from brawl.state.state_spec import StateSpec

ACTION_DIM = len(ACTIONS)
HISTORY_OFFSETS = (2, 4, 8)
BASE_DIM = StateSpec.observation_dim(HISTORY_OFFSETS)
EXECUTOR_DIM = ACTION_DIM + 4 + 3
OBSERVATION_DIM = BASE_DIM + EXECUTOR_DIM + ACTION_DIM
LAYOUT_VERSION = "state_history_executor_mask_v1"


def metadata() -> dict:
    return {
        "schema_version": 1, "policy_type": "final_agent", "goal_conditioning": False,
        "observation_dim": OBSERVATION_DIM, "action_dim": ACTION_DIM,
        "semantic_contract": contract(), "observation_layout": LAYOUT_VERSION,
        "base_dim": BASE_DIM, "history_offsets": list(HISTORY_OFFSETS),
        "executor_dim": EXECUTOR_DIM, "mask_in_observation": True,
        "executor_version": SemanticExecutor.VERSION, "executor_timing": asdict(ExecutionTiming()),
        "algorithm": "ppo_masked_observation_v1",
    }


def validate_metadata(payload: dict) -> None:
    require_contract(payload.get("semantic_contract", {}))
    for key, expected in metadata().items():
        if payload.get(key) != expected:
            raise ValueError(f"Incompatible semantic metadata: {key}")
