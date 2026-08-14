from __future__ import annotations

import json

import pytest

from train.checkpoint_roles import (
    FINAL_AGENT,
    GOAL_CONDITIONED_SKILL,
    CheckpointRoleError,
    make_checkpoint_metadata,
    metadata_path,
    read_checkpoint_metadata,
    require_checkpoint_role,
    write_checkpoint_metadata,
)


def test_skill_and_final_checkpoints_have_distinct_explicit_roles(tmp_path) -> None:
    skill = tmp_path / "skill.zip"
    final = tmp_path / "final.zip"
    skill.write_bytes(b"skill")
    final.write_bytes(b"final")

    write_checkpoint_metadata(
        skill,
        make_checkpoint_metadata(
            policy_type=GOAL_CONDITIONED_SKILL,
            observation_dim=168,
            action_dim=27,
            goal_dim=11,
        ),
    )
    write_checkpoint_metadata(
        final,
        make_checkpoint_metadata(
            policy_type=FINAL_AGENT,
            observation_dim=146,
            action_dim=27,
            initialized_from=str(skill),
        ),
    )

    assert read_checkpoint_metadata(skill)["policy_type"] == GOAL_CONDITIONED_SKILL
    assert read_checkpoint_metadata(final)["policy_type"] == FINAL_AGENT
    assert read_checkpoint_metadata(final)["goal_conditioning"] is False
    assert metadata_path(final).name == "final.zip.metadata.json"


def test_checkpoint_role_mismatch_fails_loudly(tmp_path) -> None:
    checkpoint = tmp_path / "skill.zip"
    checkpoint.write_bytes(b"skill")
    write_checkpoint_metadata(
        checkpoint,
        make_checkpoint_metadata(
            policy_type=GOAL_CONDITIONED_SKILL,
            observation_dim=168,
            action_dim=27,
            goal_dim=11,
        ),
    )

    with pytest.raises(CheckpointRoleError, match="goal_conditioned_skill.*final_agent"):
        require_checkpoint_role(checkpoint, FINAL_AGENT)


def test_missing_or_obsolete_metadata_fails_closed(tmp_path) -> None:
    checkpoint = tmp_path / "unknown.zip"
    checkpoint.write_bytes(b"unknown")
    with pytest.raises(CheckpointRoleError, match="metadata"):
        require_checkpoint_role(checkpoint, FINAL_AGENT)

    metadata_path(checkpoint).write_text(
        json.dumps({"schema_version": 1, "policy_type": "hsp"}),
        encoding="utf-8",
    )
    with pytest.raises(CheckpointRoleError, match="obsolete"):
        read_checkpoint_metadata(checkpoint)
