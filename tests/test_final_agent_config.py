from __future__ import annotations

import pytest

from train.final_config import parse_args


def test_final_agent_config_is_goal_free_cpu_match_training() -> None:
    cfg = parse_args(["--init-from", "skill.zip", "--timesteps", "0"])

    assert cfg.init_from == "skill.zip"
    assert cfg.device == "cpu"
    assert cfg.goal_conditioning is False
    assert cfg.objective == "match"
    assert cfg.timesteps == 0


@pytest.mark.parametrize(
    "obsolete",
    ["--macro-steps", "--goal-period", "--hsp-checkpoint", "--high-level-policy"],
)
def test_final_agent_rejects_obsolete_hsp_arguments(obsolete: str) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--init-from", "skill.zip", obsolete, "12"])


def test_final_agent_requires_exactly_one_initialization_path() -> None:
    with pytest.raises(SystemExit):
        parse_args([])
    with pytest.raises(SystemExit):
        parse_args(["--init-from", "skill.zip", "--resume", "final.zip"])
