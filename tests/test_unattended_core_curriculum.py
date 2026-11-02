from __future__ import annotations

from pathlib import Path

from scripts.llc.run_unattended_core_curriculum import build_run_plan


def _option(argv: tuple[str, ...], name: str) -> str:
    index = argv.index(name)
    return argv[index + 1]


def test_three_hour_plan_is_one_sequential_policy_chain_with_no_recovery() -> None:
    plan = build_run_plan(
        python="C:/venvs/brawl312/Scripts/python.exe",
        models_dir=Path("train/models/easy_bot_v0"),
        demos_dir=Path("train/models/recollected"),
        movement_init=Path("train/models/easy_bot_v0/llc_movement_fluency_masked_pilot.zip"),
        world_replay=Path("data/world_replay/easy_bot_v0_3h"),
        opponent_setup="moving_non_attacking",
        movement_steps=60_000,
        weapon_steps=60_000,
        spacing_steps=120_000,
        combat_steps=120_000,
    )

    ppo = [step for step in plan if step.kind == "ppo"]
    assert [step.phase for step in ppo] == [
        "movement_fluency",
        "weapon_acquisition",
        "spacing_neutral",
        "combat_execution",
    ]
    assert sum(int(_option(step.argv, "--timesteps")) for step in ppo) == 360_000
    assert all(step.phase != "recovery_mastery" for step in plan)

    assert _option(ppo[0].argv, "--resume").endswith(
        "llc_movement_fluency_masked_pilot.zip"
    )
    weapon_bc = next(step for step in plan if step.name == "weapon_bc")
    spacing_bc = next(step for step in plan if step.name == "spacing_bc")
    combat_bc = next(step for step in plan if step.name == "combat_bc")
    assert _option(weapon_bc.argv, "--resume").endswith("llc_movement_fluency_3h.zip")
    assert _option(spacing_bc.argv, "--resume").endswith("llc_weapon_acquisition_3h.zip")
    assert _option(combat_bc.argv, "--resume").endswith("llc_spacing_neutral_3h.zip")


def test_every_live_ppo_segment_records_world_data_but_offline_bc_does_not() -> None:
    replay_root = Path("data/world_replay/easy_bot_v0_3h")
    plan = build_run_plan(
        python="python",
        models_dir=Path("models"),
        demos_dir=Path("demos"),
        movement_init=Path("models/movement.zip"),
        world_replay=replay_root,
        opponent_setup="moving_non_attacking",
        movement_steps=1,
        weapon_steps=1,
        spacing_steps=1,
        combat_steps=1,
    )

    ppo = [step for step in plan if step.kind == "ppo"]
    assert len(ppo) == 4
    for step in ppo:
        assert _option(step.argv, "--world-replay") == str(replay_root)
        assert _option(step.argv, "--opponent-setup") == "moving_non_attacking"
        assert "--log-csv" in step.argv
        assert _option(step.argv, "--eval-every-steps") == "0"

    non_live = [step for step in plan if step.kind in {"bc", "validate", "eval"}]
    assert non_live
    assert all("--world-replay" not in step.argv for step in non_live)


def test_retention_evaluation_grows_with_the_phase_chain() -> None:
    plan = build_run_plan(
        python="python",
        models_dir=Path("models"),
        demos_dir=Path("demos"),
        movement_init=Path("models/movement.zip"),
        world_replay=Path("replay"),
        opponent_setup="moving_non_attacking",
        movement_steps=1,
        weapon_steps=1,
        spacing_steps=1,
        combat_steps=1,
    )
    evaluations = [step for step in plan if step.kind == "eval"]

    assert [_option(step.argv, "--phases") for step in evaluations] == [
        "movement_fluency",
        "movement_fluency,weapon_acquisition",
        "movement_fluency,weapon_acquisition,spacing_neutral",
        "movement_fluency,weapon_acquisition,spacing_neutral,combat_execution",
    ]
    assert len({_option(step.argv, "--best-scores") for step in evaluations}) == 1
