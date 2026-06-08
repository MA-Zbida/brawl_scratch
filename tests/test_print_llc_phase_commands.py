from __future__ import annotations

import argparse

from tools.print_llc_phase_commands import commands_for_phase


def _args(**overrides) -> argparse.Namespace:
    data = {
        "models_dir": "train/models",
        "device": "$device",
        "best_scores": "$best",
        "python": "python",
        "bc_epochs": 20,
        "eval_episodes": 5,
        "all_skills_eval_episodes": 10,
        "timesteps": 0,
        "plot": True,
    }
    data.update(overrides)
    return argparse.Namespace(**data)


def test_middle_phase_commands_include_previous_resume_and_demo_chain() -> None:
    commands = commands_for_phase(_args(), "weapon_acquisition")
    joined = "\n".join(commands)

    assert commands[0] == "python tools/validate_llc_demos.py --phase weapon_acquisition --min-samples 100"
    assert commands[1].startswith("python -m train.pretrain_bc_locomotion --phase weapon_acquisition")
    assert "--resume train/models/llc_movement_fluency.zip" in commands[1]
    assert "--demos train/models/weapon_acquisition_demos.npz" in commands[1]
    assert "--resume train/models/llc_weapon_acquisition_bc_init.zip" in commands[2]
    assert (
        '--bc-demos-path "train/models/recovery_mastery_demos.npz;'
        "train/models/movement_fluency_demos.npz;"
        'train/models/weapon_acquisition_demos.npz"'
    ) in commands[2]
    assert "--phases recovery_mastery,movement_fluency,weapon_acquisition" in commands[3]
    assert "tools/check_llc_phase_gate.py" in joined
    assert "tools/plot_llc_diagnostics.py" in joined
    assert "tools/summarize_llc_run.py" in joined
    assert "--out outputs/llc_weapon_acquisition_run_report.md" in joined
    assert "tools/record_llc_observation.py --phase weapon_acquisition --approved yes" in joined


def test_all_skills_commands_skip_bc_pretrain_and_use_all_demos() -> None:
    commands = commands_for_phase(_args(), "all_skills_llc")
    joined = "\n".join(commands)

    assert not any("pretrain_bc_locomotion" in command for command in commands)
    assert commands[0] == "python tools/validate_llc_demos.py --phase all --min-samples 100"
    assert commands[1].startswith("python -m train.train_curriculum --phase all_skills_llc")
    assert "--resume train/models/llc_combat_execution.zip" in commands[1]
    assert "--eval-phases all" in commands[1]
    assert "--phases all" in commands[2]
    assert "--phases all" in commands[3]
    assert "--phases all" in joined
    assert "tools/summarize_llc_run.py" in joined
    assert "--out outputs/llc_all_skills_llc_run_report.md" in joined
    assert "tools/record_llc_observation.py --phase all_skills_llc --approved yes" in joined
    for phase in (
        "recovery_mastery",
        "movement_fluency",
        "weapon_acquisition",
        "spacing_neutral",
        "combat_execution",
    ):
        assert f"train/models/{phase}_demos.npz" in joined
