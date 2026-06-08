from __future__ import annotations

import argparse

from tools.record_llc_observation import observation_status, write_observation


def _args(tmp_path, **overrides) -> argparse.Namespace:
    data = {
        "phase": "recovery_mastery",
        "approved": "yes",
        "outputs_dir": str(tmp_path),
        "notes": "returns to stage reliably",
        "movement_collapse_visible": "no",
        "recovery_reliable": "yes",
        "weapon_intentional": "na",
        "spacing_safe": "na",
        "combat_clean": "na",
    }
    data.update(overrides)
    return argparse.Namespace(**data)


def test_record_observation_writes_json_and_markdown(tmp_path) -> None:
    json_path, md_path, payload = write_observation(_args(tmp_path))

    assert json_path.exists()
    assert md_path.exists()
    assert payload["approved"] is True
    status, details = observation_status(str(tmp_path), "recovery_mastery")
    assert status == "PASS"
    assert "approved" in details[0]


def test_record_observation_failed_approval_blocks_advancement(tmp_path) -> None:
    write_observation(
        _args(
            tmp_path,
            approved="no",
            movement_collapse_visible="yes",
            notes="movement idles after ten seconds",
        )
    )

    status, details = observation_status(str(tmp_path), "recovery_mastery")

    assert status == "FAIL"
    assert any("do not advance" in item for item in details)
    assert any("movement collapse" in item for item in details)


def test_record_observation_missing_file_reports_missing(tmp_path) -> None:
    status, details = observation_status(str(tmp_path), "movement_fluency")

    assert status == "MISSING"
    assert "Manual observation missing" in details[0]
