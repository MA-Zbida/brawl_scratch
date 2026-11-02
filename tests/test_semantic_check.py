from __future__ import annotations

import json

import pytest

from tools.check_semantic_actions import main


def test_dry_run_never_loads_live_controller(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("Dry run tried to inject keys")

    monkeypatch.setattr("control.input.PyDirectInputController", forbidden)
    path = tmp_path / "check.json"
    main(["--action", "GC_SIGNATURE_TOWARD", "--context", "air",
          "--mirrored", "--output", str(path)])
    data = json.loads(path.read_text())
    assert data["simulated"] is True
    assert data["game_move_verified"] is False
    assert data["outcomes"][-1]["status"] == "completed"
    assert [row["key"] for row in data["events"] if row["event"] == "down"] == ["e", "num6"]


def test_output_refuses_overwrite(tmp_path):
    path = tmp_path / "existing.json"
    path.write_text("original")
    with pytest.raises(FileExistsError):
        main(["--action", "JUMP", "--output", str(path)])
    assert path.read_text() == "original"


def test_live_requires_explicit_context_for_contextual_actions():
    with pytest.raises(SystemExit):
        main(["--action", "RECOVERY_TOWARD", "--execute"])


def test_list_is_readonly(capsys):
    main(["--list"])
    output = capsys.readouterr().out
    assert "RECOVERY_TOWARD" in output
    assert "SIGNATURE_TOWARD" in output


def test_live_pickup_requires_explicit_possession(monkeypatch):
    def forbidden():
        raise AssertionError("Argument validation must precede controller creation")

    monkeypatch.setattr("control.input.PyDirectInputController", forbidden)
    with pytest.raises(SystemExit):
        main(["--action", "PICKUP", "--execute"])


def test_live_loop_with_fake_clock_and_controller(monkeypatch, tmp_path):
    from control.input import NullInputController

    created = []
    now = [0.]
    monkeypatch.setattr("control.input.PyDirectInputController",
                        lambda: created.append(True) or NullInputController())
    monkeypatch.setattr("time.perf_counter", lambda: now[0])
    monkeypatch.setattr("time.sleep", lambda delay: now.__setitem__(0, now[0] + delay))
    path = tmp_path / "live_fake.json"
    main(["--action", "RECOVERY_TOWARD", "--context", "air", "--execute",
          "--delay", "0", "--output", str(path)])
    data = json.loads(path.read_text())
    assert created == [True]
    assert data["simulated"] is False
    assert data["game_move_verified"] is False
    assert data["outcomes"][-1]["status"] == "completed"


@pytest.mark.parametrize("args", [[], ["--action", "JUMP", "--delay", "nan"]])
def test_bad_cli_arguments_fail(args):
    with pytest.raises(SystemExit):
        main(args)
