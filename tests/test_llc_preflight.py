from __future__ import annotations

from tools.llc_preflight import Check, check_module, check_python, summarize


def test_preflight_summary_passes_clean_checks() -> None:
    text, code = summarize([Check("python", "PASS", "ok"), Check("file:model", "PASS", "found")])

    assert code == 0
    assert "LLC PREFLIGHT: PASS" in text
    assert "Next: run the perception overlay" in text


def test_preflight_summary_warns_without_strict_warnings() -> None:
    text, code = summarize([Check("matplotlib", "WARN", "missing", "install")])

    assert code == 1
    assert "LLC PREFLIGHT: WARN" in text
    assert "Check WARN rows" in text


def test_preflight_summary_can_fail_on_warnings() -> None:
    text, code = summarize([Check("matplotlib", "WARN", "missing", "install")], strict_warnings=True)

    assert code == 2
    assert "LLC PREFLIGHT: FAIL" in text


def test_preflight_summary_fails_on_failures() -> None:
    text, code = summarize([Check("torch", "FAIL", "missing", "python -m pip install -r requirements-llc.txt")])

    assert code == 2
    assert "Stop: fix FAIL rows" in text
    assert "fix: python -m pip install -r requirements-llc.txt" in text


def test_missing_required_module_points_to_requirements_file() -> None:
    check = check_module("definitely_not_a_real_llc_module", "fake-package", "test", required=True)

    assert check.status == "FAIL"
    assert "requirements-llc.txt" in check.fix
    assert "fake-package" in check.fix


def test_python_check_accepts_current_runtime() -> None:
    check = check_python()

    assert check.status in {"PASS", "FAIL"}
    assert check.name == "python"
