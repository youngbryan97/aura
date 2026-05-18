from pathlib import Path

from core.tasks.managed_command import ManagedCommandResult
from core.utils.code_guardian import CodeGuardian


def _result(command: tuple[str, ...], returncode: int | None = 0, *, timed_out: bool = False) -> ManagedCommandResult:
    return ManagedCommandResult(command, returncode, "out", "err", 0.01, timed_out=timed_out)


def test_code_guardian_runs_ruff_then_mypy_through_runner(tmp_path: Path):
    target = tmp_path / "candidate.py"
    target.write_text("VALUE = 1\n")
    calls: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...], timeout_s: float) -> ManagedCommandResult:
        calls.append(command)
        return _result(command)

    report = CodeGuardian.validate_code(target, command_runner=runner)

    assert report.success is True
    assert len(calls) == 2
    assert calls[0][1] == "check"
    assert calls[1][1] == "--ignore-missing-imports"
    assert calls[0][-1] == str(target)
    assert calls[1][-1] == str(target)


def test_code_guardian_stops_after_ruff_failure(tmp_path: Path):
    target = tmp_path / "candidate.py"
    target.write_text("VALUE = 1\n")
    calls: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...], timeout_s: float) -> ManagedCommandResult:
        calls.append(command)
        return _result(command, 1)

    report = CodeGuardian.validate_code(target, command_runner=runner)

    assert report.success is False
    assert report.error_message == "Syntax or NameError detected by Ruff."
    assert report.ruff_output == "outerr"
    assert len(calls) == 1


def test_code_guardian_reports_mypy_timeout(tmp_path: Path):
    target = tmp_path / "candidate.py"
    target.write_text("VALUE = 1\n")

    def runner(command: tuple[str, ...], timeout_s: float) -> ManagedCommandResult:
        if command[1] == "check":
            return _result(command)
        return _result(command, None, timed_out=True)

    report = CodeGuardian.validate_code(target, command_runner=runner)

    assert report.success is False
    assert report.error_message == "Mypy check timed out."


def test_code_guardian_reports_runner_launch_error(tmp_path: Path):
    target = tmp_path / "candidate.py"
    target.write_text("VALUE = 1\n")
    calls: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...], timeout_s: float) -> ManagedCommandResult:
        calls.append(command)
        raise OSError("tool unavailable")

    report = CodeGuardian.validate_code(target, command_runner=runner)

    assert report.success is False
    assert "Ruff execution error" in report.error_message
    assert len(calls) == 1
