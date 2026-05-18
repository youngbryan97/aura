from core.resilience.code_verifier import CodeVerifier
from core.tasks.managed_command import ManagedCommandResult


def _result(command: tuple[str, ...], returncode: int | None = 0, *, timed_out: bool = False) -> ManagedCommandResult:
    return ManagedCommandResult(command, returncode, "compiled", "", 0.01, timed_out=timed_out)


def test_importability_report_uses_compile_command_without_running_module():
    calls: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...], timeout_s: float) -> ManagedCommandResult:
        calls.append(command)
        return _result(command)

    report = CodeVerifier.verify_importability_report(
        "VALUE = 1\n",
        module_name="../../bad-name",
        timeout=7,
        command_runner=runner,
    )

    assert report.ok is True
    assert report.syntax_ok is True
    assert report.safety_ok is True
    assert len(calls) == 1
    assert calls[0][1:3] == ("-m", "py_compile")
    assert calls[0][-1].endswith("candidate_module.py")


def test_importability_report_blocks_dangerous_import_even_when_compile_passes():
    def runner(command: tuple[str, ...], timeout_s: float) -> ManagedCommandResult:
        return _result(command)

    report = CodeVerifier.verify_importability_report("import os\nVALUE = 1\n", command_runner=runner)

    assert report.ok is False
    assert report.syntax_ok is True
    assert report.safety_ok is False
    assert report.warnings == ("Imports dangerous module: os",)
    assert CodeVerifier.verify_importability("import os\n", command_runner=runner) is False


def test_importability_report_returns_syntax_failure_without_command():
    calls = []

    def runner(command: tuple[str, ...], timeout_s: float) -> ManagedCommandResult:
        calls.append(command)
        return _result(command)

    report = CodeVerifier.verify_importability_report("def broken(:\n", command_runner=runner)

    assert report.ok is False
    assert report.syntax_ok is False
    assert report.warnings == ("Syntax error",)
    assert calls == []


def test_importability_report_reports_compile_timeout():
    def runner(command: tuple[str, ...], timeout_s: float) -> ManagedCommandResult:
        return _result(command, None, timed_out=True)

    report = CodeVerifier.verify_importability_report("VALUE = 1\n", command_runner=runner)

    assert report.ok is False
    assert report.timed_out is True


def test_importability_report_reports_compile_failure():
    def runner(command: tuple[str, ...], timeout_s: float) -> ManagedCommandResult:
        return ManagedCommandResult(command, 1, "", "compile failed", 0.01)

    report = CodeVerifier.verify_importability_report("VALUE = 1\n", command_runner=runner)

    assert report.ok is False
    assert report.stderr == "compile failed"
