import subprocess

import pytest

from core.runtime.errors import get_degradation_tracker
from core.self_modification.code_repair import CodeFix, SandboxTester


def _fix(target_file: str) -> CodeFix:
    return CodeFix(
        target_file=target_file,
        target_line=1,
        original_code="return 1",
        fixed_code="return 2",
        explanation="unit test",
        hypothesis="unit test",
        confidence="high",
    )


@pytest.fixture(autouse=True)
def _reset_tracker():
    get_degradation_tracker().reset()
    yield
    get_degradation_tracker().reset()


@pytest.mark.asyncio
async def test_core_repair_fails_closed_when_pyright_guard_unavailable(monkeypatch, tmp_path):
    sandbox_file = tmp_path / "core" / "module.py"
    sandbox_file.parent.mkdir()
    sandbox_file.write_text("def value():\n    return 1\n", encoding="utf-8")

    monkeypatch.setattr(
        "core.resilience.diagnostic_hub.get_diagnostic_hub",
        lambda: (_ for _ in ()).throw(RuntimeError("diagnostic hub offline")),
    )

    result = await SandboxTester()._run_tests_in_sandbox(tmp_path, _fix("core/module.py"))

    assert result["success"] is False
    assert any("Pyright guard unavailable" in error for error in result["errors"])
    last = get_degradation_tracker().recent(subsystem="code_repair")[-1]
    assert (
        last.action
        == "failed closed core repair validation after type guard became unavailable"
    )


@pytest.mark.asyncio
async def test_sandbox_validation_fails_closed_when_unit_test_runner_fails(monkeypatch, tmp_path):
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()
    (package_dir / "module.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    (package_dir / "test_module.py").write_text("def test_value():\n    assert True\n", encoding="utf-8")

    class _Hub:
        async def _run_pyright(self, _path):
            return {"ok": True}

    def _run_raises(*_args, **_kwargs):
        raise subprocess.SubprocessError("pytest transport failed")

    monkeypatch.setattr("core.resilience.diagnostic_hub.get_diagnostic_hub", lambda: _Hub())
    monkeypatch.setattr("core.self_modification.code_repair.subprocess.run", _run_raises)

    result = await SandboxTester()._run_tests_in_sandbox(tmp_path, _fix("pkg/module.py"))

    assert result["success"] is False
    assert result["unit_tests"] is False
    assert any("Test execution failed" in error for error in result["errors"])
    last = get_degradation_tracker().recent(subsystem="code_repair")[-1]
    assert last.action == "failed closed sandbox validation after unit test runner failed"
