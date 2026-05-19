from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.agency.tool_orchestrator import ToolOrchestrator
from core.container import ServiceContainer
from core.runtime.errors import DependencyUnavailable, get_degradation_tracker
from core.utils.code_guardian import CodeGuardian, ValidationReport


class CapturingResilience:
    def __init__(self) -> None:
        self.failures: list[dict[str, float | str]] = []
        self.successes: list[dict[str, float | str]] = []

    def record_failure(self, domain: str, severity: float, stakes: float):
        self.failures.append({"domain": domain, "severity": severity, "stakes": stakes})
        return SimpleNamespace(value="friction")

    def record_success(self, domain: str, stakes: float) -> None:
        self.successes.append({"domain": domain, "stakes": stakes})


class FailingSandboxLauncher:
    def __init__(self) -> None:
        self.called = False

    async def __call__(self) -> None:
        self.called = True
        raise DependencyUnavailable("sandbox boundary missing")


class LaunchShouldNotRun:
    def __init__(self) -> None:
        self.called = False

    async def __call__(self) -> None:
        self.called = True
        raise AssertionError("sandbox launch should not run after cleanup failure")


@pytest.mark.asyncio
async def test_python_sandbox_launch_failure_fails_closed(monkeypatch):
    tracker = get_degradation_tracker()
    tracker.reset()
    orch = ToolOrchestrator()
    fail_launch = FailingSandboxLauncher()

    def validation_ok(_cls, _filepath, command_runner=None):
        return ValidationReport(success=True)

    monkeypatch.setattr(CodeGuardian, "validate_code", classmethod(validation_ok))
    orch._ensure_repl = fail_launch

    success, output = await orch.execute_python("print('hello')")

    assert success is False
    assert fail_launch.called is True
    assert "Daemon protocol error" in output
    recent = tracker.recent(subsystem="tool_orchestrator", limit=1)
    assert recent
    assert recent[0].severity == "critical"
    assert "returned explicit tool failure" in recent[0].action


@pytest.mark.asyncio
async def test_python_sandbox_cleanup_failure_fails_before_launch(monkeypatch):
    orch = ToolOrchestrator()
    should_not_launch = LaunchShouldNotRun()

    def validation_ok(_cls, _filepath, command_runner=None):
        return ValidationReport(success=True)

    async def cleanup_failed(_path):
        return OSError("cleanup blocked")

    monkeypatch.setattr(CodeGuardian, "validate_code", classmethod(validation_ok))
    orch._remove_validation_file = cleanup_failed
    orch._ensure_repl = should_not_launch

    success, output = await orch.execute_python("print('hello')")

    assert success is False
    assert output == "Code validation cleanup failed: cleanup blocked"
    assert should_not_launch.called is False


@pytest.mark.asyncio
async def test_web_search_success_records_resilience_success(monkeypatch):
    resilience = CapturingResilience()

    def get_service(_cls, name: str, default=None):
        if name == "resilience_engine":
            return resilience
        return default

    orch = ToolOrchestrator()

    async def search_web(query: str) -> str:
        return f"1. {query} runtime health - https://example.com/aura"

    async def sanitize_output(data: str) -> str:
        return data

    monkeypatch.setattr(ServiceContainer, "get", classmethod(get_service))
    orch.search_web = search_web
    orch.sanitize_output = sanitize_output

    result = await orch.route_and_execute("web_search", "Aura")

    assert result.startswith("1. Aura runtime health")
    assert resilience.failures == []
    assert resilience.successes == [{"domain": "tool_execution", "stakes": 0.7}]


def test_tool_result_success_classifier_is_prefix_based():
    assert ToolOrchestrator._tool_result_succeeded("1. useful result") is True
    assert ToolOrchestrator._tool_result_succeeded("FAILED: upstream") is False
    assert ToolOrchestrator._tool_result_succeeded("ERROR: network") is False
    assert ToolOrchestrator._tool_result_succeeded("[EXECUTION FAILED]\nboom") is False
