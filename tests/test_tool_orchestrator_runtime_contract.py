from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import sys
import threading
from types import SimpleNamespace

import pytest

from core.agency import repl_daemon
from core.agency.tool_orchestrator import (
    SandboxTransportError,
    ToolOrchestrator,
    get_tool_orchestrator,
)
from core.container import ServiceContainer
from core.governance_context import local_internal_governed_scope
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
async def test_worker_stderr_stream_has_a_hard_total_budget():
    stream = asyncio.StreamReader()
    stream.feed_data(b"x" * (1024 * 1024 + 1))
    stream.feed_eof()

    with pytest.raises(SandboxTransportError, match="stderr exceeded"):
        await ToolOrchestrator._drain_stderr_tail(stream)


def _worker_payloads(monkeypatch, code: str) -> list[dict]:
    request = {
        "version": repl_daemon.PROTOCOL_VERSION,
        "kind": "execute",
        "request_id": "request-1",
        "authority_id": "authority-1",
        "deadline_ms": 1000,
        "code": code,
    }
    input_stream = io.BytesIO(
        repl_daemon.encode_frame(request, max_bytes=repl_daemon.MAX_REQUEST_BYTES)
    )
    output_stream = io.BytesIO()
    monkeypatch.setattr(
        repl_daemon.sys,
        "stdin",
        SimpleNamespace(buffer=input_stream),
    )
    monkeypatch.setattr(
        repl_daemon.sys,
        "stdout",
        SimpleNamespace(buffer=output_stream),
    )
    monkeypatch.setattr(repl_daemon, "_apply_resource_limits", lambda: None)

    repl_daemon.main()

    frames = []
    output_stream.seek(0)
    while output_stream.tell() < len(output_stream.getvalue()):
        frames.append(
            repl_daemon.read_frame(
                output_stream,
                max_bytes=repl_daemon.MAX_RESPONSE_BYTES,
            )
        )
    return frames


def test_repl_worker_executes_one_correlated_utf8_request(monkeypatch):
    payloads = _worker_payloads(monkeypatch, "print('hello π')")

    assert payloads[0]["kind"] == "ready"
    assert payloads[1] == {
        "version": 1,
        "kind": "result",
        "request_id": "request-1",
        "authority_id": "authority-1",
        "success": True,
        "output": "hello π\n",
        "output_truncated": False,
    }


def test_repl_worker_bounds_output_without_corrupting_frame(monkeypatch):
    payloads = _worker_payloads(monkeypatch, "print('x' * 900000)")

    result = payloads[1]
    assert result["success"] is True
    assert result["output_truncated"] is True
    assert result["output"].endswith("[output truncated by sandbox limit]\n")
    assert len(result["output"].encode("utf-8")) < repl_daemon.MAX_RESPONSE_BYTES

    escaped = _worker_payloads(monkeypatch, "print(chr(0) * 200000)")[1]
    assert escaped["success"] is True
    assert escaped["output_truncated"] is True
    assert len(escaped["output"].encode("utf-8")) < repl_daemon.MAX_RESPONSE_BYTES


def test_canonical_tool_orchestrator_getter_is_process_singleton():
    assert get_tool_orchestrator() is get_tool_orchestrator()


@pytest.mark.asyncio
async def test_python_sandbox_launch_failure_fails_closed(monkeypatch):
    tracker = get_degradation_tracker()
    tracker.reset()
    orch = ToolOrchestrator()
    fail_launch = FailingSandboxLauncher()

    def validation_ok(_cls, _filepath, command_runner=None):
        return ValidationReport(success=True)

    monkeypatch.setattr(CodeGuardian, "validate_code", classmethod(validation_ok))
    orch._spawn_ready_worker = fail_launch

    success, output = await orch.execute_python("print('hello')")

    assert success is False
    assert fail_launch.called is True
    assert "Sandbox transport failed before dispatch" in output
    recent = tracker.recent(subsystem="tool_orchestrator", limit=1)
    assert recent
    assert recent[0].severity == "critical"
    assert "failed closed before sandbox code dispatch" in recent[0].action


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
    orch._spawn_ready_worker = should_not_launch

    success, output = await orch.execute_python("print('hello')")

    assert success is False
    assert output == "Code validation cleanup failed: cleanup blocked"
    assert should_not_launch.called is False


@pytest.mark.asyncio
async def test_oversized_python_is_rejected_before_worker_launch(monkeypatch):
    orch = ToolOrchestrator()
    should_not_launch = LaunchShouldNotRun()
    orch._spawn_ready_worker = should_not_launch

    success, output = await orch.execute_python("value = 1\n" * 13000)

    assert success is False
    assert "exceeds 120 KiB" in output
    assert should_not_launch.called is False


@pytest.mark.asyncio
async def test_syntax_checked_python_keeps_native_worker_boundary(monkeypatch):
    orch = ToolOrchestrator()
    observed = {}

    async def admitted(code):
        observed["code"] = code
        return True, "ok"

    monkeypatch.setattr(orch, "_execute_admitted_python", admitted)

    assert await orch.execute_syntax_checked_python("value = 1") == (True, "ok")
    assert observed["code"] == "value = 1"


@pytest.mark.asyncio
async def test_syntax_checked_python_refuses_invalid_source_before_launch(monkeypatch):
    orch = ToolOrchestrator()
    should_not_launch = LaunchShouldNotRun()
    orch._spawn_ready_worker = should_not_launch

    success, output = await orch.execute_syntax_checked_python("def broken(:")

    assert success is False
    assert "Code Validation Failed" in output
    assert should_not_launch.called is False


@pytest.mark.asyncio
async def test_code_guardian_validation_does_not_block_event_loop(monkeypatch):
    orch = ToolOrchestrator()
    release = threading.Event()

    def blocking_validation(_cls, _filepath, command_runner=None):
        release.wait(1.0)
        return ValidationReport(success=True)

    async def admitted(_code):
        return True, "ok"

    monkeypatch.setattr(
        CodeGuardian,
        "validate_code",
        classmethod(blocking_validation),
    )
    monkeypatch.setattr(orch, "_execute_admitted_python", admitted)
    loop = asyncio.get_running_loop()
    loop.call_later(0.05, release.set)
    started = loop.time()

    assert await orch.execute_python("print('hello')") == (True, "ok")
    assert loop.time() - started < 0.4


@pytest.mark.asyncio
async def test_native_python_worker_enforces_real_macos_boundaries(monkeypatch):
    if sys.platform != "darwin" or shutil.which("sandbox-exec") is None:
        return
    orch = ToolOrchestrator()
    parent_secret = "must-not-cross-sandbox-boundary"
    monkeypatch.setenv("AURA_TEST_PARENT_SECRET", parent_secret)
    home_path = json.dumps(os.environ.get("HOME", "/Users"))

    with local_internal_governed_scope(
        "test.tool_orchestrator.native_boundaries",
        domain="tool_execution",
    ):
        success, output = await orch._execute_admitted_python("print('hello π')")
        assert (success, output) == (True, "hello π\n")

        success, output = await orch._execute_admitted_python(
            "import os\nprint(os.getenv('AURA_TEST_PARENT_SECRET'))"
        )
        assert (success, output) == (True, "None\n")

        success, output = await orch._execute_admitted_python(
            f"import os\nprint(os.listdir({home_path})[:1])"
        )
        assert success is False
        assert "Operation not permitted" in output

        success, output = await orch._execute_admitted_python(
            "import subprocess\nsubprocess.run(['/bin/echo', 'escaped'], check=True)"
        )
        assert success is False
        assert "Operation not permitted" in output

        success, output = await orch._execute_admitted_python(
            "import socket\nsocket.create_connection(('example.com', 80), timeout=0.1)"
        )
        assert success is False
        assert "socket" in output.lower()

        success, output = await orch._execute_admitted_python(
            "import numpy as np\nprint(int(np.array([1, 2, 3]).sum()))"
        )
        assert (success, output) == (True, "6\n")

    assert orch.get_status()["worker_active"] is False
    await orch.shutdown()


@pytest.mark.asyncio
async def test_post_dispatch_worker_death_is_not_replayed(monkeypatch):
    if sys.platform != "darwin" or shutil.which("sandbox-exec") is None:
        return
    orch = ToolOrchestrator()
    launches = 0
    launch_worker = orch._launch_worker

    async def counted_launch():
        nonlocal launches
        launches += 1
        return await launch_worker()

    monkeypatch.setattr(orch, "_launch_worker", counted_launch)
    with local_internal_governed_scope(
        "test.tool_orchestrator.no_replay",
        domain="tool_execution",
    ):
        success, output = await orch._execute_admitted_python("import os\nos._exit(7)")

    assert success is False
    assert "indeterminate" in output
    assert "not replayed" in output
    assert launches == 1
    assert orch.get_status()["worker_active"] is False


@pytest.mark.asyncio
async def test_cancelled_python_worker_is_reaped(monkeypatch):
    if sys.platform != "darwin" or shutil.which("sandbox-exec") is None:
        return
    orch = ToolOrchestrator()
    with local_internal_governed_scope(
        "test.tool_orchestrator.cancel",
        domain="tool_execution",
    ):
        task = asyncio.create_task(
            orch._execute_admitted_python("while True:\n    pass")
        )
        for _ in range(100):
            if orch._active_process is not None:
                break
            await asyncio.sleep(0.01)
        assert orch._active_process is not None
        pid = orch._active_process.pid
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert orch.get_status()["worker_active"] is False
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_transport_circuit_opens_after_three_recent_faults():
    orch = ToolOrchestrator()

    orch._record_transport_fault("one")
    orch._record_transport_fault("two")
    assert orch.get_status()["circuit_open"] is False
    orch._record_transport_fault("three")

    assert orch.get_status()["circuit_open"] is True
    assert orch.get_status()["transport_faults_60s"] == 3


@pytest.mark.asyncio
async def test_transport_failure_never_enters_code_auto_repair(monkeypatch):
    engine_calls = []

    class Engine:
        async def think(self, *_args, **_kwargs):
            engine_calls.append(True)
            return SimpleNamespace(content="print('replayed')")

    def get_service(_cls, name: str, default=None):
        if name == "cognitive_engine":
            return Engine()
        return default

    orch = ToolOrchestrator()
    execution_calls = 0

    async def failed_transport(_code):
        nonlocal execution_calls
        execution_calls += 1
        orch._last_python_failure_kind = "transport"
        return False, "indeterminate and not replayed"

    async def identity_sanitizer(value):
        return value

    monkeypatch.setattr(ServiceContainer, "get", classmethod(get_service))
    monkeypatch.setattr(orch, "execute_python", failed_transport)
    monkeypatch.setattr(orch, "sanitize_output", identity_sanitizer)

    result = await orch.route_and_execute("python_sandbox", "print('once')")

    assert "indeterminate and not replayed" in result
    assert execution_calls == 1
    assert engine_calls == []


@pytest.mark.asyncio
async def test_a_code_failure_goes_back_as_the_result_and_not_to_her_engine(monkeypatch):
    """A failed run's error is the tool's result, for the turn that wrote the code. It used to be sent to her
    cognitive engine as "Rewrite the code to fix the error...", which became her objective and a task commitment
    retried for days (83 in her store, 13 to 17 October 2026)."""
    engine_calls = []

    class Engine:
        async def think(self, *_args, **_kwargs):
            engine_calls.append(True)
            return SimpleNamespace(content="print('rewritten')")

    def get_service(_cls, name: str, default=None):
        if name == "cognitive_engine":
            return Engine()
        return default

    orch = ToolOrchestrator()
    execution_calls = 0

    async def failed_code(_code):
        nonlocal execution_calls
        execution_calls += 1
        orch._last_python_failure_kind = "code"
        return False, "NameError: name 'x' is not defined"

    async def identity_sanitizer(value):
        return value

    monkeypatch.setattr(ServiceContainer, "get", classmethod(get_service))
    monkeypatch.setattr(orch, "execute_python", failed_code)
    monkeypatch.setattr(orch, "sanitize_output", identity_sanitizer)

    result = await orch.route_and_execute("python_sandbox", "print(x)")

    assert result == "[EXECUTION FAILED]\nNameError: name 'x' is not defined"
    assert execution_calls == 1
    assert engine_calls == []


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
