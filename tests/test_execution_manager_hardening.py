import asyncio

import pytest

from core.brain.execution import ExecutionManager
from core.runtime.errors import get_degradation_tracker


class TraceRecorder:
    def __init__(self):
        self.events: list[dict] = []

    def log(self, event: dict) -> None:
        self.events.append(dict(event))


@pytest.mark.asyncio
async def test_execution_manager_propagates_cancellation():
    trace = TraceRecorder()
    manager = ExecutionManager(trace)

    async def cancelled_action():
        cancelled_action.called = True
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await manager.execute("cancelled", cancelled_action)

    assert getattr(cancelled_action, "called", False) is True
    assert trace.events[-1]["type"] == "execution_cancelled"


@pytest.mark.asyncio
async def test_execution_manager_records_recoverable_failure():
    get_degradation_tracker().reset()
    trace = TraceRecorder()
    manager = ExecutionManager(trace)

    async def failing_action():
        failing_action.called = True
        raise RuntimeError("boom")

    result = await manager.execute("explode", failing_action, retries=1)

    assert result.ok is False
    assert result.error == "boom"
    assert getattr(failing_action, "called", False) is True
    assert trace.events[-1]["type"] == "execution_exception"
    assert any(
        "action callable failed" in record.action
        for record in get_degradation_tracker().recent(subsystem="execution")
    )


@pytest.mark.asyncio
async def test_execution_manager_runs_sync_callable_once_when_retries_zero():
    trace = TraceRecorder()
    manager = ExecutionManager(trace)
    calls = []

    def sync_action():
        calls.append("called")
        return "ok"

    result = await manager.execute("sync", sync_action, retries=0)

    assert result.ok is True
    assert result.result == "ok"
    assert calls == ["called"]
