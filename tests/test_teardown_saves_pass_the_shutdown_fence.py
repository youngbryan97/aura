"""What teardown saves passes the shutdown fence; other new work does not.

LIVE 11 November 2026, the first shutdown after the lanes said who asked: the outcome ledger's drain of her
chat-turn memory was refused by the receipt lane, and every stop() that saved behind the loop (the aesthetic
journal, the belief graph, the epistemic map, the router's history) lost its save. The lane refused the write,
its fallback thread was refused by runtime hygiene, and each stop failed with "runtime_shutdown".
"""
from __future__ import annotations

import asyncio
import threading

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def shutting_down(monkeypatch):
    import core.runtime.executors as executors
    import core.runtime.shutdown_coordinator as shutdown

    monkeypatch.setattr(executors, "_REFUSED_AT_SHUTDOWN", set())
    monkeypatch.setattr(shutdown, "is_shutdown_requested", lambda: True)


def test_a_save_in_teardown_gets_a_lane_and_new_work_outside_it_does_not(shutting_down):
    from core.runtime.executors import submit_blocking_io
    from core.utils.task_tracker import begin_shutdown_task_creation_scope, end_shutdown_task_creation_scope

    with pytest.raises(RuntimeError, match="runtime_shutdown"):
        submit_blocking_io(lambda: None, label="background work")
    token = begin_shutdown_task_creation_scope()
    try:
        assert submit_blocking_io(lambda: "saved", label="a stop's save").result(timeout=5) == "saved"
    finally:
        end_shutdown_task_creation_scope(token)


def test_the_fallback_thread_is_teardown_work_hygiene_lets_start(monkeypatch):
    """Runtime hygiene refuses a thread once shutdown has latched unless the call site may create teardown
    resources, which is what the write-behind fallback is."""
    from core.runtime.executors import start_teardown_thread
    from core.utils.task_tracker import shutdown_resource_creation_allowed

    original = threading.Thread.start

    def hygiene(thread, *args, **kwargs):
        if not shutdown_resource_creation_allowed():
            raise RuntimeError("runtime_shutdown")
        return original(thread, *args, **kwargs)

    monkeypatch.setattr(threading.Thread, "start", hygiene)
    done = threading.Event()
    start_teardown_thread(done.set, name="a save behind the loop")
    assert done.wait(5)
    with pytest.raises(RuntimeError, match="runtime_shutdown"):
        threading.Thread(target=lambda: None).start()


def test_a_shutdown_handler_runs_as_teardown():
    from core.runtime.shutdown_coordinator import ShutdownCoordinator, ShutdownReport, _RegisteredHandler
    from core.utils.task_tracker import shutdown_task_creation_allowed

    seen: list[bool] = []

    async def drain():
        seen.append(shutdown_task_creation_allowed())

    coordinator = ShutdownCoordinator()
    record = _RegisteredHandler(name="outcome_ledger", phase="memory_commit", handler=drain, timeout=5.0)
    asyncio.run(coordinator._invoke(record, ShutdownReport()))
    assert seen == [True]
    assert shutdown_task_creation_allowed() is False
