"""Work that stood the event loop still at a boot, and at the end of a shutdown.

LIVE 2026-09-24: the first skill after a boot built the persistent-state store
on the loop, importing sqlalchemy, and the loop stood still for 5.1 s; and the
last verdict of a shutdown was fsynced on the loop thread, because the pool it
would have gone to had already shut down.
"""
from __future__ import annotations

import asyncio
import threading

import pytest

from core.runtime import executors


@pytest.mark.asyncio
async def test_a_call_whose_pool_has_gone_still_runs_off_the_loop(monkeypatch):
    loop_thread = threading.get_ident()

    async def gone(*_a, **_k):
        raise RuntimeError("cannot schedule new futures after shutdown")

    monkeypatch.setattr(executors.asyncio, "to_thread", gone)
    ran_on = await executors.off_the_loop(threading.get_ident)
    assert ran_on != loop_thread


@pytest.mark.asyncio
async def test_what_it_raises_reaches_the_caller(monkeypatch):
    async def gone(*_a, **_k):
        raise RuntimeError("Executor shutdown has been called")

    def fails():
        raise ValueError("the disk said no")

    monkeypatch.setattr(executors.asyncio, "to_thread", gone)
    with pytest.raises(ValueError, match="the disk said no"):
        await executors.off_the_loop(fails)


@pytest.mark.asyncio
async def test_any_other_failure_is_not_taken_for_a_pool_that_has_gone(monkeypatch):
    async def broken(*_a, **_k):
        raise RuntimeError("something else")

    monkeypatch.setattr(executors.asyncio, "to_thread", broken)
    with pytest.raises(RuntimeError, match="something else"):
        await executors.off_the_loop(lambda: 1)


@pytest.mark.asyncio
async def test_a_first_build_of_a_service_happens_off_the_loop():
    from core.container import ServiceContainer, ServiceLifetime

    loop_thread = threading.get_ident()
    built_on: list[int] = []

    def factory():
        built_on.append(threading.get_ident())
        return object()

    name = "test_first_build_off_the_loop"
    ServiceContainer.register(name, factory, lifetime=ServiceLifetime.SINGLETON)
    try:
        first = await ServiceContainer.get_async(name, default=None)
        again = await ServiceContainer.get_async(name, default=None)
    finally:
        ServiceContainer._services.pop(name, None)
    assert first is not None and again is first
    assert built_on and built_on[0] != loop_thread
    assert len(built_on) == 1, "a built service is handed back, not built again"


@pytest.mark.asyncio
async def test_an_absent_service_gives_the_default():
    from core.container import ServiceContainer

    assert await ServiceContainer.get_async("no_such_service_anywhere", default=None) is None


def test_the_skill_path_asks_for_its_services_without_building_them_on_the_loop():
    import inspect

    from core import capability_engine

    source = inspect.getsource(capability_engine)
    assert 'await rt.container.get_async("persistent_state"' in source
    assert 'await rt.container.get_async("memory_governor"' in source


def test_the_shutdown_verdict_is_written_through_the_lane_that_keeps_it_off_the_loop():
    import inspect

    from core.ops import graceful_shutdown

    source = inspect.getsource(graceful_shutdown)
    assert "off_the_loop(publish_shutdown_verdict" in source
    assert "\n                    publish_shutdown_verdict(**verdict)" not in source


class _Request:
    def __init__(self, gone: bool) -> None:
        self._gone = gone

    async def is_disconnected(self) -> bool:
        return self._gone


@pytest.mark.asyncio
async def test_a_request_nobody_waits_for_is_an_ending(monkeypatch):
    from interface import server

    assert await server._nobody_is_waiting(_Request(True), RuntimeError("No response returned."))
    monkeypatch.setattr(
        "core.runtime.shutdown_coordinator.is_shutdown_requested", lambda: True
    )
    assert await server._nobody_is_waiting(_Request(False), RuntimeError("No response returned."))


@pytest.mark.asyncio
async def test_a_request_somebody_still_waits_for_is_an_error(monkeypatch):
    from interface import server

    monkeypatch.setattr(
        "core.runtime.shutdown_coordinator.is_shutdown_requested", lambda: False
    )
    assert not await server._nobody_is_waiting(_Request(False), RuntimeError("No response returned."))
    assert not await server._nobody_is_waiting(_Request(True), RuntimeError("something else"))
    assert not await server._nobody_is_waiting(_Request(True), ValueError("No response returned."))


def test_the_loop_is_left_free_while_a_call_runs_on_its_own_thread(monkeypatch):
    """The point of the fallback: the loop keeps turning."""
    import time

    async def gone(*_a, **_k):
        raise RuntimeError("cannot schedule new futures after shutdown")

    async def both():
        ticks = 0

        async def tick():
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.01)

        ticker = asyncio.create_task(tick())
        await executors.off_the_loop(time.sleep, 0.2)
        ticker.cancel()
        return ticks

    monkeypatch.setattr(executors.asyncio, "to_thread", gone)
    assert asyncio.run(both()) >= 5


@pytest.mark.asyncio
async def test_the_call_on_its_own_thread_runs_in_the_callers_context(monkeypatch):
    """A governed write is allowed by the scope its caller holds, which lives in the context."""
    import contextvars

    scope: contextvars.ContextVar[str] = contextvars.ContextVar("scope", default="none")

    async def gone(*_a, **_k):
        raise RuntimeError("cannot schedule new futures after shutdown")

    monkeypatch.setattr(executors.asyncio, "to_thread", gone)
    scope.set("governed")
    assert await executors.off_the_loop(scope.get) == "governed"
