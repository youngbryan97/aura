import asyncio

import pytest

from core.runtime.control_plane import (
    DesiredServiceSpec,
    PressureSnapshot,
    ResourceAdmissionController,
    RuntimeControlPlane,
)
from core.supervisor.tree import SupervisionTree


@pytest.mark.asyncio
async def test_unavailable_probe_preserves_service_and_dependency_until_observed():
    plane = RuntimeControlPlane(admission=ResourceAdmissionController(
        pressure_provider=lambda: PressureSnapshot(memory_percent=40.0),
    ))
    outcome = "timeout"
    stopped = []

    def probe():
        if outcome == "timeout":
            raise TimeoutError("observation could not be scheduled")
        return outcome == "healthy"

    for name, dependencies in (("monitor", ()), ("dependent", ("monitor",))):
        plane.register_service(
            DesiredServiceSpec(name=name, dependencies=dependencies, critical=True),
            start=lambda: None,
            stop=lambda name=name: stopped.append(name),
            probe=probe if name == "monitor" else lambda: True,
            adopt_running=True,
        )

    for _ in range(2):
        report = await plane.reconcile_once()
        assert stopped == []
        assert report["critical_ready"] is False
        assert report["services"]["monitor"]["reason"] == "probe_unavailable"
        assert report["services"]["monitor"]["restart_times"] == []

    outcome = "healthy"
    assert (await plane.reconcile_once())["critical_ready"] is True
    assert stopped == []

    outcome = "dead"
    await plane.reconcile_once()
    assert "monitor" in stopped
    assert "dependent" in stopped


@pytest.mark.asyncio
async def test_real_probe_deadline_does_not_stop_the_observed_service():
    plane = RuntimeControlPlane(admission=ResourceAdmissionController(
        pressure_provider=lambda: PressureSnapshot(memory_percent=40.0),
    ))
    stopped = []

    async def unavailable():
        await asyncio.Event().wait()

    plane.register_service(
        DesiredServiceSpec(name="slow_observer", start_timeout_s=0.01, critical=True),
        start=lambda: None,
        stop=lambda: stopped.append(True),
        probe=unavailable,
        adopt_running=True,
    )
    report = await plane.reconcile_once()
    assert stopped == []
    assert report["services"]["slow_observer"]["probe_available"] is False
    assert report["critical_ready"] is False


@pytest.mark.asyncio
async def test_desktop_lifetime_survives_managed_monitor_replacement(monkeypatch):
    tree = SupervisionTree()
    monkeypatch.setattr(tree, "_register_with_control_plane", lambda: None)
    monkeypatch.setattr(tree, "_publish_conditions", lambda: None)
    stop = asyncio.Event()
    root = asyncio.create_task(tree.wait_forever(until=stop))
    try:
        await asyncio.sleep(0)
        first = tree._monitor_task
        await tree.stop()
        await asyncio.sleep(0)
        assert not root.done()
        await tree.start()
        assert tree._monitor_task is not first
        assert tree.is_alive()
        stop.set()
        await asyncio.wait_for(root, 2.0)
    finally:
        if not root.done():
            root.cancel()
        await asyncio.gather(root, return_exceptions=True)
        await tree.stop()


@pytest.mark.asyncio
async def test_desktop_wait_still_obeys_explicit_cancellation(monkeypatch):
    tree = SupervisionTree()
    monkeypatch.setattr(tree, "_register_with_control_plane", lambda: None)
    monkeypatch.setattr(tree, "_publish_conditions", lambda: None)
    root = asyncio.create_task(tree.wait_forever(until=asyncio.Event()))
    try:
        await asyncio.sleep(0)
        root.cancel()
        with pytest.raises(asyncio.CancelledError):
            await root
    finally:
        await tree.stop()
