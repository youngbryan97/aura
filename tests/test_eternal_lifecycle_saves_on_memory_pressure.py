"""Under memory pressure her lifecycle saves the orchestrator's state, once a minute, through a method that exists.

LIVE 2026-10-21: above 88% memory every turn of the loop (one a second) called StateManager.snapshot(), which does
not exist; the AttributeError was recorded each second and escalated to an emergency incident.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace


def test_one_emergency_save_through_the_orchestrator(monkeypatch):
    import core.organism.eternal_lifecycle as lifecycle
    import core.runtime as runtime
    import core.runtime.shutdown_coordinator as shutdown

    turns = iter([False, False])
    monkeypatch.setattr(shutdown, "is_shutdown_requested", lambda: next(turns, True))
    monkeypatch.setattr(lifecycle.psutil, "virtual_memory", lambda: SimpleNamespace(percent=95.0))
    monkeypatch.setattr(runtime.CoreRuntime, "get_sync", staticmethod(lambda: object()))
    saved: list[str] = []

    class _Orchestrator:
        async def _save_state_async(self, reason):
            saved.append(reason)

    monkeypatch.setattr(lifecycle.ServiceContainer, "get",
                        staticmethod(lambda name, default=None: _Orchestrator() if name == "orchestrator" else default))
    faults: list = []
    monkeypatch.setattr(lifecycle, "record_degradation", lambda *a, **k: faults.append(a))
    pause = asyncio.sleep
    monkeypatch.setattr(lifecycle.asyncio, "sleep", lambda _s: pause(0))
    asyncio.run(lifecycle.eternal_lifecycle())
    assert saved == ["thermal_emergency"]  # two turns above 88%, one save
    assert not [f for f in faults if isinstance(f[1], AttributeError)]
