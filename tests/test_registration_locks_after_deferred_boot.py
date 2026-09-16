"""Registration is locked after the deferred initialisers land, not under them.

Live, 2026-09-15 at load average 33: boot finished at 23:49:15 and locked
the container; the deferred autonomy initialiser reached its registrations
at 23:49:49 and every one — interiority, reliability_engine,
state_authority — was refused as "Registration locked".
"""
from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

import pytest

from core.orchestrator.boot import OrchestratorBootMixin
from core.runtime import boot_profile

ROOT = Path(__file__).resolve().parent.parent


class _Orch(OrchestratorBootMixin):
    def __init__(self) -> None:
        pass


@pytest.mark.asyncio
async def test_the_lock_waits_for_the_deferred_initialisers():
    orch = _Orch()
    landed: list[str] = []

    async def slow(name, delay):
        await asyncio.sleep(delay)
        landed.append(name)

    orch._deferred_boot_inits = [
        asyncio.create_task(slow("a", 0.05), name="orchestrator.init_a"),
        asyncio.create_task(slow("b", 0.1), name="orchestrator.init_b"),
    ]
    still_running = await orch.wait_for_deferred_boot(5.0)
    assert still_running == []
    assert sorted(landed) == ["a", "b"]


@pytest.mark.asyncio
async def test_the_window_bounds_the_wait_and_names_the_stragglers(caplog):
    orch = _Orch()

    async def forever():
        await asyncio.sleep(60)

    task = asyncio.create_task(forever(), name="orchestrator.init_slow")
    orch._deferred_boot_inits = [task]
    with caplog.at_level(logging.WARNING):
        still_running = await orch.wait_for_deferred_boot(0.05)
    task.cancel()
    assert still_running == ["orchestrator.init_slow"]
    assert any("init_slow" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_no_deferred_inits_means_no_wait():
    orch = _Orch()
    assert await orch.wait_for_deferred_boot(10.0) == []


def test_the_desktop_window_is_one_number():
    gui = (ROOT / "interface/gui_actor.py").read_text(encoding="utf-8")
    assert "DESKTOP_BOOT_WINDOW_S" in gui
    assert not re.search(r"start_time\) > 90\b", gui)
    main = (ROOT / "aura_main.py").read_text(encoding="utf-8")
    lock = main.index("ServiceContainer.lock_registration()")
    assert "wait_for_deferred_boot(_remaining_desktop_boot_window_s())" in main[:lock]
    assert boot_profile.DESKTOP_BOOT_WINDOW_S == 90.0


def test_the_remaining_window_shrinks_with_the_boot_clock(monkeypatch):
    import aura_main

    class _Clock:
        def elapsed_s(self):
            return 30.0

    monkeypatch.setattr(boot_profile, "get_boot_profiler", lambda: _Clock())
    assert aura_main._remaining_desktop_boot_window_s() == pytest.approx(60.0)
