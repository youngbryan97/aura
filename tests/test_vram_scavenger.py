"""tests/test_vram_scavenger.py
==================================
The desktop runtime holds a ~20GB model resident in unified memory. When Aura
sits idle — especially under memory pressure — that memory should be reclaimed
and the model transparently reloaded on the next request.

These tests lock in the lane-safe idle VRAM scavenger:
  - it NEVER unloads a busy/foreground/warming lane (safety blocker),
  - it unloads quickly under memory pressure, and only after a long idle when
    there is no pressure,
  - a fresh lane with no activity anchor is never treated as idle,
  - unload routes through reboot_worker (clean teardown, no degradation), and
  - the module-level driver honours the AURA_VRAM_SCAVENGER kill-switch.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path

import core.brain.llm.mlx_client as mc
from core.brain.llm.mlx_client import MLXLocalClient, scavenge_idle_model_vram

_TEST_MODEL_PATH = str(Path(tempfile.gettempdir()) / "aura-test-fake-model")


class _FakeProc:
    def __init__(self, *, alive: bool = True, pid: int | None = None):
        self._alive = alive
        self.pid = pid if pid is not None else os.getpid()

    def is_alive(self) -> bool:
        return self._alive

    def kill(self) -> None:
        self._alive = False

    def terminate(self) -> None:
        self._alive = False

    def join(self, timeout: float | None = None) -> None:
        return None


class _Snap:
    def __init__(self, *, warning: bool):
        self.warning = warning
        self.available_gb = 4.0 if warning else 64.0
        self.level = "warning" if warning else "normal"
        self.reason = "test"


def _make_alive_client(
    monkeypatch,
    *,
    idle_s: float,
    model_path: str = _TEST_MODEL_PATH,
    runtime_assignment=None,
) -> MLXLocalClient:
    """An MLXLocalClient that looks alive+initialized and idle for idle_s, with
    the teardown path stubbed so the test exercises only scavenge logic."""
    c = MLXLocalClient(model_path=model_path, runtime_assignment=runtime_assignment)
    c._init_done = True
    c._process = _FakeProc(alive=True)
    now = time.time()
    c._last_generation_completed_at = now - idle_s
    c._process_started_at = now - idle_s - 5.0

    rebooted: list[str] = []

    async def _fake_reboot(reason: str = "manual_reboot", mark_failed: bool = False):
        rebooted.append(reason)
        c._init_done = False
        c._process = None

    monkeypatch.setattr(c, "reboot_worker", _fake_reboot)
    c._rebooted = rebooted  # type: ignore[attr-defined]
    # Deterministic environment: no foreground owner, not shutting down.
    monkeypatch.setattr(mc, "_foreground_owner_active", lambda: False)
    monkeypatch.setattr(mc, "_runtime_shutdown_requested", lambda: False)
    return c


def test_idle_age_zero_without_anchor():
    c = MLXLocalClient(model_path=_TEST_MODEL_PATH)
    # No activity anchors set at all → never treated as idle.
    assert c.idle_age() == 0.0


def test_idle_age_tracks_latest_anchor():
    c = MLXLocalClient(model_path=_TEST_MODEL_PATH)
    now = time.time()
    c._last_generation_completed_at = now - 300.0
    c._last_progress_at = now - 10.0  # most recent activity
    assert 8.0 <= c.idle_age(now=now) <= 12.0


def test_safety_blocker_refuses_busy_lane(monkeypatch):
    c = _make_alive_client(monkeypatch, idle_s=10_000.0)
    assert c._unload_safety_blocker() is None

    c._active_generations = 1
    assert c._unload_safety_blocker() == "active_generation"
    c._active_generations = 0

    c._warmup_in_flight = True
    assert c._unload_safety_blocker() == "warming"
    c._warmup_in_flight = False

    c._current_request_started_at = time.time()
    assert c._unload_safety_blocker() == "request_in_flight"
    c._current_request_started_at = 0.0

    monkeypatch.setattr(mc, "_foreground_owner_active", lambda: True)
    assert c._unload_safety_blocker() == "foreground_active"


def test_dead_lane_reports_already_unloaded(monkeypatch):
    c = _make_alive_client(monkeypatch, idle_s=10_000.0)
    c._init_done = False  # not alive
    assert c._unload_safety_blocker() == "already_unloaded"


def test_unloads_under_pressure_after_short_idle(monkeypatch):
    c = _make_alive_client(monkeypatch, idle_s=120.0)
    monkeypatch.setattr(mc, "get_memory_pressure_snapshot", lambda: _Snap(warning=True))
    out = asyncio.run(c.maybe_unload_idle(pressure_idle_s=90.0, hard_idle_s=900.0))
    assert out["unloaded"] is True
    assert out["under_pressure"] is True
    assert c._rebooted == ["idle_vram_scavenge"]


def test_keeps_model_under_pressure_when_recently_active(monkeypatch):
    c = _make_alive_client(monkeypatch, idle_s=30.0)
    monkeypatch.setattr(mc, "get_memory_pressure_snapshot", lambda: _Snap(warning=True))
    out = asyncio.run(c.maybe_unload_idle(pressure_idle_s=90.0, hard_idle_s=900.0))
    assert out["unloaded"] is False
    assert out["reason"] == "not_idle_enough"
    assert c._rebooted == []


def test_no_pressure_waits_for_hard_idle(monkeypatch):
    # Idle past the pressure threshold but below the hard threshold, no pressure.
    c = _make_alive_client(monkeypatch, idle_s=300.0)
    monkeypatch.setattr(mc, "get_memory_pressure_snapshot", lambda: _Snap(warning=False))
    out = asyncio.run(c.maybe_unload_idle(pressure_idle_s=90.0, hard_idle_s=900.0))
    assert out["unloaded"] is False
    assert out["reason"] == "not_idle_enough"


def test_no_pressure_unloads_after_hard_idle(monkeypatch):
    c = _make_alive_client(monkeypatch, idle_s=1000.0)
    monkeypatch.setattr(mc, "get_memory_pressure_snapshot", lambda: _Snap(warning=False))
    out = asyncio.run(c.maybe_unload_idle(pressure_idle_s=90.0, hard_idle_s=900.0))
    assert out["unloaded"] is True
    assert out["under_pressure"] is False
    assert c._rebooted == ["idle_vram_scavenge"]


def test_busy_lane_never_unloads_even_when_ancient(monkeypatch):
    c = _make_alive_client(monkeypatch, idle_s=10_000.0)
    c._active_generations = 1
    monkeypatch.setattr(mc, "get_memory_pressure_snapshot", lambda: _Snap(warning=True))
    out = asyncio.run(c.maybe_unload_idle(pressure_idle_s=90.0, hard_idle_s=900.0))
    assert out["unloaded"] is False
    assert out["reason"] == "active_generation"
    assert c._rebooted == []


def test_driver_killswitch_disables(monkeypatch):
    monkeypatch.setenv("AURA_VRAM_SCAVENGER", "0")
    out = asyncio.run(scavenge_idle_model_vram())
    assert out["enabled"] is False
    assert out["unloaded"] == 0


def test_driver_iterates_clients_and_counts(monkeypatch):
    monkeypatch.delenv("AURA_VRAM_SCAVENGER", raising=False)
    idle = _make_alive_client(monkeypatch, idle_s=10_000.0)
    busy = _make_alive_client(monkeypatch, idle_s=10_000.0)
    busy._active_generations = 1
    monkeypatch.setattr(mc, "get_memory_pressure_snapshot", lambda: _Snap(warning=True))
    # _make_alive_client patches mc._foreground_owner_active per call; both False.
    monkeypatch.setattr(mc, "_CLIENTS", {"idle": idle, "busy": busy})
    out = asyncio.run(scavenge_idle_model_vram(pressure_idle_s=90.0, hard_idle_s=900.0))
    assert out["enabled"] is True
    assert out["unloaded"] == 1
    assert idle._rebooted == ["idle_vram_scavenge"]
    assert busy._rebooted == []


# ── primary-lane residency policy ────────────────────────────────────────────
# The citizenship (hard-idle) unload evicted the serving cortex during a quiet
# afternoon at 34% system RAM; the next turn paid a 120-150s cold start and
# (pre-fix) seeded the 20260708 gate-orphan cascade. Without pressure, the
# PRIMARY lane stays resident; the 90s pressure path still reclaims it the
# moment RAM matters. Small lanes keep the citizenship unload.

def _make_primary_client(monkeypatch, *, idle_s: float) -> MLXLocalClient:
    """A client the runtime recognises as the primary lane.

    Assigning `model_path` after construction does not make one: since CP941
    the lane comes from the runtime assignment the client was built with, not
    from tokens in its path, so this produced an auxiliary client and the
    primary-lane branch under test never ran.
    """
    from core.brain.llm.model_runtime_roles_for_tests import assignment_for

    cortex_path = "/models/Qwen2.5-32B-cortex-test"
    c = _make_alive_client(
        monkeypatch,
        idle_s=idle_s,
        model_path=cortex_path,
        runtime_assignment=assignment_for(cortex_path, role="cortex"),
    )
    return c


def test_primary_lane_stays_resident_without_pressure(monkeypatch):
    monkeypatch.delenv("AURA_VRAM_SCAVENGE_PRIMARY_HARD", raising=False)
    c = _make_primary_client(monkeypatch, idle_s=10_000.0)  # way past hard idle
    monkeypatch.setattr(mc, "get_memory_pressure_snapshot", lambda: _Snap(warning=False))
    out = asyncio.run(c.maybe_unload_idle(pressure_idle_s=90.0, hard_idle_s=900.0))
    assert out["unloaded"] is False
    assert out["reason"] == "primary_lane_stays_resident_without_pressure"
    assert c._rebooted == []


def test_primary_lane_still_unloads_under_real_pressure(monkeypatch):
    c = _make_primary_client(monkeypatch, idle_s=120.0)
    monkeypatch.setattr(mc, "get_memory_pressure_snapshot", lambda: _Snap(warning=True))
    out = asyncio.run(c.maybe_unload_idle(pressure_idle_s=90.0, hard_idle_s=900.0))
    assert out["unloaded"] is True
    assert out["under_pressure"] is True


def test_primary_hard_unload_restorable_by_env(monkeypatch):
    monkeypatch.setenv("AURA_VRAM_SCAVENGE_PRIMARY_HARD", "1")
    c = _make_primary_client(monkeypatch, idle_s=10_000.0)
    monkeypatch.setattr(mc, "get_memory_pressure_snapshot", lambda: _Snap(warning=False))
    out = asyncio.run(c.maybe_unload_idle(pressure_idle_s=90.0, hard_idle_s=900.0))
    assert out["unloaded"] is True


def test_deep_solver_lane_keeps_citizenship_unload(monkeypatch):
    c = _make_alive_client(monkeypatch, idle_s=10_000.0)
    c.model_path = "/models/Qwen2.5-72B-Instruct-4bit-solver"
    monkeypatch.setattr(mc, "get_memory_pressure_snapshot", lambda: _Snap(warning=False))
    out = asyncio.run(c.maybe_unload_idle(pressure_idle_s=90.0, hard_idle_s=900.0))
    assert out["unloaded"] is True, "the 40GB solver must not squat without pressure"
