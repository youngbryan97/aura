"""The Brainstem background-cognition gate must be MEETABLE on a desktop that
is already holding the ~16-20GB 32B Cortex.

Regression guard for the 2026-07 respawn loop: the old default (defer unless
pressure <48% AND >=34GB free) could never admit on a box whose steady state is
~56% / ~28GB available, so background cognition never ran, mind_tick never
completed a successful tick, and the launcher respawned a duplicate 32B.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.brain.llm_health_router import BRAINSTEM_ENDPOINT, EndpointHealth, HealthAwareLLMRouter


@dataclass
class _Snap:
    pressure_pct: float
    available_gb: float
    process_rss_gb: float = 20.0
    process_rss_limit_gb: float = 42.0


def _brainstem_ep() -> EndpointHealth:
    return EndpointHealth(name=BRAINSTEM_ENDPOINT, url="local://brainstem", model="qwen-7b")


def _reason(monkeypatch, snap: _Snap) -> str | None:
    monkeypatch.setenv("AURA_DESKTOP_RESOURCE_GUARD", "1")
    # thresholds unset -> exercise the shipped defaults
    monkeypatch.delenv("AURA_BACKGROUND_BRAINSTEM_MAX_PRESSURE_PCT", raising=False)
    monkeypatch.delenv("AURA_BACKGROUND_BRAINSTEM_MIN_AVAILABLE_GB", raising=False)
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot", lambda: snap
    )
    monkeypatch.setattr(
        "core.utils.memory_monitor.kernel_memory_pressure_level", lambda: "normal"
    )
    return HealthAwareLLMRouter._desktop_background_endpoint_deferral_reason(_brainstem_ep())


def test_admits_at_desktop_steady_state_with_cortex_loaded(monkeypatch):
    # ~56% pressure / ~28GB available: the real steady state with the 32B resident.
    assert _reason(monkeypatch, _Snap(pressure_pct=56.0, available_gb=28.0)) is None


def test_still_defers_when_memory_is_genuinely_tight(monkeypatch):
    # Genuinely low headroom must still defer (the OOM guard is intact).
    assert _reason(monkeypatch, _Snap(pressure_pct=75.0, available_gb=15.0)) is not None


def test_defers_when_available_below_floor(monkeypatch):
    # Kernel-normal cannot erase the absolute 22GB allocation floor.
    assert _reason(monkeypatch, _Snap(pressure_pct=50.0, available_gb=18.0)) is not None


def test_process_rss_near_limit_still_defers(monkeypatch):
    # Even with nominal pressure, an RSS within 6GB of the limit defers.
    r = _reason(monkeypatch, _Snap(pressure_pct=40.0, available_gb=30.0, process_rss_gb=37.0, process_rss_limit_gb=42.0))
    assert r is not None


def test_guard_disabled_never_defers(monkeypatch):
    monkeypatch.setenv("AURA_DESKTOP_RESOURCE_GUARD", "0")
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: _Snap(pressure_pct=95.0, available_gb=2.0),
    )
    assert HealthAwareLLMRouter._desktop_background_endpoint_deferral_reason(_brainstem_ep()) is None
