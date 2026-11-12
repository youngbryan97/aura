"""A synchronous reader never calls a coroutine function and drops what it returns.

LIVE, her log to 12 November 2026: 1,569 "coroutine 'MotivationEngine.get_status' was never awaited" from her
aesthetic engine, which therefore never read a drive, and 620 from the health probe autopoiesis was handed for the
same engine.
"""
from __future__ import annotations

import gc
import warnings

import core.adaptation.autonomous_resilience as resilience
from core.adaptation.autonomous_resilience import IntegrationAuditor
from core.container import ServiceContainer
from core.creativity.aesthetic_engine import AestheticEngine, _wants


class _AsyncStatusDrives:
    """Her drive engine's shape: an async get_status, synchronous levels."""

    boredom_level = 0.8

    async def get_status(self):
        return {"curiosity": {"level": 10.0}}

    def get_drive_vector(self):
        return {"energy": 0.9, "curiosity": 0.2, "social": 0.7}


def _no_unawaited_coroutine(action):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = action()
        gc.collect()
    assert not [w for w in caught if "never awaited" in str(w.message)]
    return result


def test_a_health_probe_does_not_call_an_async_status():
    resilience._reported_uninstrumented.clear()
    probe = IntegrationAuditor._health_probe_for(_AsyncStatusDrives())
    assert probe is not None
    assert _no_unawaited_coroutine(probe) == resilience._UNKNOWN_HEALTH_SCORE
    assert "coroutine" in resilience._reported_uninstrumented["_AsyncStatusDrives"]


def test_a_health_probe_reads_a_synchronous_reading_beside_an_async_one():
    class _Both(_AsyncStatusDrives):
        def is_healthy(self):
            return True

    probe = IntegrationAuditor._health_probe_for(_Both())
    assert _no_unawaited_coroutine(probe) == 1.0


def test_a_status_that_returns_a_coroutine_is_closed_not_dropped():
    class _Wrapped:
        def get_status(self):
            return _AsyncStatusDrives().get_status()

    resilience._reported_uninstrumented.clear()
    probe = IntegrationAuditor._health_probe_for(_Wrapped())
    assert _no_unawaited_coroutine(probe) == resilience._UNKNOWN_HEALTH_SCORE


def test_what_a_drive_wants_is_what_its_budget_lacks():
    wants = _wants({"energy": 0.9, "curiosity": 0.2, "social": 0.7}, 0.8)
    assert wants == {"energy": 0.9, "curiosity": 0.8, "social": 0.30000000000000004, "boredom": 0.8}


def test_the_aesthetic_engine_reads_her_drives(monkeypatch, tmp_path):
    drives = _AsyncStatusDrives()
    monkeypatch.setattr(
        ServiceContainer, "get",
        staticmethod(lambda name, default=None, **_: drives if name == "drive_engine" else default),
    )
    engine = AestheticEngine(journal_path=tmp_path / "journal.jsonl")
    _no_unawaited_coroutine(engine._refresh_state)
    assert engine._drives["curiosity"] == 0.8
    assert engine._drives["energy"] == 0.9
    # Energy is a resource, not a theme; the strongest themed want leads.
    assert engine._dominant_drive() in {"curiosity", "boredom"}
