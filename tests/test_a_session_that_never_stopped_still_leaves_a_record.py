"""Continuity across a session that is still running, and one that died.

Two defects this covers.

A consumer that guards its load with `if _record is None` never stops asking
when there is no record to find, so a run with no prior session re-read the
directory and re-announced the first awakening on every cycle. One campaign
logged it 7,886 times.

And the record was written only on graceful shutdown, although the method that
writes it has said since it was written that it should also be written
periodically. A session that crashed, or one that simply never stopped, left
nothing behind at all.
"""

from __future__ import annotations

import logging

import pytest

from core.continuity import ContinuityEngine


def test_a_missing_record_is_answered_once(monkeypatch, tmp_path, caplog) -> None:
    """The look happens once; the answer is remembered, absence included."""
    path = tmp_path / "continuity.json"
    looks = {"count": 0}

    def _path() -> object:
        looks["count"] += 1
        return path

    monkeypatch.setattr("core.continuity._get_continuity_path", _path)
    engine = ContinuityEngine()
    with caplog.at_level(logging.INFO, logger="core.continuity"):
        for _ in range(50):
            assert engine.load() is None
    assert looks["count"] == 1, "the directory was read more than once"
    announcements = [r for r in caplog.records if "First awakening" in r.getMessage()]
    assert len(announcements) == 1, f"announced {len(announcements)} times"


def test_a_forced_load_looks_again(monkeypatch, tmp_path) -> None:
    """The caller that has reason to think the file changed can still ask."""
    path = tmp_path / "continuity.json"
    looks = {"count": 0}

    def _path() -> object:
        looks["count"] += 1
        return path

    monkeypatch.setattr("core.continuity._get_continuity_path", _path)
    engine = ContinuityEngine()
    engine.load()
    engine.load(force=True)
    assert looks["count"] == 2


def test_a_checkpoint_is_not_a_clean_shutdown() -> None:
    """A record written mid-session says the session was alive, not that it stopped."""
    from core.soma.source_body import SourceBodyAwareness as SourceBody

    assert "checkpoint" not in SourceBody._CLEAN_SHUTDOWN_REASONS
    assert "graceful" in SourceBody._CLEAN_SHUTDOWN_REASONS


@pytest.mark.asyncio
async def test_the_metabolic_cycle_checkpoints_continuity(monkeypatch) -> None:
    """The periodic save the contract named is actually called, and rate limited."""
    from core.coordinators.metabolic_coordinator import MetabolicCoordinator

    saved: list[str] = []

    class _Engine:
        def save(self, reason: str = "graceful", **_: object) -> None:
            saved.append(reason)

    monkeypatch.setattr("core.continuity.get_continuity", lambda: _Engine())
    coordinator = MetabolicCoordinator.__new__(MetabolicCoordinator)
    await coordinator._continuity_checkpoint()
    assert saved == ["checkpoint"], "the first cycle left no record"
    await coordinator._continuity_checkpoint()
    assert saved == ["checkpoint"], "it wrote again inside its own interval"


@pytest.mark.asyncio
async def test_a_failed_checkpoint_is_recorded_not_swallowed(monkeypatch) -> None:
    from core.coordinators.metabolic_coordinator import MetabolicCoordinator

    noted: list[str] = []

    class _Engine:
        def save(self, **_: object) -> None:
            raise RuntimeError("no disk")

    monkeypatch.setattr("core.continuity.get_continuity", lambda: _Engine())
    monkeypatch.setattr(
        "core.coordinators.metabolic_coordinator.record_degradation",
        lambda subsystem, exc, **kw: noted.append(subsystem),
    )
    coordinator = MetabolicCoordinator.__new__(MetabolicCoordinator)
    await coordinator._continuity_checkpoint()
    assert noted == ["continuity"]
