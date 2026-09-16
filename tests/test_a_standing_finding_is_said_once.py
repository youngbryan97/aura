"""The integrity monitor says a finding when it appears and when it clears.

230 lines of "Thermal pressure is fair" at warning in one session, one every
five minutes, for a condition that never changed.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

from core.resilience.integrity_monitor import SystemIntegrityMonitor


def _monitor() -> SystemIntegrityMonitor:
    monitor = SystemIntegrityMonitor.__new__(SystemIntegrityMonitor)
    monitor._check_count = 0
    monitor._last_findings = ((), ())
    return monitor


def _levels(caplog):
    return [(r.levelno, r.getMessage()) for r in caplog.records if "integrity" in r.getMessage().lower()]


def test_the_same_warning_is_warned_once_then_noted(caplog):
    monitor = _monitor()
    report = SimpleNamespace(errors=[], warnings=["Thermal pressure is fair"])
    with caplog.at_level(logging.INFO, logger="Aura.IntegrityMonitor"):
        monitor._report_findings(report)
        monitor._report_findings(report)
        monitor._report_findings(report)
    levels = [lv for lv, _ in _levels(caplog)]
    assert levels == [logging.WARNING, logging.INFO, logging.INFO]


def test_a_change_in_the_set_is_an_event_and_so_is_the_clear(caplog):
    monitor = _monitor()
    with caplog.at_level(logging.INFO, logger="Aura.IntegrityMonitor"):
        monitor._report_findings(SimpleNamespace(errors=[], warnings=["Thermal pressure is fair"]))
        monitor._report_findings(
            SimpleNamespace(errors=[], warnings=["Thermal pressure is fair", "High memory usage: 9000MB"])
        )
        monitor._report_findings(SimpleNamespace(errors=[], warnings=[]))
    entries = _levels(caplog)
    assert [lv for lv, _ in entries] == [logging.WARNING, logging.WARNING, logging.INFO, logging.INFO]
    assert "cleared" in entries[2][1]
    assert "passed" in entries[3][1]


def test_errors_follow_the_same_rule(caplog):
    monitor = _monitor()
    with caplog.at_level(logging.INFO, logger="Aura.IntegrityMonitor"):
        monitor._report_findings(SimpleNamespace(errors=["CRITICAL thermal pressure: level 2"], warnings=[]))
        monitor._report_findings(SimpleNamespace(errors=["CRITICAL thermal pressure: level 2"], warnings=[]))
        monitor._report_findings(SimpleNamespace(errors=[], warnings=["Thermal pressure is fair"]))
    entries = _levels(caplog)
    assert [lv for lv, _ in entries] == [logging.ERROR, logging.INFO, logging.INFO, logging.WARNING]
    assert "errors cleared" in entries[2][1]
