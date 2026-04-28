"""Tests for the degradation receipt system.

Verifies that:
  - record_degradation produces structured records
  - Severity levels are logged correctly
  - DegradationTracker counts and stores records
  - SubsystemHealth state machine works
  - SubsystemRegistry tracks multiple subsystems
"""
from __future__ import annotations

import pytest

from core.runtime.errors import (
    DegradationTracker,
    SubsystemHealth,
    SubsystemRegistry,
    record_degradation,
    get_degradation_tracker,
)


@pytest.fixture(autouse=True)
def clean_tracker():
    tracker = get_degradation_tracker()
    tracker.reset()
    yield
    tracker.reset()


def test_record_degradation_creates_record():
    exc = ValueError("test error")
    record = record_degradation(
        subsystem="phi_core",
        error=exc,
        severity="degraded",
        action="fell back to default phi",
    )
    assert record.subsystem == "phi_core"
    assert record.severity == "degraded"
    assert record.error_type == "ValueError"
    assert "test error" in record.error_message
    assert record.action == "fell back to default phi"
    assert record.timestamp > 0


def test_tracker_counts():
    tracker = get_degradation_tracker()
    record_degradation("memory_facade", ValueError("a"), "warning", "retry")
    record_degradation("memory_facade", ValueError("b"), "degraded", "cache fallback")
    record_degradation("phi_core", RuntimeError("c"), "critical", "disabled")

    assert tracker.count("memory_facade") == 2
    assert tracker.count("memory_facade", "warning") == 1
    assert tracker.count("memory_facade", "degraded") == 1
    assert tracker.count("phi_core", "critical") == 1


def test_tracker_recent():
    tracker = get_degradation_tracker()
    for i in range(10):
        record_degradation(f"sub_{i}", ValueError(f"err {i}"), "debug", "logged")

    recent = tracker.recent(limit=5)
    assert len(recent) == 5


def test_tracker_status():
    tracker = get_degradation_tracker()
    record_degradation("test_sub", ValueError("err"), "warning", "logged")
    status = tracker.status()
    assert status["total_degradations"] == 1
    assert "test_sub" in status["counts_by_subsystem"]


def test_subsystem_health_lifecycle():
    h = SubsystemHealth(name="phi_core")
    assert h.status == "healthy"

    h.mark_degraded("high latency", impact="slower inference")
    assert h.status == "degraded"
    assert h.reason == "high latency"
    assert h.impact == "slower inference"
    assert h.last_failed_at > 0

    h.mark_ok()
    assert h.status == "healthy"
    assert h.last_ok_at > 0


def test_subsystem_health_unavailable():
    h = SubsystemHealth(name="vector_db")
    h.mark_unavailable("connection refused")
    assert h.status == "unavailable"
    assert h.reason == "connection refused"


def test_subsystem_registry():
    reg = SubsystemRegistry()
    h1 = reg.register("memory")
    h2 = reg.register("phi_core")

    assert not reg.any_critical()

    h1.mark_unavailable("disk full")
    assert reg.any_critical()

    all_status = reg.all_status()
    assert "memory" in all_status
    assert "phi_core" in all_status
    assert all_status["memory"]["status"] == "unavailable"
    assert all_status["phi_core"]["status"] == "healthy"


def test_no_silent_pass():
    """record_degradation must never silently pass — it must produce a record."""
    exc = RuntimeError("something broke")
    record = record_degradation(
        subsystem="test",
        error=exc,
        severity="debug",
        action="no recovery",
    )
    assert record is not None
    assert record.error_message != ""
    assert record.action != ""
