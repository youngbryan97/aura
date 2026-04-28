"""Tests for Executive Closure hysteresis — ObjectiveCommitment lifecycle.

Verifies that:
  - User-anchored tasks create commitments
  - Minor pressures are dampened during commitment
  - Critical integrity failures bypass hysteresis
  - Task completion releases commitment
"""
from __future__ import annotations

import time

import pytest

from core.consciousness.executive_closure import (
    ObjectiveCommitment,
    ExecutiveClosureEngine,
)


def test_commitment_lifecycle():
    """ObjectiveCommitment tracks active/completed/expired state."""
    now = time.time()
    c = ObjectiveCommitment(
        objective="help user debug code",
        origin="user",
        started_at=now,
        last_confirmed_at=now,
        min_hold_s=60.0,
        max_hold_s=300.0,
    )

    # Active right away
    assert c.active(now)
    assert c.age_s(now) == 0.0

    # Still active after 100s
    assert c.active(now + 100.0)
    assert abs(c.age_s(now + 100.0) - 100.0) < 0.01

    # Expired after max_hold_s
    assert not c.active(now + 301.0)

    # Completed → no longer active
    c.completed = True
    assert not c.active(now)


def test_hysteresis_dampens_minor_pressures():
    """_apply_executive_hysteresis should cap non-critical pressures."""
    engine = ExecutiveClosureEngine()
    now = time.time()
    commitment = ObjectiveCommitment(
        objective="write tests",
        origin="user",
        started_at=now,
        last_confirmed_at=now,
    )

    pressures = {
        "stability": 0.85,
        "integrity": 0.30,
        "curiosity": 0.65,
        "social": 0.55,
        "growth": 0.50,
    }

    adjusted = engine._apply_executive_hysteresis(pressures, commitment=commitment)

    # Stability should be capped at 0.72
    assert adjusted["stability"] <= 0.72
    # Curiosity capped at 0.35
    assert adjusted["curiosity"] <= 0.35
    # Social capped at 0.40
    assert adjusted["social"] <= 0.40
    # Growth capped at 0.35
    assert adjusted["growth"] <= 0.35
    # Integrity preserved (not capped)
    assert adjusted["integrity"] == pressures["integrity"]


def test_critical_interrupt_bypasses_hysteresis():
    """Critical vitality failure should produce an interrupt reason."""
    engine = ExecutiveClosureEngine()

    reason = engine._critical_interrupt_reason(
        pressures={"stability": 0.5, "integrity": 0.95},
        homeostasis_status={"will_to_live": 0.15, "integrity": 1.0, "sovereignty": 1.0},
        closed_loop_status={"free_energy": 0.3},
    )
    assert reason == "critical_vitality"


def test_no_interrupt_under_normal_conditions():
    """Normal pressures should not produce an interrupt."""
    engine = ExecutiveClosureEngine()

    reason = engine._critical_interrupt_reason(
        pressures={"stability": 0.4, "integrity": 0.3},
        homeostasis_status={"will_to_live": 0.8, "integrity": 0.9, "sovereignty": 0.85},
        closed_loop_status={"free_energy": 0.2},
    )
    assert reason == ""


def test_integrity_interrupt():
    """Low integrity should trigger interrupt."""
    engine = ExecutiveClosureEngine()

    reason = engine._critical_interrupt_reason(
        pressures={"stability": 0.5, "integrity": 0.5},
        homeostasis_status={"will_to_live": 0.6, "integrity": 0.25, "sovereignty": 0.8},
        closed_loop_status={"free_energy": 0.3},
    )
    assert reason == "critical_integrity"


def test_user_task_detection():
    """_is_user_task should return True for actionable objectives."""
    engine = ExecutiveClosureEngine()

    # Actionable user task
    assert engine._is_user_task("help me write a function", "user")
    # Intrinsic self-goal should not count
    assert not engine._is_user_task("", "user")
