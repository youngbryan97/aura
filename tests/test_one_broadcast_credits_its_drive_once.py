"""A broadcast that served a drive replenishes it once, not on every turn after.

The broadcast consumer records the drive a winner served and replaces the record
only when a later winner serves some drive. The motivation phase credited the
record on every tick and never took it, so on seed 7 one early win for growth,
priority 0.9105007597813606, raised growth by 0.015 a turn for 300 rounds while
the winner served no drive at all. Growth passed curiosity near turn 2,030 and
her dominant drive changed for no reason in her.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.phases.motivation_update import MotivationUpdatePhase

pytestmark = pytest.mark.unit


def _motivation() -> SimpleNamespace:
    return SimpleNamespace(budgets={"growth": {"level": 50.0, "capacity": 100.0, "decay": 0.0}})


def test_the_reading_is_taken_when_it_is_credited(monkeypatch) -> None:
    workspace = SimpleNamespace(last_drive_attention={"drive": "growth", "priority": 0.9})
    monkeypatch.setattr(
        "core.runtime.service_registry.get_runtime_service",
        lambda name, default=None: workspace if name == "global_workspace" else default,
    )
    mot = _motivation()
    first = MotivationUpdatePhase._credit_attended_drive(mot, 60.0)
    second = MotivationUpdatePhase._credit_attended_drive(mot, 60.0)
    assert first == pytest.approx(0.9)
    assert second == 0.0
    assert workspace.last_drive_attention is None
    assert mot.budgets["growth"]["level"] == pytest.approx(50.9)


def test_a_new_broadcast_credits_again(monkeypatch) -> None:
    workspace = SimpleNamespace(last_drive_attention={"drive": "growth", "priority": 0.5})
    monkeypatch.setattr(
        "core.runtime.service_registry.get_runtime_service",
        lambda name, default=None: workspace if name == "global_workspace" else default,
    )
    mot = _motivation()
    MotivationUpdatePhase._credit_attended_drive(mot, 60.0)
    workspace.last_drive_attention = {"drive": "growth", "priority": 0.5}
    MotivationUpdatePhase._credit_attended_drive(mot, 60.0)
    assert mot.budgets["growth"]["level"] == pytest.approx(51.0)
