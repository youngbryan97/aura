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
    # Half of growth is unmet at 50 of 100, so half the priority is credited.
    assert first == pytest.approx(0.45)
    assert second == 0.0
    assert workspace.last_drive_attention is None
    assert mot.budgets["growth"]["level"] == pytest.approx(50.45)


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
    assert mot.budgets["growth"]["level"] == pytest.approx(50.0 + 0.25 + 0.5 * (1.0 - 50.25 / 100.0))


def test_a_need_nearly_met_is_satisfied_less(monkeypatch) -> None:
    """The credit shrinks as the drive nears its set point, so it approaches it and does not run past."""
    workspace = SimpleNamespace(last_drive_attention=None)
    monkeypatch.setattr(
        "core.runtime.service_registry.get_runtime_service",
        lambda name, default=None: workspace if name == "global_workspace" else default,
    )
    low = SimpleNamespace(budgets={"growth": {"level": 20.0, "capacity": 100.0}})
    high = SimpleNamespace(budgets={"growth": {"level": 90.0, "capacity": 100.0}})
    gains = []
    for mot in (low, high):
        workspace.last_drive_attention = {"drive": "growth", "priority": 1.0}
        gains.append(MotivationUpdatePhase._credit_attended_drive(mot, 60.0))
    assert gains[0] == pytest.approx(0.8)
    assert gains[1] == pytest.approx(0.1)
    level = {"growth": {"level": 50.0, "capacity": 100.0}}
    mot = SimpleNamespace(budgets=level)
    for _ in range(2400):
        workspace.last_drive_attention = {"drive": "growth", "priority": 0.75}
        MotivationUpdatePhase._credit_attended_drive(mot, 1.0)
    assert 60.0 < level["growth"]["level"] < 70.0, "a run of wins brings growth towards its set point, not past curiosity"
