"""The forces on the drives, which the budgets cannot show inside a turn.

run_032 tested nine paths into deliberation and kept none. Every one was
significant — q of six in a million — and the largest moved the domain by
0.153 of its own spread against a bar of 0.3, so D finished the campaign with
no incoming edge, and causal closure, robust recurrence, cycles per domain,
re-entry and the closed core all failed on that one fact.

The couplings were there. A budget moves at a rate per second and an arm runs
for two turns, so what reaches D in that window is a rounding error on a level
that takes minutes to move. What does move inside the turn is the force: the
surprise multiplying the decay, the conversation holding the social drain, the
warmth returning a need toward rest, the workspace crediting the drive that
won it. Those were computed every turn and thrown away.
"""

from __future__ import annotations

import pytest

from core.state.aura_state import AuraState
from core.subject.state import domain_width, feature_names

pytestmark = pytest.mark.unit


def test_the_drive_domain_records_the_forces_beside_the_levels():
    names = feature_names("D")
    for force in ("pressure", "social_hold", "warmth_return", "attended_credit", "resolve_hold"):
        assert f"D.force_{force}" in names
    assert domain_width("D") == len(names)


def test_a_force_reaches_the_recorded_row_in_the_turn_it_is_applied():
    from core.subject.state import read_core_state

    state = AuraState.default()
    calm = read_core_state(state)
    state.motivation.forces = {
        "pressure": 1.8,
        "social_hold": 0.25,
        "warmth_return": 0.4,
        "attended_credit": 0.06,
        "resolve_hold": 1.0,
    }
    pressed = read_core_state(state)
    names = list(feature_names("D"))

    def at(reading, column: str) -> float:
        return float(reading.values["D"][names.index(column)])

    assert at(calm, "D.force_pressure") != at(pressed, "D.force_pressure")
    assert at(pressed, "D.force_pressure") == pytest.approx(1.8)
    assert at(pressed, "D.force_social_hold") == pytest.approx(0.25)
    assert at(pressed, "D.force_resolve_hold") == pytest.approx(1.0)


def test_surprise_from_the_world_model_moves_the_force_the_same_turn(monkeypatch):
    """The path the campaign could not find: C reaching D inside one arm."""
    from core.phases import motivation_signals
    from core.phases.motivation_update import MotivationUpdatePhase

    phase = MotivationUpdatePhase.__new__(MotivationUpdatePhase)

    def pressure_at(surprise: float) -> float:
        monkeypatch.setattr(
            motivation_signals._ReadsTheDriveSignals,
            "_world_surprise_ratio",
            staticmethod(lambda: surprise),
        )
        return 1.0 + phase._surprise_pressure(AuraState.default())

    calm = pressure_at(0.0)
    startled = pressure_at(0.9)
    assert startled > calm
    # And it is bounded: the reading runs 0 to 1, so the drives press between
    # once and twice as fast and never faster.
    assert calm >= 1.0
    assert startled <= 2.0


def test_the_warmth_that_returns_a_need_reports_what_it_returned():
    """A reading that changes something has to say how much, or it cannot be
    recorded as the force it was."""
    from core.phases.motivation_update import MotivationUpdatePhase

    state = AuraState.default()
    budgets = state.motivation.budgets
    social = budgets["social"]
    rest = float(social["level"])
    social["level"] = rest * 0.5
    returned = MotivationUpdatePhase._warmth_returns_a_drive_to_rest(state.motivation, 30.0)
    assert returned > 0.0
    assert float(social["level"]) > rest * 0.5
