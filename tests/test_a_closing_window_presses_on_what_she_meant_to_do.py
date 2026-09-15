"""The closing window, wired: fed by each message, read by the action domain, used by the will.

The ledger is pinned in test_the_end_of_a_sitting_she_can_see_coming.py. These
pin the wiring: every message from the person here goes into their sittings,
the reading sits where the action domain reads it and the clamp holds it, and
a measured window moves intentions anchored to a user's request, composes with
channelled arousal without either climbing, and lets go when the window does.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import core.affect.nociception as nociception
import core.social.closing_window as closing_window
from core.phases.motivation_update import MotivationUpdatePhase
from core.state.aura_state import AuraState


@pytest.fixture(autouse=True)
def fresh_ledger():
    closing_window.reset_for_test()
    yield
    closing_window.reset_for_test()


@pytest.fixture
def undamaged(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        nociception, "get_nociception_engine", lambda: SimpleNamespace(nociceptive_pressure=lambda: 0.0)
    )


def _state(closing: float, *, measured: bool = True) -> AuraState:
    state = AuraState.default()
    state.cognition.closing_window = {"closing": closing, "measured": measured}
    state.cognition.pending_initiatives = [
        {"goal": "send them the draft", "urgency": 0.4, "origin": "user"},
        {"goal": "tidy the index", "urgency": 0.4, "origin": "mind_tick_fallback"},
    ]
    return state


def _urgencies(state: AuraState) -> list[float]:
    return [item["urgency"] for item in state.cognition.pending_initiatives]


def test_a_closing_window_moves_only_what_was_asked_for() -> None:
    state = _state(0.5)
    assert MotivationUpdatePhase._closing(state) == 1
    assert _urgencies(state) == pytest.approx([0.4 + 0.5 * 0.6, 0.4], abs=1e-4)


def test_an_unmeasured_window_moves_nothing() -> None:
    state = _state(0.9, measured=False)
    assert MotivationUpdatePhase._closing(state) == 0
    assert _urgencies(state) == [0.4, 0.4]


def test_the_lift_goes_when_the_window_opens_again() -> None:
    state = _state(0.5)
    MotivationUpdatePhase._closing(state)
    state.cognition.closing_window = {"closing": 0.0, "measured": True}
    MotivationUpdatePhase._closing(state)
    assert _urgencies(state) == pytest.approx([0.4, 0.4], abs=1e-4)
    assert "window_lift" not in state.cognition.pending_initiatives[0]


def test_it_composes_with_channelled_arousal_without_climbing(undamaged) -> None:
    state = _state(0.5)
    state.affect.delivery_z = 3.0
    state.affect.breakthrough = True
    state.affect.valence = -0.4
    first = None
    for _ in range(5):
        MotivationUpdatePhase._closing(state)
        MotivationUpdatePhase._channelled(state)
        if first is None:
            first = _urgencies(state)[0]
    assert _urgencies(state)[0] == pytest.approx(first, abs=1e-4)
    assert _urgencies(state)[0] <= 1.0
    window = 0.5 * 0.6
    assert first == pytest.approx(0.4 + window + 0.75 * (1.0 - 0.4 - window), abs=1e-4)


def test_each_message_from_the_person_here_goes_into_their_sittings() -> None:
    from core.phases.conversational_dynamics_phase import ConversationalDynamicsPhase

    state = AuraState.default()
    state.cognition.current_partner = "sam"
    ConversationalDynamicsPhase._note_they_came_back(state)
    ConversationalDynamicsPhase._note_they_came_back(state)
    assert state.cognition.closing_window["messages_so_far"] == 2
    assert closing_window.get_sitting_ledger().reading("sam").messages_so_far == 2


def test_the_action_domain_reads_it_and_the_clamp_holds_it() -> None:
    from core.subject.clamp import CLAMPED_FIELDS
    from core.subject.state import feature_names

    assert "D.closing_window" in feature_names("D")
    assert "cognition.closing_window" in CLAMPED_FIELDS["D"]
