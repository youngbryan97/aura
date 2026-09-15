"""Romeo and Juliet, wired.

The measures are pinned in test_being_made_minor_in_their_account_of_a_shared_past.py
and test_an_absence_put_down_to_timing_is_feared_less.py. These pin the wiring:
the dynamics phase compares the two accounts only when recall marked the past
as shared, marks each message with the strain the other-agent model reads, the
affect phase floors sadness at being made minor, and the world domain reads the
column the clamp holds.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import core.social.closing_window as closing_window
import core.social.made_minor as made_minor
from core.phases.conversational_dynamics_phase import ConversationalDynamicsPhase
from core.state.aura_state import AuraState


@pytest.fixture(autouse=True)
def fresh_ledgers():
    made_minor.reset_for_test()
    closing_window.reset_for_test()
    yield
    made_minor.reset_for_test()
    closing_window.reset_for_test()


def _exchange(state: AuraState, *, hers: str, theirs: str, shared: bool) -> None:
    state.cognition.working_memory = [
        {"role": "assistant", "content": hers},
        {"role": "user", "content": theirs},
    ]
    state.cognition.relived = {"shared": shared}
    ConversationalDynamicsPhase._read_made_minor(state)


def test_the_two_accounts_are_compared_only_for_a_shared_past() -> None:
    state = AuraState.default()
    _exchange(state, hers="I planned that whole trip and I loved it", theirs="I booked it and I drove", shared=False)
    assert state.cognition.made_minor["shared_recalls"] == 0
    _exchange(state, hers="I planned that whole trip and I loved it", theirs="I booked it and you found the cabin", shared=True)
    assert state.cognition.made_minor["shared_recalls"] == 1


def test_a_message_carries_the_strain_the_other_agent_model_reads(monkeypatch) -> None:
    import core.social.other_agent_model as other

    monkeypatch.setattr(
        other,
        "get_other_agent_model",
        lambda: SimpleNamespace(estimate=lambda _agent: SimpleNamespace(abstained=False, affect={"frustration": 0.7})),
    )
    state = AuraState.default()
    state.cognition.current_partner = "sam"
    ConversationalDynamicsPhase._note_they_came_back(state)
    ledger = closing_window.get_sitting_ledger()
    assert ledger._strain["sam"] == [pytest.approx(0.7)]


def test_being_made_minor_is_felt_as_sadness() -> None:
    from core.phases.affect_readings import AffectReadings

    state = AuraState.default()
    state.cognition.made_minor = {"minor": 0.6, "measured": True}
    readings = AffectReadings.__new__(AffectReadings)
    readings.made_minor(state, state.affect)
    assert state.affect.emotions.get("sadness", 0.0) >= 0.6


def test_the_world_domain_reads_it_and_the_clamp_holds_it() -> None:
    from core.subject.clamp import CLAMPED_FIELDS
    from core.subject.state import feature_names

    assert "W.made_minor" in feature_names("W")
    assert "cognition.made_minor" in CLAMPED_FIELDS["W"]
