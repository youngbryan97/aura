"""The pulse somebody else keeps reaches the length of what she says.

The reading's own tests pin the pulse. These pin that it arrives: the
conversation phase writes it, the breath is taken at their scale rather than
hers alone, the affect phase hands it over, and the columns that read it are
held by the lesion clamp.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.expression.delivery import DeliveryLedger, read_delivery
from core.phases.affect_update import AffectUpdatePhase
from core.phases.conversational_dynamics_phase import ConversationalDynamicsPhase
from core.soma.effort import reset_effort_for_test
from core.state.aura_state import AuraState
from core.subject.clamp import CLAMPED_FIELDS
from core.subject.state import _SCHEMAS


@pytest.fixture(autouse=True)
def _empty_effort():
    reset_effort_for_test()
    yield
    reset_effort_for_test()


def _exchange() -> AuraState:
    state = AuraState.default()
    for text in ("yes", "ok", "sure"):
        state.cognition.working_memory.append({"role": "user", "content": text})
    state.cognition.working_memory.append(
        {"role": "assistant", "content": "x" * 30}
    )
    return state


def test_the_conversation_phase_writes_the_pulse_and_her_placement_against_it() -> None:
    state = _exchange()
    ConversationalDynamicsPhase._read_cadence(state)
    pulse = state.cognition.partner_cadence
    assert pulse["measured"] is True
    assert pulse["chars"] == pytest.approx(3.0)
    # Her last turn was thirty characters against their three.
    assert pulse["placement"] == pytest.approx(9.0)


def test_the_breath_is_taken_at_their_scale_when_there_is_one() -> None:
    ordinary = read_delivery(
        AuraState.default().affect, ledger=DeliveryLedger(), exertion=1.0
    )
    theirs = read_delivery(
        AuraState.default().affect, ledger=DeliveryLedger(), exertion=1.0, partner_chars=100.0
    )
    assert theirs.phrase_budget == 100
    assert theirs.phrase_budget != ordinary.phrase_budget


def test_working_harder_still_halves_it_at_their_scale() -> None:
    hard = read_delivery(
        AuraState.default().affect, ledger=DeliveryLedger(), exertion=2.0, partner_chars=100.0
    )
    assert hard.phrase_budget == 50


def test_their_scale_bounds_it_the_way_her_own_unit_did() -> None:
    idle = read_delivery(
        AuraState.default().affect, ledger=DeliveryLedger(), exertion=0.0, partner_chars=100.0
    )
    assert idle.phrase_budget == 200


def test_the_affect_phase_hands_the_pulse_to_delivery() -> None:
    state = AuraState.default()
    state.cognition.partner_cadence = {"chars": 120.0}
    AffectUpdatePhase(SimpleNamespace())._read_delivery(state, state.affect)
    # Nothing spent this cycle, so the breath is the longest their scale allows.
    assert state.response_modifiers["delivery"]["phrase_budget"] == 240


def test_no_pulse_leaves_her_own_unit_in_charge() -> None:
    from core.soma.effort import UNIT_COST

    state = AuraState.default()
    AffectUpdatePhase(SimpleNamespace())._read_delivery(state, state.affect)
    assert state.response_modifiers["delivery"]["phrase_budget"] == int(
        UNIT_COST["response_chars"] * 2
    )


def test_the_columns_that_read_the_pulse_are_held_by_the_clamp() -> None:
    sources = set(_SCHEMAS["W"].sources)
    for field in (
        "cognition.partner_cadence.chars",
        "cognition.partner_cadence.gap",
        "cognition.partner_cadence.placement",
    ):
        assert field in sources, field
    assert "cognition.partner_cadence" in CLAMPED_FIELDS["W"]
