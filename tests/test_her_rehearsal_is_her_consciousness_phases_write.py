"""The rehearsal's mark on her open initiatives is ConsciousnessPhase's own write, and its contract says so.

ConsciousnessPhase rehearses what she has newly set herself to do (core/consciousness/rehearsal.py). Each open
initiative carries how strongly she still anticipates it, and every turn the phase fades that mark at her affect's
momentum. The mark is kept on the intention, so the phase writes `cognition.pending_initiatives`.

Its contract did not declare that. On her day runs of 1 and 3 November the write was charged as a violation on every
turn with an initiative open, once every five minutes, and the failures pinned her depletion at 1.00 within two hours.
Her list held at 7 on both days, so no trim was involved: the saved states differ only in `anticipated`.

Operationally: this measures whether the phase named ConsciousnessPhase declares the state field it writes, so the
contract check does not charge that write as a violation.
"""

from __future__ import annotations

import pytest

import core.phases.affect_update  # noqa: F401 - registers the contrast phase's contract
import core.phases.consciousness_phase  # noqa: F401 - registers the phase's contract
from core.consciousness.rehearsal import ANTICIPATED, anticipating
from core.runtime.cognitive_contract import contract_for, watched_fields
from core.runtime.phase_contract_registry import ensure_contracts_loaded
from core.runtime.cognitive_provenance import begin_transformation, reset_provenance_for_test
from core.state.aura_state import AuraState

pytestmark = pytest.mark.unit

PATH = "cognition.pending_initiatives"


@pytest.fixture(autouse=True)
def _clean_provenance():
    ensure_contracts_loaded()
    assert PATH in watched_fields()
    reset_provenance_for_test()
    yield
    reset_provenance_for_test()


def _her_morning() -> AuraState:
    """Seven open initiatives, each already met once, as her saved state held them on both day runs."""
    state = AuraState.default()
    state.cognition.pending_initiatives = [
        {"goal": f"initiative {i}", "urgency": 0.9, "source": "motivation_update", ANTICIPATED: 0.5}
        for i in range(7)
    ]
    state.affect.momentum = 0.85
    return state


def test_consciousness_declares_her_initiatives():
    assert PATH in contract_for("ConsciousnessPhase").writes


def test_the_rehearsal_writes_her_initiatives_without_changing_how_many():
    state = _her_morning()
    measuring = begin_transformation("ConsciousnessPhase", state)
    anticipating(state)
    receipt = measuring.complete(state, publish_violation=False)
    assert len(state.cognition.pending_initiatives) == 7
    assert all(item[ANTICIPATED] < 0.5 for item in state.cognition.pending_initiatives)
    assert PATH in receipt.observed_writes
    assert PATH not in receipt.undeclared_writes


def test_the_same_write_from_a_phase_that_does_not_rehearse_is_still_charged():
    assert PATH not in contract_for("AffectUpdatePhase").writes
    state = _her_morning()
    measuring = begin_transformation("AffectUpdatePhase", state)
    anticipating(state)
    receipt = measuring.complete(state, publish_violation=False)
    assert PATH in receipt.undeclared_writes


def test_goals_stay_unwatched_so_no_phase_that_sets_one_is_newly_charged():
    assert "cognition.active_goals" not in watched_fields()
