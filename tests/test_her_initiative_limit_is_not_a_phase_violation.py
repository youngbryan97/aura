"""Her working memory keeping her open initiatives to what she can carry is not the phase it happened inside.

The compaction trims her open initiatives to her cap, inside whichever phase pushed her memory over its bound. These
pin that the trim is observed and not charged to that phase, and that a phase's own undeclared write to the same field
still is. They use AffectUpdatePhase, whose contract does not declare her initiatives.

This trim was blamed on 3 November for the "ConsciousnessPhase wrote undeclared state fields:
cognition.pending_initiatives" failures of 1 November. It was not the cause: her list held at 7 that morning and never
shrank. ConsciousnessPhase writes her initiatives itself, through the rehearsal, and now declares it
(tests/test_her_rehearsal_is_her_consciousness_phases_write.py).
"""

from __future__ import annotations

import pytest

import core.phases.affect_update  # noqa: F401 - registers the phase's contract
from core.runtime.cognitive_contract import contract_for, watched_fields
from core.runtime.phase_contract_registry import ensure_contracts_loaded
from core.runtime.cognitive_provenance import begin_transformation, reset_provenance_for_test
from core.state.aura_state import AuraState

pytestmark = pytest.mark.unit

PATH = "cognition.pending_initiatives"
PHASE = "AffectUpdatePhase"


@pytest.fixture(autouse=True)
def _clean_provenance():
    # Every contract, as her running pipeline has them: the field is watched because the phases that write it
    # (initiative generation, memory consolidation, consciousness) declare it.
    ensure_contracts_loaded()
    assert PATH in watched_fields()
    reset_provenance_for_test()
    yield
    reset_provenance_for_test()


def _initiative(index: int) -> dict:
    return {"goal": f"initiative {index}", "urgency": 0.5, "source": "test"}


def test_the_phase_used_here_does_not_declare_her_initiatives():
    assert PATH not in contract_for(PHASE).writes


def test_the_trim_her_memory_makes_is_observed_and_not_the_phases_violation():
    state = AuraState.default()
    state.cognition.pending_initiatives = [_initiative(i) for i in range(30)]
    measuring = begin_transformation(PHASE, state)
    state.cognition.trim_working_memory()
    receipt = measuring.complete(state, publish_violation=False)
    assert len(state.cognition.pending_initiatives) < 30
    assert PATH in receipt.observed_writes
    assert PATH not in receipt.undeclared_writes


def test_a_phase_writing_her_initiatives_itself_is_still_charged():
    state = AuraState.default()
    state.cognition.pending_initiatives = []
    measuring = begin_transformation(PHASE, state)
    state.cognition.pending_initiatives.append(_initiative(0))
    receipt = measuring.complete(state, publish_violation=False)
    assert PATH in receipt.undeclared_writes


def test_a_trim_with_nothing_over_the_cap_excuses_nothing():
    state = AuraState.default()
    state.cognition.pending_initiatives = [_initiative(0)]
    measuring = begin_transformation(PHASE, state)
    state.cognition.trim_working_memory()
    state.cognition.pending_initiatives.append(_initiative(1))
    receipt = measuring.complete(state, publish_violation=False)
    assert PATH in receipt.undeclared_writes
