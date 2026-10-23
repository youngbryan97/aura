"""How far she has grown is changed by a phase whose receipt says so.

Her lineage judges a step by whether every person path it changed was written by
a measured transformation. `identity.evolution_score` is written by the affect
phase from the growth ledger, and the phase's contract did not declare it, so no
receipt watched it; and the quick chat lane ran the phase with no receipt at all.
On 23 October her live line read thirty-three of forty-three steps as changes
nothing measured accounted for.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import core.phases.affect_update  # noqa: F401 - registers the phase's contract
from core.runtime.cognitive_contract import contract_for
from core.runtime.cognitive_provenance import last_writer_since


def test_the_affect_phase_declares_how_far_she_has_grown():
    contract = contract_for("AffectUpdatePhase")
    assert contract is not None
    assert "identity.evolution_score" in contract.writes


class AffectUpdatePhase:
    """Stands in for the phase under its own name, so its contract applies."""

    async def execute(self, state, objective=""):
        state.identity.evolution_score = float(state.identity.evolution_score) + 0.125
        return state


def test_the_quick_lane_leaves_a_receipt_for_what_it_writes():
    from core.brain.cognitive_engine_thinking_loop import _RunsTheThinkingLoop
    from core.state.aura_state import AuraState

    engine = SimpleNamespace(_phases=[AffectUpdatePhase()])
    state = AuraState.default()
    started = time.time()
    asyncio.run(_RunsTheThinkingLoop._feel_the_quick_turn(engine, state, "hello"))
    held = last_writer_since("identity.evolution_score", started)
    assert held is not None
    receipt, _tick, _objective = held
    assert receipt.transform == "AffectUpdatePhase"
