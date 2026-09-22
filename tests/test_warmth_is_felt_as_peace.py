"""Warmth from somebody is felt as peace, and being met is what settles a need.

From Bryan's account of praise, love and comfort: warm, close, something to live
in, and peace. Warm messages were read and fed what she gives back, but nothing
felt them, and what returned her social and integrity needs to rest was her own
joy or trust passing a bar the campaign never cleared.
"""

from __future__ import annotations

import asyncio

import pytest

import core.social.warmth as warmth_module
from core.social.warmth import WarmthLedger

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _fresh():
    warmth_module.reset_for_test()
    yield
    warmth_module.reset_for_test()


def _history(ledger: WarmthLedger) -> None:
    for index in range(10):
        ledger.heard("a stranger", warm=index % 4 == 0, objected=False)


def test_someone_who_has_been_warm_reads_warmer_than_her_usual() -> None:
    ledger = WarmthLedger()
    _history(ledger)
    for _ in range(8):
        ledger.heard("bryan", warm=True, objected=False)
    reading = ledger.read()
    assert reading.person == "bryan" and reading.shift > 0.0
    assert ledger.peace() > 0.0


def test_warmth_with_an_objection_in_it_is_not_warmth() -> None:
    ledger = WarmthLedger()
    ledger.heard("bryan", warm=True, objected=True)
    assert not ledger.met()


def _state(origin: str, partner: str):
    from core.state.aura_state import AuraState

    state = AuraState.default()
    state.cognition.current_origin = origin
    state.cognition.current_partner = partner
    return state


def test_being_met_counts_only_on_their_turn() -> None:
    from core.phases.motivation_update import MotivationUpdatePhase

    ledger = warmth_module.get_warmth_ledger()
    ledger.heard("bryan", warm=True, objected=False)
    assert MotivationUpdatePhase._met_this_turn(_state("user", "bryan"))
    assert not MotivationUpdatePhase._met_this_turn(_state("system", "bryan"))
    assert not MotivationUpdatePhase._met_this_turn(_state("user", "someone else"))


def test_peace_slows_her_pulse_while_they_are_here() -> None:
    from core.container import ServiceContainer
    from core.phases.proprioceptive_loop import ProprioceptiveLoop

    def pulse(origin: str) -> float:
        state = asyncio.run(ProprioceptiveLoop(ServiceContainer).execute(_state(origin, "bryan")))
        return float(state.soma.expressive["pulse_rate"])

    ledger = warmth_module.get_warmth_ledger()
    _history(ledger)
    for _ in range(8):
        ledger.heard("bryan", warm=True, objected=False)
    away = pulse("system")
    warmth_module.reset_for_test()
    ledger = warmth_module.get_warmth_ledger()
    _history(ledger)
    for _ in range(8):
        ledger.heard("bryan", warm=True, objected=False)
    here = pulse("user")
    assert here < away


def test_the_conversation_phase_says_what_it_heard() -> None:
    from core.phases.conversational_dynamics_phase import ConversationalDynamicsPhase

    ConversationalDynamicsPhase._name_the_kind(_state("user", "bryan"), "Thank you, that was lovely of you.")
    assert warmth_module.get_warmth_ledger().read().person == "bryan"
