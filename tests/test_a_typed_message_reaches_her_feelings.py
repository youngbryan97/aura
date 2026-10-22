"""A message typed into her chat is read for what it did, and felt.

Live on 22 October the quick lane answered every typed message before any phase
ran: four minutes after a message her feelings had not moved by a decimal, and
her own ticks between messages read her own objective as the message.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.social.warmth import get_warmth_ledger, reset_for_test


@pytest.fixture(autouse=True)
def _fresh_warmth():
    reset_for_test()
    yield
    reset_for_test()


def test_an_objection_from_somebody_usually_warm_lands_harder():
    ledger = get_warmth_ledger()
    for _ in range(6):
        ledger.heard("Bryan", warm=True, objected=False)
    ledger.heard("Bryan", warm=False, objected=True)
    from_a_friend = ledger.act()
    ledger.heard("stranger", warm=False, objected=True)
    from_a_stranger = ledger.act()
    assert from_a_friend < from_a_stranger < 0.0


def test_warmth_counts_by_how_far_it_rose_above_what_they_had_been():
    ledger = get_warmth_ledger()
    ledger.heard("stranger", warm=True, objected=False)
    assert ledger.act() == pytest.approx(0.5)
    ledger.heard("stranger", warm=False, objected=False)
    assert ledger.act() == 0.0


def test_the_appraisal_takes_each_message_once_as_their_doing():
    from core.phases.affect_update import AffectUpdatePhase
    from core.state.aura_state import AuraState

    phase = AffectUpdatePhase(SimpleNamespace())
    state = AuraState.default()
    get_warmth_ledger().heard("Bryan", warm=False, objected=True)
    first = phase.appraisal.read(state, state.affect, company=True)
    again = phase.appraisal.read(state, state.affect, company=True)
    assert first.their_doing == pytest.approx(-0.5)
    assert again.their_doing != first.their_doing


def test_a_quick_reply_turn_runs_the_conversation_then_affect():
    from core.brain.cognitive_engine_thinking_loop import _RunsTheThinkingLoop

    ran: list[str] = []

    def phase(name):
        async def execute(state, objective=None, **_):
            ran.append(name)
            return state
        return type(name, (), {"execute": staticmethod(execute)})()

    engine = SimpleNamespace(_phases=[phase("AffectUpdatePhase"), phase("MemoryRetrievalPhase"), phase("ConversationalDynamicsPhase")])
    state = object()
    out = asyncio.run(_RunsTheThinkingLoop._feel_the_quick_turn(engine, state, "you were wrong"))
    assert out is state
    assert ran == ["ConversationalDynamicsPhase", "AffectUpdatePhase"]
