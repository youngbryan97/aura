"""F13: when a page she worked through ends a question, the page goes to her turn and her own turn answers it.

LIVE 11 November 2026, asked in her chat "Can you take a look at r/nosleep on Reddit and tell me what the top thread
is about right now?", she reached r/nosleep, stopped because she could read the thread, and was served her reason for
stopping: "The page loaded past the gate and I can already read the top thread's title and opening — no further
navigation needed."
"""
from __future__ import annotations

import pytest

from core.conversation.turn_evidence_custody import bind_turn_evidence_custody, turn_world_evidence
from interface.routes.chat_desktop_objective import _the_page_goes_to_her_turn

pytestmark = pytest.mark.unit

_ASKED = "Can you take a look at r/nosleep on Reddit and tell me what the top thread is about right now?"
_PAGE = "r/nosleep\nI found a quarantine broadcast on a frequency nobody uses. It repeats every night at 3:12."


def _pursuit(**changes):
    result = {"ok": True, "planner": "browser_pursuit", "concluded_from": "stopping", "url": "https://www.reddit.com/r/nosleep/",
              "result_text": _PAGE, "concluded": "The page loaded past the gate and I can already read the top thread."}
    return {**result, **changes}


def test_the_page_she_ended_on_goes_beside_her_turn_labelled_as_what_she_read():
    with bind_turn_evidence_custody(session_id="s", turn_id="t"):
        assert _the_page_goes_to_her_turn(_pursuit(), _ASKED)
        (given,) = turn_world_evidence()
    assert "quarantine broadcast" in given and "read in her browser at " in given
    assert "https://www.reddit.com/r/nosleep/" in given


@pytest.mark.parametrize("changes, asked", [
    ({"concluded_from": "outcome"}, _ASKED),            # a forecast she checked keeps her verdict
    ({}, "Open r/nosleep on Reddit"),                   # a task that asks nothing keeps its account
    ({"result_text": ""}, _ASKED),                      # no page to give
    ({"ok": False}, _ASKED),
    ({"planner": "computer_use"}, _ASKED),
])
def test_anything_else_keeps_the_reply_it_had(changes, asked):
    with bind_turn_evidence_custody(session_id="s", turn_id="t"):
        assert not _the_page_goes_to_her_turn(_pursuit(**changes), asked)
        assert turn_world_evidence() == ()


def test_with_no_turn_to_attach_it_to_the_reply_it_had_stands():
    assert not _the_page_goes_to_her_turn(_pursuit(), _ASKED)


def test_a_turn_holding_the_page_she_read_is_hers_to_write_not_the_desktop_lanes():
    """LIVE 11 November 2026, the same question after the page went to her turn: the desktop contract served its
    promise to execute in place of her answer, the promise failed its authorship gate, and her background model
    answered that Reddit had blocked her before she could read anything."""
    from interface.routes.chat import _SEAM_FELL_THROUGH
    from interface.routes.chat_desktop_objective import _a_page_she_read_awaits_her_answer
    from interface.routes.chat_turn_steps import _the_desktop_contract_answers_for_itself

    def served() -> object:
        return _the_desktop_contract_answers_for_itself(
            desktop_execution_contract=True, require_engine=True, turn_trace=None, visible=_ASKED)

    with bind_turn_evidence_custody(session_id="s", turn_id="before"):
        assert not _a_page_she_read_awaits_her_answer()
        assert served() is not _SEAM_FELL_THROUGH          # as before: nothing read yet, the lane carries it
    with bind_turn_evidence_custody(session_id="s", turn_id="after"):
        assert _the_page_goes_to_her_turn(_pursuit(), _ASKED)
        assert _a_page_she_read_awaits_her_answer()
        assert served() is _SEAM_FELL_THROUGH              # her turn goes on to her model


def test_a_web_source_alone_is_not_a_page_she_read():
    from core.conversation.turn_evidence_custody import record_turn_world_source
    from core.conversation.what_the_world_says import WorldSource
    from interface.routes.chat_desktop_objective import _a_page_she_read_awaits_her_answer

    with bind_turn_evidence_custody(session_id="s", turn_id="t"):
        record_turn_world_source(WorldSource("web", "r/nosleep", "https://www.reddit.com/r/nosleep/", _PAGE))
        assert not _a_page_she_read_awaits_her_answer()


def test_with_the_page_in_hand_her_turn_does_not_go_searching():
    import asyncio

    from interface.routes.chat_desktop_evidence import _what_the_world_says_for_the_turn

    with bind_turn_evidence_custody(session_id="s", turn_id="t"):
        assert _the_page_goes_to_her_turn(_pursuit(), _ASKED)
        assert asyncio.run(_what_the_world_says_for_the_turn(_ASKED, _ASKED, session_id="s")) == (None, _ASKED)


def test_a_reader_outside_the_turn_is_told_why_its_sources_are_withheld():
    from core.conversation.session_scope import conversation_session_scope
    from core.conversation.turn_evidence_custody import current_turn_evidence_custody

    with bind_turn_evidence_custody(session_id="s", turn_id="t"):
        custody = current_turn_evidence_custody()
        assert custody.why_its_sources_are_withheld() == ""          # nothing kept yet
        assert _the_page_goes_to_her_turn(_pursuit(), _ASKED)
        assert custody.why_its_sources_are_withheld() == ""          # kept, and readable here
        with conversation_session_scope("another"):
            assert "kept for session 's'" in custody.why_its_sources_are_withheld()
