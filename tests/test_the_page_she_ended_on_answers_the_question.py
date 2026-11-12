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
