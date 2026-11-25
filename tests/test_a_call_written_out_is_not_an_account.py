"""A reply that writes out a call to one of her tools in place of making it has not told anything
(core/conversation/response_reliability.py ``first_call_written_out``; interface/routes/chat_desktop_objective.py).

LIVE 25 November 2026, her account of a game she could not finish reached the person as "Let me try to find my way
back to the board ... [TOOLCALL computer_use {"action": "pursue_on_screen", ...}] Wait — I need to use the proper tool
format." No tool was offered to that turn, and nothing was done.
"""
from __future__ import annotations

import asyncio

import pytest

from core.conversation.response_reliability import first_call_written_out

pytestmark = pytest.mark.unit

TOOLS = ["open_door", "fetch_water", "count"]


def test_a_tool_named_and_followed_by_its_arguments_is_a_call_written_out_whatever_wraps_it() -> None:
    for wrapped in ('[CALL open_door {"which": "front"}]', '<<open_door {"which": "front"}>>',
                    'open_door {"which": "front", "how": {"slowly": true}}'):
        text = f"I'll let them in.\n\n{wrapped}\n\nThere."
        assert first_call_written_out(text, TOOLS) == text.index("open_door"), wrapped


def test_a_call_cut_off_where_the_reply_ends_is_a_call_written_out() -> None:
    text = 'Going back for more.\n\n[CALL fetch_water {"bucket": "the big one'
    assert first_call_written_out(text, TOOLS) == text.index("fetch_water")


def test_a_tool_named_in_a_sentence_or_beside_something_that_is_not_its_arguments_is_not_a_call() -> None:
    assert first_call_written_out("I used open_door to let them in, and count to keep track.", TOOLS) is None
    assert first_call_written_out("The count {of three} was right.", TOOLS) is None
    assert first_call_written_out("recount {\"n\": 3}", TOOLS) is None, "a longer word is not the tool"
    assert first_call_written_out('open_door {"which": "front"}', []) is None, "with no tools, nothing is a call"


def test_her_account_of_a_game_that_writes_out_a_call_is_dropped_and_the_record_told(monkeypatch) -> None:
    import core.skills.watching_along as watching
    import interface.routes.chat_desktop_objective as route

    async def writes_a_call(*_args, **_kwargs) -> str:
        return 'The app is still open though.\n\n[CALL open_door {"goal": "get back to the board"}]'

    async def tells_it(*_args, **_kwargs) -> str:
        return "I lost that one in thirty moves."

    monkeypatch.setattr(route, "_her_tool_names", lambda: TOOLS)
    monkeypatch.setattr(watching, "her_turn", writes_a_call)
    assert asyncio.run(route._her_account_of_a_game("play a game", "Played a game: 30 moves; I lost.")) == ""
    monkeypatch.setattr(watching, "her_turn", tells_it)
    assert asyncio.run(route._her_account_of_a_game("play a game", "Played a game: 30 moves; I lost.")) == (
        "I lost that one in thirty moves.")
