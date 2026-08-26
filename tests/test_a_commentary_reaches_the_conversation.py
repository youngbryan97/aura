"""A running commentary is not a second answer to the question.

Asked to narrate a task and then watched doing it, she said nothing in the
conversation for the whole run. The bubble had every line and the chat window
had none.

The rule that swallowed them exists for a real failure — one answer arriving
over two lanes, so the person reads it twice. But it withholds any unprompted
speech while a turn is open, and a task she is doing runs INSIDE the turn that
asked for it. The turn stays open until the work finishes, so every line of
commentary was withheld until there was nothing left to narrate.

An answer supersedes another answer. It does not supersede an event.
"""
from __future__ import annotations

import interface.event_bridge as eb
from core.conversation.surface_delivery import note_turn_started, route_answer_supersedes
from core.schemas import (
    ActionResultPayload,
    AuraMessagePayload,
    ChatStreamChunkPayload,
    ChatThoughtChunkPayload,
    CognitiveThoughtPayload,
    WebsocketMessage,
)

A_MOVE = "Board: Up — keeping the largest tiles along the top."

SCHEMAS = {
    "AuraMessagePayload": AuraMessagePayload,
    "WebsocketMessage": WebsocketMessage,
    "ChatStreamChunkPayload": ChatStreamChunkPayload,
    "ChatThoughtChunkPayload": ChatThoughtChunkPayload,
    "CognitiveThoughtPayload": CognitiveThoughtPayload,
    "ActionResultPayload": ActionResultPayload,
}


def _narration(line: str = A_MOVE) -> dict:
    return {
        "type": "aura_message",
        "message": line,
        "timestamp": 1.0,
        "metadata": {"autonomic": True, "narration": True},
    }


def test_a_second_answer_to_an_open_turn_is_still_withheld():
    """The rule this fix touches has to keep doing its job."""
    note_turn_started(conversation_id="c-answer", turn_id="t-answer")
    assert route_answer_supersedes(
        "Here is the answer to what you asked.",
        conversation_id="c-answer",
        turn_id="t-answer",
        unprompted=True,
    )


def test_a_commentary_is_not_withheld_while_the_turn_is_open():
    note_turn_started(conversation_id="c-narrate", turn_id="t-narrate")
    assert not route_answer_supersedes(
        A_MOVE, conversation_id="c-narrate", turn_id="t-narrate", unprompted=True, answering=False
    )


def test_a_narrated_move_survives_the_whole_bridge_during_an_open_turn():
    note_turn_started(conversation_id="c-bridge", turn_id="t-bridge")
    mapped = eb._map_event_to_ws_message("telemetry", _narration(), **SCHEMAS)
    assert mapped and mapped["message"] == A_MOVE
    assert not eb._suppress_internal_leak(mapped), "the commentary was withheld as a second answer"
    shaped = eb._shape_user_facing_ws_message(mapped, is_gui_proxy=False)
    assert shaped and shaped["message"] == A_MOVE


def test_the_narration_marker_survives_to_the_surface():
    """Without it the chat window collapses repeated moves into one.

    Pressing the same key twice is two things that happened.
    """
    mapped = eb._map_event_to_ws_message("telemetry", _narration(), **SCHEMAS)
    shaped = eb._shape_user_facing_ws_message(mapped, is_gui_proxy=False)
    assert shaped["metadata"].get("narration") is True


def test_the_chat_window_does_not_deduplicate_a_commentary():
    """Read from the surface itself, so a rule kept only in Python is not
    mistaken for a rule the browser follows."""
    from pathlib import Path

    surface = Path("interface/static/aura.js").read_text(encoding="utf-8")
    where = surface.index("} else if (type === 'aura_message'")
    handler = surface[where : where + 1200]
    assert "if (!meta.narration)" in handler
    assert "rememberMessageFingerprint" in handler


def test_the_bridge_asks_whether_this_is_an_answer_at_all():
    import inspect

    source = inspect.getsource(eb._suppress_internal_leak)
    assert 'answering=not bool(metadata.get("narration"))' in source


def test_a_question_that_names_its_own_answer_is_not_faulted_for_getting_it_back():
    """Live: "Reply with one word: ready" answered "ready" was rejected as
    adding nothing beyond the question, and the person was refused with a
    canned line instead. The word WAS the answer.

    Where the request pins the reply to a handful of words, coverage is not
    the test to apply — and the contract that pinned it is already parsed.
    """
    import core.conversation.response_reliability as rr

    for asked, answered in (
        ("Reply with one word: ready", "ready"),
        ("Answer in one word: up or down?", "up"),
    ):
        contract = rr.requested_output_contract(asked)
        assert contract.word_max is not None and contract.word_max <= rr._COVERAGE_EXEMPT_WORDS
        assert rr._adds_nothing_to_the_question(asked, answered), "the premise of the old fault"


def test_giving_an_open_question_back_is_still_faulted():
    """The rule keeps doing its job where nothing pinned the reply."""
    import core.conversation.response_reliability as rr

    asked = "Explain how the cache works"
    assert rr.requested_output_contract(asked).word_max is None
    assert rr._adds_nothing_to_the_question(asked, "How the cache works")
    assert rr._requires_substantive_reply(asked)


def test_the_exemption_reads_the_contract_rather_than_the_wording():
    """Every phrasing that pins a reply to one word has to be covered, and
    there are many. Reading the parsed contract covers them all; a list of
    phrasings covers the ones somebody thought of."""
    import inspect

    import core.conversation.response_reliability as rr

    source = inspect.getsource(rr._assess_user_facing_reply)
    where = source.index("pinned_to_a_fragment")
    assert "requested_output_contract(user_message).word_max" in source[where : where + 400]
