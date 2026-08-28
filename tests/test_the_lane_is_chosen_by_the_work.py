"""A turn whose answer must be worked out does not go down the quick lane.

The compact lane exists because present state, recall and capability all have a
snapshot to read from, so the full phase stack adds no evidence. That is not
true of a question whose answer does not exist until it has been derived.

LIVE, 2026-08-27: "45 becomes 15. 28 becomes 14. 66 becomes 22. What am I
doing, what does 91 become?" went compact on two question parts and stopped
mid-derivation. The same class of question with three parts had gone the long
way one turn earlier and found the rule cleanly.
"""

from __future__ import annotations

import pytest

from core.runtime.structured_input import analyze_prompt_shape
from interface.routes.chat import (
    _is_compact_desktop_chat_contract,
    _the_answer_has_to_be_worked_out,
)

_A_DERIVATION = (
    "Another sequence for you. 45 becomes 15. 28 becomes 14. 66 becomes 22. "
    "What am I doing, what does 91 become, and is three examples enough to pin "
    "it down or could a different rule fit these too?"
)

_A_READING = (
    "how are you doing?",
    "what are you running on?",
    "are you ok?",
    "what did we talk about earlier?",
    "who are you?",
    "what can you do?",
    "what is on my screen?",
    "how much memory are you using?",
    "tell me about yourself",
    "what time is it",
    "are you still there",
)


def test_a_sequence_rule_is_work_not_a_reading() -> None:
    assert _the_answer_has_to_be_worked_out(
        _A_DERIVATION, analyze_prompt_shape(_A_DERIVATION)
    )


@pytest.mark.parametrize("asked", _A_READING)
def test_reading_the_present_stays_on_the_quick_lane(asked: str) -> None:
    assert not _the_answer_has_to_be_worked_out(asked, analyze_prompt_shape(asked))


def test_three_parts_still_count_as_work_on_their_own() -> None:
    asked = "Where is the log, what is in it, and when did it last change?"
    assert _the_answer_has_to_be_worked_out(asked, analyze_prompt_shape(asked))


def test_the_derivation_is_refused_the_compact_contract() -> None:
    assert not _is_compact_desktop_chat_contract(
        _A_DERIVATION,
        _A_DERIVATION,
        desktop_execution_contract=False,
        capability_inventory_contract=False,
    )


#: Identity and self-process turns are held off the compact lane by their own
#: older branches, for reasons that have nothing to do with derivation. They
#: belong in the predicate list above and not in this one.
_STILL_COMPACT = tuple(
    asked
    for asked in _A_READING
    if asked not in ("who are you?", "how much memory are you using?")
)


@pytest.mark.parametrize("asked", _STILL_COMPACT)
def test_the_quick_turns_keep_the_compact_contract(asked: str) -> None:
    assert _is_compact_desktop_chat_contract(
        asked,
        asked,
        desktop_execution_contract=False,
        capability_inventory_contract=False,
    )


def test_a_named_path_is_not_in_any_snapshot() -> None:
    """The quick lane reads a snapshot; the bytes are somewhere else.

    LIVE, 2026-08-28: "Something's off in <path> ... Go through the code and
    tell me what's actually happening, with the file and line" is one long
    sentence with one question in it, so nothing about its shape asked for
    room. It went compact with 512 tokens, was handed diagnose_repo, and ran
    out of budget before it could say what the tool found.
    """

    import os

    here = os.getcwd()
    asked = (
        f"Something's off in {here} and I can't put my finger on it. No error, "
        "nothing crashes, the tests such as they are pass. Go through the code "
        "and tell me what's actually happening, with the file and line."
    )
    assert _the_answer_has_to_be_worked_out(asked, analyze_prompt_shape(asked))


def test_a_path_that_does_not_exist_is_not_a_place_to_look() -> None:
    asked = "Something's off in /no/such/place/at/all, go through it"
    assert not _the_answer_has_to_be_worked_out(asked, analyze_prompt_shape(asked))


def test_one_reader_owns_pointing_at_something_real() -> None:
    from core.intent import capability_selection

    assert hasattr(capability_selection, "points_at_something_real")
    assert "points_at_something_real" in capability_selection.__all__
