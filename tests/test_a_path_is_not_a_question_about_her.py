"""The words inside an address are not words the person said.

LIVE, 2026-08-25: "Something weird is happening in a little project of mine at
/private/tmp/claude-501/-Users-bryan--aura-live-source/.../invoice-tools.
There's no error and no failing test, but the second invoice comes out with the
first one's lines in it. I've stared at it for an hour. What's actually going
on?"

The answer was a recitation of 79 capability entries ending "For this turn I am
only describing the measured tool surface; I am not opening apps, browsing,
typing, moving files, or executing tools." The word "aura" sits inside the
PATH, which put her within eighty characters of a capability word, and the
structural rule counted that as her being the subject of the question.

`core/intent/opaque_spans.py` was written for this and cites this same
directory. It just was not being called here.
"""

from __future__ import annotations

import pytest

from interface.routes.chat_preflight import _is_explicit_capability_inventory_request

_LOADED_PATH = (
    "/private/tmp/claude-501/-Users-bryan--aura-live-source/"
    "7a6cdc9e-da7f-47f7-8c38-8cfadf95a75e/scratchpad/invoice-tools"
)


def test_a_diagnosis_naming_a_path_with_her_name_in_it_is_not_an_inventory_question() -> None:
    asked = (
        f"Something weird is happening in a little project of mine at {_LOADED_PATH}. "
        "There's no error and no failing test, but the second invoice comes out with "
        "the first one's lines in it. What's actually going on?"
    )
    assert _is_explicit_capability_inventory_request(asked) is False


def test_the_same_sentence_without_the_path_is_also_not_one() -> None:
    """So the path is what changed the verdict, not the question."""
    asked = (
        "Something weird is happening in a little project of mine. There's no error "
        "and no failing test, but the second invoice comes out with the first one's "
        "lines in it. What's actually going on?"
    )
    assert _is_explicit_capability_inventory_request(asked) is False


@pytest.mark.parametrize(
    "asked",
    [
        "what tools do you have",
        "what are your capabilities",
        "what can you actually do on this computer right now?",
        "show me your tools",
    ],
)
def test_a_real_inventory_question_still_gets_the_inventory(asked: str) -> None:
    assert _is_explicit_capability_inventory_request(asked) is True


@pytest.mark.parametrize(
    "asked",
    [
        "what is the capital of Peru",
        "can you do the marble problem?",
        "how does confusion change your planning?",
        "write me a one-pager about the migration",
    ],
)
def test_questions_that_are_not_about_her_inventory(asked: str) -> None:
    assert _is_explicit_capability_inventory_request(asked) is False


def test_adding_a_loaded_path_does_not_change_any_verdict() -> None:
    """The general rule, not the one phrasing that bit.

    A path is an address. Appending one must not move a prose judgement, for
    any of these, in either direction.
    """
    sentences = [
        "what is the capital of Peru",
        "read that file and tell me what it says",
        "why is this test failing",
        "what tools do you have",
    ]
    for sentence in sentences:
        plain = _is_explicit_capability_inventory_request(sentence)
        with_path = _is_explicit_capability_inventory_request(f"{sentence} at {_LOADED_PATH}")
        assert plain == with_path, f"the path changed the verdict for {sentence!r}"
