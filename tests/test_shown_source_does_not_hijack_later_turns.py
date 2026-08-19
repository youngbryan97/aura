"""Showing code once must not make every later turn a provenance question.

Both canned refusals in the 2026-08-19 session came from the same mechanism,
and neither question had anything to do with source code:

    read me the first line of CONTRIBUTING.md   -> "I couldn't get to an
    spell 'necessary' backwards                     answer I'd stand behind"

An earlier turn had shown an excerpt of associative_entity_memory.py, which
makes the "shown source" state sticky. Every turn after it was then scored for
provenance by meaning, both of these scored TRUE, and the corrector replaced
the answer with a citation sentence. The replacement failed its authorship
proof, and the turn died — over a file that had been read successfully and a
word that had been reversed successfully.

A turn another reader already answers is not a question about code shown
earlier. "Where did that come from?" is untouched, because nothing else claims
it.
"""

from __future__ import annotations

import pytest

import interface.routes.chat as chat


@pytest.fixture(autouse=True)
def shown_source():
    """The state the live session was in: an excerpt has been shown."""
    from core.self.source_excerpt import remember_shown_excerpt

    remember_shown_excerpt(
        "core/memory/associative_entity_memory.py:136 (grounded)\n\n"
        "```python\ndef grounded(self):\n    return True\n```"
    )
    assert chat._has_current_shown_source()


@pytest.mark.parametrize(
    "message",
    [
        "read me the first line of CONTRIBUTING.md",
        "spell 'necessary' backwards",
        "what is 7919 * 6367?",
        "how many r's in strawberry",
        "tell me a joke",
    ],
)
def test_an_answered_turn_is_not_a_provenance_question(message: str) -> None:
    assert not chat._turn_asks_where_that_came_from(message)


@pytest.mark.parametrize(
    "message",
    ["where did that come from?", "which file was that in?"],
)
def test_a_real_provenance_question_still_works(message: str) -> None:
    assert chat._turn_asks_where_that_came_from(message)


def test_the_rule_is_that_another_reader_owns_it() -> None:
    assert chat._another_reader_owns_this_turn("read me the first line of CONTRIBUTING.md")
    assert chat._another_reader_owns_this_turn("spell 'necessary' backwards")
    assert not chat._another_reader_owns_this_turn("where did that come from?")
