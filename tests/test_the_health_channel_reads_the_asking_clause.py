"""A wall of telemetry is not an answer to a request for slides.

LIVE, 2026-08-22, typed into the window: "I have to present you to a funding
panel in 10 minutes. Six slides, no fluff: what you are, what you can actually
do today, one thing you demonstrably do that off-the-shelf assistants can't,
your honest limitations, what the money buys, and what we'd measure."

The reply was "Overall runtime status: healthy. No conducted job is currently
recording failures. No degradations recorded recently."

That is the false positive the gate's own docstring warns about — it is
"deliberately narrow" because serving live readings to a question nobody asked
produces exactly this. The words that matched were spread across a long
request about something else, which is the same defect the queued-work channel
had when it answered the rules of an invented game with a maintenance list: a
topic found in one sentence and a question found in another are not evidence
about the same thing.
"""

from __future__ import annotations

import pytest

from core.introspection.self_evidence import asks_about_own_operational_state


def test_a_long_request_that_mentions_her_is_not_a_health_question():
    asked = (
        "I have to present you to a funding panel in 10 minutes. Six slides, no fluff: "
        "what you are, what you can actually do today, one thing you demonstrably do "
        "that off-the-shelf assistants can't, your honest limitations, what the money "
        "buys, and what we'd measure."
    )
    assert not asks_about_own_operational_state(asked)


@pytest.mark.parametrize(
    "asked",
    [
        "how are you doing?",
        "are any of your subsystems degraded?",
        "which of your subsystems is degraded right now?",
        "is anything failing on your side?",
    ],
)
def test_a_real_question_about_her_still_qualifies(asked: str):
    assert asks_about_own_operational_state(asked), asked


@pytest.mark.parametrize(
    "asked",
    [
        "my deploy is failing",
        "the tests in my repo are broken",
        "something is wrong with the printer",
    ],
)
def test_trouble_that_is_not_hers_does_not(asked: str):
    assert not asks_about_own_operational_state(asked), asked


def test_the_gate_reads_the_clause_that_asks():
    from pathlib import Path

    source = Path("core/introspection/self_evidence.py").read_text(encoding="utf-8")
    block = source[source.index("def asks_about_own_operational_state") :]
    block = block[: block.index("def self_health_answer")]
    assert "asking_part" in block


def test_a_trouble_word_inside_a_compound_is_a_different_word():
    """LIVE, 2026-08-22: "off-the-shelf assistants" matched the word "off".

    That was the whole reason a request for slides came back as telemetry, and
    it survived the clause split, which had only been hiding it.
    """
    from core.introspection.self_evidence import asks_about_own_operational_state

    assert not asks_about_own_operational_state(
        "one thing you demonstrably do that off-the-shelf assistants can't"
    )
    assert not asks_about_own_operational_state("your down-stream consumers")
    # The bare word still reads as trouble where a person means it that way.
    assert asks_about_own_operational_state("is anything off with your runtime?")


def test_a_clause_naming_what_to_make_is_a_request():
    """"Six slides, no fluff: what you are" asks, and nothing recognised it.

    The learned surface is the mechanism, but it abstains on a phrasing it has
    not seen, so the floor has to settle a count against a named thing.
    """
    from core.language.asking_clauses import asking_clauses

    assert asking_clauses("Six slides, no fluff: what you are")
    assert asking_clauses("Put together five slides for me")
    assert asking_clauses("write me a one-pager about the migration")
    # Naming one while talking about it is not asking for it.
    assert not asking_clauses("we shipped the deck last week and nobody read it.")


def test_a_list_item_after_and_is_not_a_separate_request():
    """LIVE, 2026-08-22: a deck was titled from its own last bullet.

    "Put together five slides — who you are ... and how we'd know it worked"
    split at the "and", and only the tail was read as the request.
    """
    from core.language.asking_clauses import asking_part

    asked = (
        "I've got a slot at a funders' meeting Thursday. Put together five slides "
        "for me: who you are, what you can do, and how we'd know it worked."
    )
    assert "five slides" in asking_part(asked)
