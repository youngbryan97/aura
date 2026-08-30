"""A question about what she can do is answered from the register, not from priors.

LIVE 2026-08-30, asked to prove she can invent primitives for the language she
makes rules out of, she said her architecture "does not possess the capability
to dynamically extend, modify, or add new fundamental primitives" and called
that language "the static set of instructions defined by my developers" — two
turns after answering with a word she had derived, kept, and recalled across a
restart. Nothing was wrong with the model. There was nothing for it to read.
"""

from __future__ import annotations

import pytest

from core.self.what_is_established import (
    asks_what_is_established,
    what_is_established_block,
)


@pytest.mark.parametrize(
    "asked",
    [
        "You claim you can invent new primitives for your own representation "
        "language. Prove it.",
        "You cannot extend your own representation language.",
        "How do you know you actually learn anything?",
        "What are your capabilities, with evidence?",
        "Do you really have a world model, or is that marketing?",
        "you're not able to change how you represent anything",
    ],
)
def test_a_challenge_to_what_she_is_reaches_the_record(asked):
    assert asks_what_is_established(asked)


@pytest.mark.parametrize(
    "asked",
    [
        "Can you open Safari for me?",
        "Could you summarise this file?",
        "Please play 2048.",
        "what's the weather like",
        "read my clipboard",
    ],
)
def test_a_request_to_do_something_does_not(asked):
    """"Can you X" is a request. Dragging the register in would answer nothing."""
    assert not asks_what_is_established(asked)


def test_the_statements_it_returns_are_the_ones_the_question_is_about():
    said = what_is_established_block(
        "You claim you can invent new primitives for your own representation "
        "language. Prove it. What could you not express before that you can now?"
    )
    assert "the language she makes rules out of" in said
    assert "way of building words" in said


def test_every_statement_carries_the_test_that_checks_it():
    said = what_is_established_block("prove your representation language can grow")
    assert said
    for line in said.splitlines():
        if line.startswith("- "):
            continue
        if line.strip().startswith("checked by:"):
            assert "test_" in line or "_" in line


def test_a_word_that_appears_everywhere_does_not_decide_the_ranking():
    """Function words are in every statement and tell them apart not at all.

    Counting shared words without weighting put six statements about reply
    custody above the one the question was asking about.
    """
    said = what_is_established_block(
        "what can you tell me about the record of that turn and the reply"
    )
    generic = what_is_established_block(
        "prove your representation language can grow new words"
    )
    assert said != generic


def test_nothing_comes_back_for_a_turn_that_is_not_asking():
    assert what_is_established_block("open the browser") == ""


def test_the_whole_register_stands_down_for_a_question_about_one_thing():
    from core.brain.validated_claims_grounding import validated_claims_block

    asked = (
        "You claim you can invent new primitives for your own representation "
        "language. Prove it."
    )
    assert what_is_established_block(asked).strip()
    assert validated_claims_block(asked) == ""


def test_an_uninstalled_register_is_built_rather_than_reported_empty():
    """Reporting "no claim can be supported" reads as a denial of all of it."""
    from core.brain.validated_claims_grounding import validated_claims_block

    said = validated_claims_block("what have you actually measured about yourself?")
    assert "could not be built" not in said
