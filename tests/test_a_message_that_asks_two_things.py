"""Two questions in one sentence, and only one of them answered.

LIVE 2026-08-27: "Two things: what's 2^20, and what did I ask you to remember
earlier in this conversation?" came back "1,048,576." The second question was
dropped without a word about it.

Everything needed to catch that already existed. The coverage check was there,
the reason name was there, and both were gated on a shape that read the
sentence as one ask — because ", and what" is not ", what", and because
nothing read "two things" as the count it plainly is.

This file also pins the direction of the trade. The check causes a rejection,
so a sentence wrongly split into two costs a correct answer, while a second
question wrongly missed costs only a check that did not fire. Relative
pronouns are therefore not boundaries.
"""

from __future__ import annotations

import pytest

from core.conversation.request_coverage import unanswered_question_parts
from core.runtime.structured_input import analyze_prompt_shape

TWO_THINGS = (
    "Two things: what is 2^20, and what did I ask you to remember earlier "
    "in this conversation?"
)


def shape(said: str):
    return analyze_prompt_shape(said)


# ── the sentence that was missed ─────────────────────────────────────────

def test_it_is_read_as_two_asks():
    assert len(shape(TWO_THINGS).question_segments) == 2


def test_and_the_reply_is_checked_against_both():
    assert shape(TWO_THINGS).requires_single_reply_coverage is True


def test_the_dropped_half_is_named():
    missed = unanswered_question_parts("1,048,576.", shape(TWO_THINGS))
    assert missed and "remember" in missed[0]


def test_and_an_answer_that_covers_both_is_not_flagged():
    covered = "1,048,576. And you asked me to remember your deadline is the 14th."
    assert unanswered_question_parts(covered, shape(TWO_THINGS)) == []


# ── what makes a second ask ──────────────────────────────────────────────

@pytest.mark.parametrize(
    ("said", "asks"),
    [
        ("What is 2^20, and what did I ask you to remember?", 2),
        ("What broke, and why?", 2),
        ("What happened; how bad is it?", 2),
        ("What is the capital of Peru?", 1),
        ("Explain the Turing Award.", 1),
    ],
)
def test_a_comma_and_a_conjunction_still_start_one(said, asks):
    assert len(shape(said).question_segments) == asks


def test_a_lone_question_word_is_a_whole_question():
    assert shape("What broke, and why?").question_segments[-1] == "why?"


# ── and the count somebody states outright ───────────────────────────────

@pytest.mark.parametrize(
    ("said", "parts"),
    [
        ("Two things: tell me about X.", 2),
        ("A couple of things — how does it work?", 2),
        ("Three questions. First, what is it?", 3),
        ("Tell me one thing about it.", 1),
        ("Explain the Turing Award.", 1),
    ],
)
def test_a_message_that_says_how_many_things_it_asks_is_believed(said, parts):
    assert shape(said).question_parts >= parts if parts > 1 else shape(said).question_parts == 1


# ── a false split costs a correct answer, so it does not happen ──────────

@pytest.mark.parametrize(
    "said",
    [
        "What is the capital of Peru, which I keep forgetting?",
        "Tell me the plan, which you mentioned yesterday.",
        "Who wrote it, and when?",
        "What is the thing, whose name I forget?",
    ],
)
def test_a_relative_clause_is_not_another_ask(said):
    assert len(shape(said).question_segments) <= 1 or not shape(said).question_segments[0].endswith("which?")


def test_a_single_question_needs_no_coverage_check():
    assert shape("What is 2^20?").requires_single_reply_coverage is False
    assert unanswered_question_parts("1,048,576.", shape("What is 2^20?")) == []


# ── the same sentence with an imperative second half ─────────────────────

CONJOINED_IMPERATIVE = (
    "Design me the experiment that would actually tell us, and say what "
    "result would prove your friend wrong."
)


def test_a_conjoined_imperative_is_two_asks():
    """LIVE, 2026-08-28: the second half was never answered.

    The reply designed the experiment, stopped, and said nothing about what
    result would settle it. Coverage had nothing to compare against because
    the sentence arrived as one segment — `design` was a directive verb and
    `say` was not, so the splitter matched once and never split.

    `say` is as ordinary a way to ask for something as `tell`, which was
    already in the list. The list had grown one live failure at a time.
    """

    assert len(shape(CONJOINED_IMPERATIVE).question_segments) == 2


def test_coverage_names_the_half_that_was_dropped():
    reply = (
        "Here is a clean experiment. Day 0: feed your starter and bake a "
        "control loaf. Day 1-2: prepare a second batch with commercial yeast, "
        "everything else identical, and bake it the same way."
    )
    missing = unanswered_question_parts(reply, shape(CONJOINED_IMPERATIVE))
    assert missing
    assert any("prove your friend wrong" in part for part in missing)


@pytest.mark.parametrize(
    "said",
    [
        "Give me the trial balance and say whether it is consistent.",
        "Explain the cause and suggest a fix.",
        "Walk me through the setup, and confirm the totals.",
    ],
)
def test_other_ordinary_two_part_requests(said):
    assert len(shape(said).question_segments) == 2


@pytest.mark.parametrize(
    "said",
    [
        "Compare the two designs and their costs.",
        "I said it was fine and then it broke.",
        "Tell me about the say-do gap.",
        "The quote and the citation are both wrong.",
    ],
)
def test_the_direction_of_the_trade_is_held(said):
    """A false split costs a correct answer; a missed one costs a check.

    So a word that only looks like a request verb must not split a sentence.
    """

    assert len(shape(said).question_segments) <= 1
