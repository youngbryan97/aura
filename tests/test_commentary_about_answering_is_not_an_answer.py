"""A model handed a scaffold and a failed draft sometimes writes about the job.

What the user asked, why the last attempt was rejected, what the contract
requires. Every word of it is fluent, on topic and well formed, so nothing
that measures shape catches it — and it arrives where an answer should be.

LIVE 2026-08-27, in full, to "What is 17 times 23?":

    We need answer user's current message: "What is 17 times 23?" Need direct
    arithmetic. Previous draft failed for missing numeric answer. We must
    return only requested user-facing content? User asks simple math. Need
    maybe just "391". But contract says direct first-person continuity...

The test is grammatical rather than topical. A reply speaks TO a person: there
is no third party in it called "the user", and nothing in an answer needs to
report what the user asked, because the user knows. The same goes for its own
earlier attempts, which the reader was never shown.
"""

from __future__ import annotations

import pytest

from core.conversation.response_reliability import internal_leak_reasons
from core.utils.an_answer import talks_about_the_asking

LEAKED = (
    'We need answer user\'s current message: "What is 17 times 23?" Need direct '
    "arithmetic. Previous draft failed for missing numeric answer. We must return "
    "only requested user-facing content? User asks simple math. Need maybe just "
    '"391". But contract says direct first-person continuity, warmth concrete '
    "attention ordinary conversation."
)


# ── the thing that shipped ───────────────────────────────────────────────

def test_the_reply_that_reached_a_person():
    assert talks_about_the_asking(LEAKED) is True


def test_and_it_is_treated_as_an_internal_leak():
    assert "internal_task_prompt_leak" in internal_leak_reasons(LEAKED)


# ── what marks it ────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "said",
    [
        "The user asked for three sentences.",
        "The user wants a number.",
        "I'll answer the user directly.",
        "user's current message is about arithmetic",
        "User asks simple math.",
        "Previous draft failed for missing numeric answer.",
        "The contract says first-person continuity.",
        "We must return only the requested content.",
    ],
)
def test_commentary_about_the_exchange_is_recognised(said):
    assert talks_about_the_asking(said) is True


# ── and ordinary sentences are not ───────────────────────────────────────

@pytest.mark.parametrize(
    "said",
    [
        "391",
        "It is 391.",
        "The user interface is on the left.",
        "The user account settings live under Preferences.",
        "The user experience of that page is poor.",
        "Your users will expect a fast reply.",
        "A sliding puzzle moves one tile at a time.",
        "I could not work that out, and I would rather say so.",
        "Users of the old version will need to migrate.",
    ],
)
def test_an_ordinary_sentence_is_left_alone(said):
    assert talks_about_the_asking(said) is False


@pytest.mark.parametrize("said", ["", "   ", "\n"])
def test_nothing_is_not_commentary(said):
    assert talks_about_the_asking(said) is False


def test_a_real_answer_carries_no_leak_reasons():
    assert internal_leak_reasons("It is 391.") == ()
    assert internal_leak_reasons("The user interface is on the left.") == ()


# ── announcing an answer, versus saying what she does ────────────────────


def test_the_announcement_alone_is_still_commentary():
    """"I'll answer the user directly." is the whole reply, and that is the fault."""

    assert talks_about_the_asking("I'll answer the user directly.") is True


def test_the_same_phrase_inside_a_real_answer_is_not():
    """Asked how she handles a turn, saying so IS the answer.

    Live 2026-08-28: a substantive draft was dropped as an internal prompt leak
    for containing "answer the user" — in a sentence that went on to name three
    other things she would do. Banning the phrase banned her from describing
    how she works, which is a thing people ask her about.

    The mark is not the subject and not the phrase. It is whether anything else
    was said.
    """

    said = (
        "I would answer the user directly, preserve the current thread, and "
        "keep the live lane moving instead of detonating a long retry cascade."
    )
    assert talks_about_the_asking(said) is False


def test_a_plural_or_absent_subject_is_still_a_plan():
    """The exemption is first person singular, because a plan is not."""

    assert talks_about_the_asking("We must answer the user with the requested content.") is True
    assert talks_about_the_asking("Need to answer the user before the contract expires.") is True
