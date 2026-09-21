"""A correct answer was thrown away for being brief.

The last-resort salvage in the chat route required 80 characters and 12 words
before it would serve anything. Asked "what was the very first thing I asked
you in this conversation?", the true answer is::

    You asked what it's actually like in here right now.

Eleven words, fifty-five characters — under both floors. It was discarded, and
what went out instead was thirty-five words of "I couldn't get to an answer I'd
stand behind". The apology was longer than the answer, which is the whole
absurdity: length was never the property being tested.

What the floor is actually trying to exclude is a *fragment* — half a sentence,
a stray clause, a thought the generator dropped partway. So that is what gets
tested now: a sentence that finishes is worth serving at any length, and text
that stops mid-thought has to carry enough substance to stand on its own.
"""
from __future__ import annotations

import pytest

from interface.routes.chat import _worth_more_than_a_refusal


# ── The live failure ───────────────────────────────────────────────────────

def test_the_live_answer_is_no_longer_discarded() -> None:
    assert _worth_more_than_a_refusal("You asked what it's actually like in here right now.")


@pytest.mark.parametrize(
    "answer",
    [
        "It's 10:52 on a Monday morning.",
        "No, that build failed — there's nothing on your Desktop.",
        "Max Verstappen won the most recent one.",
        "About twenty gigabytes, wired.",
        "I haven't done anything in the last hour.",
    ],
)
def test_short_true_answers_survive(answer: str) -> None:
    """Every one of these beats "ask me again in a moment"."""
    assert _worth_more_than_a_refusal(answer)


# ── What the floor was really for ─────────────────────────────────────────

@pytest.mark.parametrize(
    "fragment",
    [
        "So the",
        "and then",
        "Well",
        "",
        "   ",
        "I think that the",
    ],
)
def test_a_fragment_is_still_refused(fragment: str) -> None:
    assert not _worth_more_than_a_refusal(fragment)


def test_an_unfinished_thought_needs_substance_to_stand() -> None:
    """No terminal punctuation, so it has to be long enough to be useful."""
    assert not _worth_more_than_a_refusal("The train leaves at quarter past and")
    assert _worth_more_than_a_refusal(
        "The second train catches the first at 5:15 in the afternoon and both "
        "are then a hundred and eighty miles from the station where they"
    )


def test_a_quoted_or_parenthesised_ending_counts_as_finished() -> None:
    assert _worth_more_than_a_refusal('You asked me to "just say it plainly."')
    assert _worth_more_than_a_refusal("The file is on your Desktop (107 bytes)")


def test_the_apology_it_replaces_is_longer_than_what_it_was_refusing() -> None:
    """The measurement that makes the old floor indefensible."""
    apology = (
        "I couldn't get to an answer I'd stand behind on that one, and I "
        "won't send you a thinner one and pass it off as the real thing. "
        "Ask me again in a moment and I should have it."
    )
    answer = "You asked what it's actually like in here right now."
    assert len(apology) > len(answer) * 2
    assert _worth_more_than_a_refusal(answer)


def test_a_finished_sentence_is_served_at_any_length():
    """The rule the docstring states, applied.

    Underneath it sat a length floor — four words and twenty characters —
    which is the check the docstring says was never the property being
    tested. LIVE 2026-09-07: a 26-character cortex reply at confidence=high
    was discarded here, and the turn said the cortex was unavailable while
    the cortex had just answered.
    """
    from interface.routes.chat_lane_bookkeeping import _worth_more_than_a_refusal

    asked = "does the cache survive a restart?"
    assert _worth_more_than_a_refusal("Yes, it does.", asked)
    assert _worth_more_than_a_refusal("It survives a restart.", asked)
    # a fragment still needs substance to stand without its ending
    assert not _worth_more_than_a_refusal("because the cache", asked)
    assert _worth_more_than_a_refusal(
        "because the cache is written before the process exits and read back on the next boot",
        asked,
    )
    # and punctuation alone is not an answer
    assert not _worth_more_than_a_refusal("...", asked)
    assert not _worth_more_than_a_refusal("", asked)
    # with no question in hand, a two-word sentence cannot be shown to answer
    # anything, and the floor is the honest default
    assert not _worth_more_than_a_refusal("Both red.", "")
