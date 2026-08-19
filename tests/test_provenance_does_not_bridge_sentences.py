"""A "why" in one sentence is not a question about a "you" in the next.

LIVE 2026-08-18, a three-part request ending "Don't pad it." The three answers
came back correct — a two-sentence explanation, the exact file count, and an
honest note about which was less certain — and then the reply continued:

    From the runtime's own record of that turn, not from my impression of it:
    • ProprioceptiveLoop — changed cognition.modifiers, soma.hardware...
    • SocialContextPhase — changed cognition.modifiers...

A phase-by-phase provenance dump, appended to a request that had explicitly
asked for no padding.

The provenance matcher spanned the whole message: its window stopped only at a
question mark, so "an explanation of why eventual consistency is hard" in the
first sentence joined "you're less sure" in the third and read as a question
about her own decisions. One sentence is the boundary.
"""

from __future__ import annotations

import pytest

from core.introspection.decision_provenance import asks_why_she_did_that

_THREE_PART_REQUEST = (
    "I need three things and I'm going to be picky. One: a two-sentence "
    "explanation of why eventual consistency is hard to reason about. Two: the "
    "exact number of .py files in core/agency. Three: tell me which of those "
    "two you're less sure about and why. Don't pad it."
)


def test_a_why_about_the_world_does_not_bridge_into_a_why_about_her() -> None:
    assert not asks_why_she_did_that(_THREE_PART_REQUEST)


@pytest.mark.parametrize(
    "question",
    [
        "why did you do that?",
        "why did you choose that file?",
        "why did you skip the verification step?",
        "why do you think that is?",
    ],
)
def test_a_real_question_about_her_decision_still_routes(question: str) -> None:
    assert asks_why_she_did_that(question)


@pytest.mark.parametrize(
    "question",
    [
        # The neighbours this matcher already disclaimed.
        "why do you think people find it hard to admit they were wrong?",
        "why do humans procrastinate?",
        "explain why eventual consistency is hard",
    ],
)
def test_a_question_about_the_world_is_not_claimed(question: str) -> None:
    assert not asks_why_she_did_that(question)
