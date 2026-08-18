"""A screen question is never answered by a promise to look.

LIVE 2026-08-17: "what's on my screen right now?" returned "I couldn't get to
an answer I'd stand behind on that one." The turn was classified
self-sufficient, so it served the desktop execution placeholder — "I will
execute this through the governed desktop_task lane and report only
receipt-verified effects" — without invoking the engine. The authorship gate
then refused that placeholder, correctly: it is not her answer, because it is
not an answer.

The narrower predicate it replaced admitted only questions a description cannot
answer ("what was that repo you saw?"), reasoning that a plain "what's on my
screen" IS served by the reading. That reasoning is right. It is served by the
reading arriving as grounding, which the observable registry now does, and not
by a sentence about what the executor intends to do.
"""

from __future__ import annotations

import pytest

from interface.routes.chat import (
    _desktop_objective_self_sufficient_without_cognitive_text as self_sufficient,
)


@pytest.mark.parametrize(
    "question",
    [
        "what's on my screen right now?",
        "what is on my screen?",
        "what do you see on my screen",
        "what was that repo you saw on my screen?",
    ],
)
def test_screen_questions_require_her_own_answer(question: str) -> None:
    assert self_sufficient(question) is False


@pytest.mark.parametrize(
    "objective",
    [
        "create a file called notes.txt on my desktop",
        "put BUILD-42 on my clipboard",
    ],
)
def test_real_desktop_actions_still_take_the_executor_path(objective: str) -> None:
    """The exclusion is about ANSWERS, not about doing things."""
    assert self_sufficient(objective) is True
