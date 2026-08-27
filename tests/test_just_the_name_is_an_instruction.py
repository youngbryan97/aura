""""Just the name" is how people ask for a bare answer, and it was not heard.

Only explicit counts were recognised — three sentences, fifty words, four
bullets — so the ordinary phrasing registered as no instruction at all.

LIVE 2026-08-27: "Look up who won the 2026 Turing Award and tell me just the
name." came back with three hundred and sixty-four characters, most of them
pasted from the page it was read off, in the page's own voice:

    I checked live web evidence. We heartily congratulate the Turing Award
    Prize Winners of 2026. #...: Charles Bennett and Gilles Brassard have been
    named the winners...

The names were in there. So was a hashtag fragment and somebody else's
congratulations.
"""

from __future__ import annotations

import pytest

from core.conversation.response_reliability import (
    _REQUEST_COVERAGE_REASONS,
    _asked_for_a_bare_answer,
    _instruction_coverage_reasons,
)

PASTE = (
    "I checked live web evidence. We heartily congratulate the Turing Award "
    "Prize Winners of 2026. #...: Charles Bennett and Gilles Brassard have "
    "been named the winners of the A.M. Turing Award."
)
BARE = "Charles Bennett and Gilles Brassard."


# ── hearing the instruction ──────────────────────────────────────────────

@pytest.mark.parametrize(
    "said",
    [
        "Look up who won the 2026 Turing Award and tell me just the name.",
        "Who won? Just the name.",
        "Reply with only the name.",
        "Give me just the number.",
        "The date only, please.",
        "What is the capital of Peru? Nothing else.",
        "Who wrote it? No preamble.",
        "Say only the title.",
    ],
)
def test_a_request_for_a_bare_answer_is_recognised(said):
    assert _asked_for_a_bare_answer(said) is True


@pytest.mark.parametrize(
    "said",
    [
        "Tell me about the Turing Award.",
        "Explain the Turing Award in detail.",
        "What happened, and why does it matter?",
        "Give me the background on quantum information theory.",
        "",
    ],
)
def test_an_ordinary_question_is_not_one(said):
    assert _asked_for_a_bare_answer(said) is False


# ── and acting on it ─────────────────────────────────────────────────────

def test_a_paste_where_a_name_was_asked_for_is_flagged():
    asked = "Look up who won the 2026 Turing Award and tell me just the name."
    assert "missing_requested_bare_answer" in _instruction_coverage_reasons(asked, PASTE)


def test_and_the_bare_answer_is_not():
    asked = "Look up who won the 2026 Turing Award and tell me just the name."
    assert "missing_requested_bare_answer" not in _instruction_coverage_reasons(asked, BARE)


def test_a_long_answer_to_an_ordinary_question_is_left_alone():
    assert _instruction_coverage_reasons("Tell me about the Turing Award.", PASTE) == []


# ── and it is treated like its siblings ──────────────────────────────────

def test_it_is_a_coverage_reason_like_the_counts_are():
    """So a caller that cannot isolate the request discounts it the same way."""
    assert "missing_requested_bare_answer" in _REQUEST_COVERAGE_REASONS
    assert "missing_requested_word_count" in _REQUEST_COVERAGE_REASONS


def test_it_can_be_said_back_as_something_to_do():
    from core.brain.llm import mlx_worker

    import inspect

    assert "give the answer on its own, as asked" in inspect.getsource(mlx_worker)
