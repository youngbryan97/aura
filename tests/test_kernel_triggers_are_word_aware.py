"""A pronoun is a word, and the current year is not a constant.

Two triggers in the cognitive kernel decided what a turn needs, both by
substring:

  * ambiguity. "this", "it", "that", "they" are PRONOUNS, but "waiting"
    contains "it" and so does "quit", so a short message like "I'm waiting"
    was judged ambiguous and answered with a clarifying question instead of an
    answer. A spurious inquiry costs the turn its reply and hands the person a
    question they did not ask for.
  * recency. The signal list held the literal strings "2025" and "2026", so a
    question naming any later year would stop counting as being about now — an
    expiry nobody chose, arriving silently.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from core.cognition.cognitive_kernel import CognitiveKernel


@pytest.fixture
def kernel() -> CognitiveKernel:
    return object.__new__(CognitiveKernel)


@pytest.mark.parametrize("message", ["I'm waiting", "sit tight", "with the unit"])
def test_a_pronoun_inside_a_word_does_not_make_a_turn_ambiguous(
    kernel: CognitiveKernel, message: str
) -> None:
    assert not CognitiveKernel._should_inquire(kernel, message, [], "simple")


@pytest.mark.parametrize("message", ["quit that", "is it working?", "tell me about this"])
def test_a_real_pronoun_still_reads_as_ambiguous(
    kernel: CognitiveKernel, message: str
) -> None:
    assert CognitiveKernel._should_inquire(kernel, message, [], "simple")


def test_the_current_year_counts_as_recent_whatever_year_it_is(
    kernel: CognitiveKernel,
) -> None:
    year = datetime.now(UTC).year

    assert CognitiveKernel._needs_research(kernel, f"what shipped in {year}", None, 0.9)
    assert CognitiveKernel._needs_research(
        kernel, f"what shipped in {year - 1}", None, 0.9
    )


def test_a_timeless_question_does_not_demand_research(kernel: CognitiveKernel) -> None:
    assert not CognitiveKernel._needs_research(kernel, "who wrote Hamlet", None, 0.9)


def test_recency_words_still_demand_research(kernel: CognitiveKernel) -> None:
    for message in ("what happened in the news", "the current status", "latest release"):
        assert CognitiveKernel._needs_research(kernel, message, None, 0.9)
