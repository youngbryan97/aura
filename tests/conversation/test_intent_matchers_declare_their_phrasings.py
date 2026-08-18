"""Every matcher that gates a reading must state what it was probed on.

Nine matchers were fixed in one session for the same reason: each was a
hand-written regex covering the phrasings its author imagined, and each met a
real question it did not recognise.

    "what did I just copy?"                    clipboard, missed
    "the first THING I said"                   transcript, missed
    "are you planning to do anything later?"   queued work, missed
    "how many python files LIVE in X"          count, missed
    "what does X.md SAY about Y"               file read, missed
    "put BUILD-42 on my clipboard"             screen, matched wrongly
    "why do you think PEOPLE lie"              provenance, matched wrongly
    "put BUILD-42 on my clipboard"             clipboard, matched wrongly

Every one was found by a person asking, live — the most expensive place to find
it — and widening a regex afterwards only ever fixes the phrasing that was
tried. The generalisation is not a better regex: it is that the phrasings live
beside the matcher, where whoever changes it will see them, and a test proves
the matcher still agrees with them.

Observable carries the same contract for the grounding registry. These tests
cover both, so adding either kind forces the same question: what did you
actually try?
"""

from __future__ import annotations

import pytest

import core.conversation.intent_registry  # noqa: F401  (registers)
from core.conversation.intent_contract import INTENT_MATCHERS, matcher_failures


def test_matchers_are_registered() -> None:
    """An empty registry would make every assertion below vacuous."""
    assert len(INTENT_MATCHERS) >= 8


@pytest.mark.parametrize("matcher", INTENT_MATCHERS, ids=lambda m: m.name)
def test_each_matcher_declares_examples(matcher) -> None:
    assert matcher.examples, (
        f"{matcher.name} ({matcher.where}) declares no examples. State the "
        "questions it must recognise."
    )


@pytest.mark.parametrize("matcher", INTENT_MATCHERS, ids=lambda m: m.name)
def test_each_matcher_declares_counter_examples(matcher) -> None:
    """Over-matching is where these do their damage: they steal other turns."""
    assert matcher.counter_examples, (
        f"{matcher.name} ({matcher.where}) declares no counter-examples."
    )


@pytest.mark.parametrize("matcher", INTENT_MATCHERS, ids=lambda m: m.name)
def test_each_matcher_agrees_with_its_own_examples(matcher) -> None:
    failures = matcher.failures()

    assert not failures, "\n".join(failures)


def test_the_whole_set_agrees() -> None:
    """One assertion that names every disagreement at once."""
    failures = matcher_failures()

    assert not failures, "\n".join(failures)
