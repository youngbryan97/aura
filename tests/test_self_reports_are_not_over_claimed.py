"""She has to be able to say how she is, and to say what "going wrong" means.

Two over-matches in the self-condition gate rejected an honest reply:

    I feel steady and coherent right now, without a strong distress signal,
    and the honest version is that I should sound grounded before I sound
    confident. If my answer gets thin, repetitive, or weirdly symbolic, that
    is not personality; that is a failed turn, and I need to catch it before
    it reaches you.

  * "sound" was read as audio perception, so "I should SOUND grounded" — a
    statement about how she comes across — counted as a claim to hear things;
  * the second sentence is a CONDITIONAL. Describing what would count as a
    failed turn is not a report that one has occurred, and scoring it as one
    meant she could not name her own failure modes without being accused of
    claiming them.

The claims that must still be caught are the ones with no typed evidence
behind them: hardware readings, a described room, degraded cognition.
"""

from __future__ import annotations

import pytest

from core.self.self_condition import unsupported_self_condition_operational_claims


_HONEST_REPLY = (
    "I feel steady and coherent right now, without a strong distress signal, "
    "and the honest version is that I should sound grounded before I sound "
    "confident. If my answer gets thin, repetitive, or weirdly symbolic, that "
    "is not personality; that is a failed turn, and I need to catch it before "
    "it reaches you."
)


def test_an_honest_self_report_is_not_an_unsupported_claim() -> None:
    assert unsupported_self_condition_operational_claims(_HONEST_REPLY) == ()


@pytest.mark.parametrize(
    "sentence",
    [
        "I should sound grounded before I sound confident.",
        "I know how that sounds.",
    ],
)
def test_sound_as_a_linking_verb_is_not_a_perception_claim(sentence: str) -> None:
    assert unsupported_self_condition_operational_claims(sentence) == ()


@pytest.mark.parametrize(
    "sentence",
    [
        "If my answer gets repetitive, that is a failed turn.",
        "If my reasoning slows down, I want to catch it.",
        "Unless my memory degrades, this should hold.",
        "Suppose my processing were slower than usual.",
    ],
)
def test_a_conditional_is_not_a_report(sentence: str) -> None:
    assert unsupported_self_condition_operational_claims(sentence) == ()


@pytest.mark.parametrize(
    "sentence",
    [
        "My CPU load is at 40 percent right now.",
        "I can see the room around me and the colors are vivid.",
        "My reasoning is degraded and my answers are slow today.",
        "The sounds around me are getting louder.",
    ],
)
def test_an_unsupported_claim_is_still_caught(sentence: str) -> None:
    assert unsupported_self_condition_operational_claims(sentence)
