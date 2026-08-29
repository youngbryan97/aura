"""Two different questions wear the same words, and one record answers only one.

The provenance graph says which phase ran, on which branch, under which
criteria. It has nothing to say about whether a thing is true — and "show me
why you believe it" asks the second question.

LIVE 2026-08-29: "Tell me the single most interesting thing in it and show me
why you believe it." She answered honestly that the file was not there, and
then "• AffectUpdatePhase — took the ordinary_decay branch — on arousal=0.4 —
changed affect.curiosity" was stapled underneath it. The verb alternation in
the matcher ends in a bare "that|this|it", so any "why ... you ... it" matched.
"""

from __future__ import annotations

import pytest

from core.introspection.decision_provenance import asks_why_she_did_that

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "asked",
    [
        "Tell me the most interesting thing in it and show me why you believe it.",
        "why do you believe it",
        "why are you sure about that",
        "why are you confident in that number",
    ],
)
def test_grounds_for_a_claim_do_not_summon_the_phase_record(asked: str) -> None:
    """Asked for grounds, the grounds are the answer."""

    assert asks_why_she_did_that(asked) is False


@pytest.mark.parametrize(
    "asked",
    [
        "why did you pick that file",
        "why did you skip that step",
        "why did you answer that way",
        "why did you choose that approach",
        "why did you decide to stop",
    ],
)
def test_an_account_of_what_she_did_still_gets_the_record(asked: str) -> None:
    assert asks_why_she_did_that(asked) is True


def test_a_question_about_the_world_is_still_excluded() -> None:
    """The exclusion this one sits beside, kept."""

    assert asks_why_she_did_that("why do you think people lie") is False


def test_think_is_left_alone_because_it_is_genuinely_ambiguous() -> None:
    """"Why do you think that is?" is a real question about her reasoning.

    It has routed to the record for as long as this has existed, and the
    certainty verbs are the ones that are only ever about a claim.
    """

    assert asks_why_she_did_that("why do you think that is?") is True


def test_the_record_is_only_appended_when_it_was_asked_for() -> None:
    from pathlib import Path

    source = Path("interface/routes/chat.py").read_text(encoding="utf-8")
    assert "if not asks_why_she_did_that(user_message):" in source
    assert "return reply_text" in source
