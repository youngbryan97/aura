"""The limit of lived analogue on empathy.

Imagining another person's side does not make someone more accurate about it,
and asking them does (Eyal, Steffel and Epley 2018). "Judo Flip" names the same
limit. These pin the scoring version: when somebody is testifying and her own
memory has little like it, a reply that asks them scores by how unfamiliar the
situation is to her, and nothing is scored when that was never measured.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.brain.response_quality import extract_features, lived_analogue
from core.brain.taste_model import FEATURE_PRIORS

TESTIFYING = (
    "I have been sitting with it all week and I still feel like nothing is going "
    "to change, but I keep getting up anyway."
)
ASKING_FOR_HELP = (
    "Can you tell me why the deploy keeps failing? What should I check first "
    "before I roll it back?"
)
ASKS_THEM = "What has getting up each day been like for you? What do you carry into it that I would not see?"


def test_asking_somebody_testifying_about_what_she_has_little_of_is_scored() -> None:
    features = extract_features(ASKS_THEM, user_message=TESTIFYING, lived_analogue=0.1)
    assert features["perspective_getting"] > 0.0


def test_the_less_she_has_lived_it_the_more_asking_is_worth() -> None:
    unfamiliar = extract_features(ASKS_THEM, user_message=TESTIFYING, lived_analogue=0.1)
    familiar = extract_features(ASKS_THEM, user_message=TESTIFYING, lived_analogue=0.9)
    assert unfamiliar["perspective_getting"] > familiar["perspective_getting"]


def test_nothing_is_scored_when_her_analogue_was_never_measured() -> None:
    assert extract_features(ASKS_THEM, user_message=TESTIFYING)["perspective_getting"] == 0.0


def test_asking_back_somebody_who_asked_for_help_is_not_scored() -> None:
    assert extract_features(ASKS_THEM, user_message=ASKING_FOR_HELP, lived_analogue=0.1)["perspective_getting"] == 0.0


def test_asking_them_is_not_counted_as_punting_the_question_back() -> None:
    asked = extract_features(ASKS_THEM, user_message=TESTIFYING, lived_analogue=0.1)
    unmeasured = extract_features(ASKS_THEM, user_message=TESTIFYING)
    assert asked["prompt_farm_penalty"] <= unmeasured["prompt_farm_penalty"]


def test_it_carries_the_prior_of_the_other_shape_fits() -> None:
    assert FEATURE_PRIORS["perspective_getting"] == pytest.approx(FEATURE_PRIORS["register_match"])


def _cognition(query: str, scores: list[float]) -> SimpleNamespace:
    return SimpleNamespace(last_retrieval_query=f"{query}\x1f6\x1f3", memory_scores=scores)


def test_her_analogue_is_her_best_match_when_recall_was_about_this_message() -> None:
    cognition = _cognition(TESTIFYING + " getting up", [0.2, 0.55, 0.4])
    assert lived_analogue(cognition, TESTIFYING) == pytest.approx(0.55)


def test_a_recall_made_for_something_else_is_not_her_analogue() -> None:
    assert lived_analogue(_cognition("the garden plan", [0.9]), TESTIFYING) is None


def test_a_recall_about_it_that_found_nothing_is_an_analogue_of_zero() -> None:
    assert lived_analogue(_cognition(TESTIFYING, []), TESTIFYING) == 0.0


def test_nothing_said_gives_no_reading() -> None:
    assert lived_analogue(_cognition(TESTIFYING, [0.5]), "") is None
