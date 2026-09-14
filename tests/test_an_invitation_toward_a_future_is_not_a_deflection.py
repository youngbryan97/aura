"""Encouragement asked as a question about a future the other could have.

"Can you feel the sunshine?" is sixty-one per cent questions and seventy-four
per cent second person, with thirteen future markers. Said to somebody who is
testifying, a reply shaped like that is encouragement. Said to somebody asking
for help, the same shape punts the question back. These pin both halves.
"""

from __future__ import annotations

import pytest

from core.brain.response_quality import extract_features
from core.brain.taste_model import FEATURE_PRIORS
from core.expression.register import read

TESTIFYING = (
    "I have been sitting with it all week and I still feel like nothing is going "
    "to change, but I keep getting up anyway."
)
ASKING_FOR_HELP = (
    "Can you tell me why the deploy keeps failing? What should I check first "
    "before I roll it back?"
)
INVITATION = "Can you feel it starting to shift? Could tomorrow be the day you let yourself rest a little?"
ADVICE = "You should try sleeping more and drinking water, that usually helps with this kind of thing."


def test_the_register_reads_how_much_of_an_utterance_is_about_a_future() -> None:
    assert read(INVITATION).future > 0.0
    assert read("I went to the store and came back home after.").future == 0.0


def test_an_invitation_to_somebody_testifying_is_scored() -> None:
    assert read(TESTIFYING).asks_to_be_witnessed()
    assert extract_features(INVITATION, user_message=TESTIFYING)["invitation"] > 0.0


def test_advice_to_somebody_testifying_is_not_an_invitation() -> None:
    assert extract_features(ADVICE, user_message=TESTIFYING)["invitation"] == 0.0


def test_the_same_shape_answering_a_request_is_not_scored() -> None:
    assert not read(ASKING_FOR_HELP).asks_to_be_witnessed()
    assert extract_features(INVITATION, user_message=ASKING_FOR_HELP)["invitation"] == 0.0


def test_an_invitation_is_not_counted_as_punting_the_question_back() -> None:
    invited = extract_features(INVITATION, user_message=TESTIFYING)
    deflected = extract_features(INVITATION, user_message=ASKING_FOR_HELP)
    assert invited["prompt_farm_penalty"] < deflected["prompt_farm_penalty"]


def test_the_score_is_bounded() -> None:
    value = extract_features(INVITATION, user_message=TESTIFYING)["invitation"]
    assert 0.0 <= value <= 1.0


def test_it_carries_the_prior_of_the_other_shape_fit() -> None:
    assert FEATURE_PRIORS["invitation"] == pytest.approx(FEATURE_PRIORS["register_match"])
