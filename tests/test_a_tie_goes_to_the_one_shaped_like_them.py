"""How close a candidate is to the shape of what they said is scored.

The taste model reads specificity, stance, callbacks, length. It had nothing
about the shape of the exchange, so a reply shaped like a question and a reply
shaped like a report were the same to it however the other person was
speaking. The register reading measures the same quantities on both sides, and
matching them carries the weight matching the length carries.
"""

from __future__ import annotations

import pytest

from core.brain.response_quality import extract_features
from core.brain.taste_model import FEATURE_PRIORS

THEY_ASK = (
    "Do you remember when we first set this up? Do you remember what we said "
    "we wanted? Do you remember how long it took?"
)
THEY_TELL = (
    "I have been at the migration since this morning. I got the schema across "
    "and I am still holding the rest of it."
)
ASKING = "Do you remember the branch we started on? What did we call it?"
TELLING = "I was working through the migration and I have been at it since morning."


def _match(candidate: str, message: str) -> float:
    return extract_features(candidate, user_message=message)["register_match"]


def test_the_feature_is_read_for_every_candidate() -> None:
    assert "register_match" in extract_features("anything at all", user_message=THEY_ASK)


def test_a_candidate_shaped_like_their_question_matches_it() -> None:
    assert _match(ASKING, THEY_ASK) > _match(TELLING, THEY_ASK)


def test_a_candidate_shaped_like_their_report_matches_that() -> None:
    assert _match(TELLING, THEY_TELL) > _match(ASKING, THEY_TELL)


def test_nothing_to_match_scores_nothing() -> None:
    assert _match(ASKING, "") == 0.0
    assert _match("", THEY_ASK) == 0.0


def test_the_match_is_bounded() -> None:
    for candidate in (ASKING, TELLING, "ok"):
        for message in (THEY_ASK, THEY_TELL):
            assert 0.0 <= _match(candidate, message) <= 1.0


def test_matching_how_they_speak_is_weighted_like_matching_how_much_they_said() -> None:
    assert FEATURE_PRIORS["register_match"] == pytest.approx(FEATURE_PRIORS["length_fit"])


def test_a_message_too_short_to_have_a_shape_is_not_matched() -> None:
    """Six coordinates cannot be filled by five words."""
    from core.expression.register import comparable, read

    assert not comparable(read("thoughts on blade runner 2049?"), read(TELLING))
    assert _match(TELLING, "thoughts on blade runner 2049?") == 0.0
    assert comparable(read(THEY_ASK), read(TELLING))
