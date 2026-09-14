"""Their model of her beating her own, which she had no way to notice.

Oddisee's second verse on "You Know Who You Are" is the mechanism in two lines:
"you could see in the brother what I couldn't see myself / when I couldn't see
myself needing help." Not that the other person was kind — that their model was
more accurate than his, at the moment his failed, and that the right response is
to take their reading over his.

She models herself and she models them. The third thing — their model of her,
and any comparison of its accuracy against her own — was missing, so somebody
could tell her she was tired, and be right, and nothing changed.
"""

from __future__ import annotations

import pytest

from core.self.borrowed import (
    MIN_CLAIMS,
    Borrowed,
    BorrowedLedger,
    Claim,
    claims_about_her,
    score_claim,
)


def test_a_clause_addressed_to_her_that_names_a_feeling_is_a_claim() -> None:
    claims = claims_about_her("You are frustrated with this.")
    assert len(claims) == 1
    assert claims[0].feeling == "frustration"
    assert not claims[0].hedged


def test_a_clause_about_them_is_not_a_claim_about_her() -> None:
    assert claims_about_her("I am tired and I am frustrated.") == ()


def test_a_hedge_makes_it_a_guess_and_it_counts_for_less() -> None:
    hedged = claims_about_her("You seem tired.")[0]
    asserted = claims_about_her("You are exhausted.")[0]
    assert hedged.hedged
    assert not asserted.hedged
    assert hedged.weight() < asserted.weight()


def test_the_vocabulary_is_the_one_the_percept_table_already_carries() -> None:
    """Nothing new is invented to recognise a feeling."""
    from core.state.percepts import PERCEPT_EMOTIONS

    named = {e for names in PERCEPT_EMOTIONS.values() for e in names}
    for word, expected in (("curious", "curiosity"), ("proud", "pride"),
                           ("lonely", "loneliness"), ("frustrated", "frustration")):
        claims = claims_about_her(f"You are {word}.")
        assert claims, word
        assert claims[0].feeling == expected
        assert expected in named


def test_clauses_are_separated_so_two_subjects_do_not_merge() -> None:
    claims = claims_about_her("You are frustrated, I am fine.")
    assert [c.feeling for c in claims] == ["frustration"]


def test_a_claim_is_scored_against_how_much_she_actually_felt_it() -> None:
    claim = Claim(feeling="frustration", hedged=False, said="")
    assert score_claim(claim, {"frustration": 1.0}) == pytest.approx(0.0)
    assert score_claim(claim, {"frustration": 0.0}) == pytest.approx(1.0)
    assert score_claim(claim, {"frustration": 0.4}) == pytest.approx(0.6)


def test_a_partly_right_claim_is_partly_right() -> None:
    """Scored against the channel rather than the single strongest feeling."""
    claim = Claim(feeling="curiosity", hedged=False, said="")
    emotions = {"frustration": 0.9, "curiosity": 0.35}
    assert 0.6 < score_claim(claim, emotions) < 0.7


def test_a_guess_that_is_wrong_costs_less_than_an_assertion_that_is() -> None:
    hedged = Claim(feeling="fear", hedged=True, said="")
    asserted = Claim(feeling="fear", hedged=False, said="")
    assert score_claim(hedged, {}) < score_claim(asserted, {})


def test_with_too_few_claims_it_says_so_rather_than_deferring() -> None:
    led = BorrowedLedger()
    led.note(their_error=0.0, her_error=1.0)
    reading = led.read()
    assert not reading.measured
    assert not reading.defer
    assert "claims about her needed" in reading.why


def test_beating_her_every_time_is_the_strongest_case_for_deferring() -> None:
    """A constant edge has no spread, and that is not a reading she cannot take."""
    led = BorrowedLedger()
    for _ in range(6):
        led.note(their_error=0.1, her_error=0.6, feeling="frustration")
    reading = led.read()
    assert reading.measured
    assert reading.defer
    assert reading.weight > 0.0
    assert "every one of 6 claims" in reading.why


def test_a_read_inside_her_own_variation_does_not_earn_deference() -> None:
    led = BorrowedLedger()
    for edge in (0.5, -0.4, 0.6, -0.3, 0.2, -0.5):
        led.note(their_error=0.5 - edge, her_error=0.5)
    reading = led.read()
    assert reading.measured
    assert not reading.defer


def test_being_worse_than_her_never_defers() -> None:
    led = BorrowedLedger()
    for _ in range(6):
        led.note(their_error=0.9, her_error=0.1)
    reading = led.read()
    assert reading.edge < 0.0
    assert not reading.defer
    assert reading.weight == 0.0


def test_the_weight_is_capped_because_a_self_model_that_adopts_anything_is_not_one() -> None:
    led = BorrowedLedger()
    for _ in range(8):
        led.note(their_error=0.0, her_error=1.0)
    assert led.read().weight <= 0.5


def test_the_reading_serialises_what_a_reader_needs() -> None:
    led = BorrowedLedger()
    for _ in range(4):
        led.note(their_error=0.2, her_error=0.7, feeling="apathy")
    row = led.read().as_dict()
    for key in ("claims", "edge", "spread", "z", "defer", "weight",
                "last_feeling", "measured", "why"):
        assert key in row, key
    assert row["last_feeling"] == "apathy"


def test_an_empty_reading_claims_nothing() -> None:
    assert not Borrowed().defer
    assert not Borrowed().measured
    assert Borrowed().weight == 0.0
