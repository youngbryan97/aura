"""Register: who it is about, whether it asks, and whether it holds on.

Derived from fourteen records and checked against them. The discrimination
that matters is the one she had no way to make: somebody testifying is not
somebody asking, and assistance is the wrong response to testimony. Nobody
wants Sam Cooke's record fixed.

The classification these tests pin is the one the measurements make on the
records themselves — Sam Cooke and "Somehow." as testimony wanting a witness,
"Remember the Time" as a request, "Dancing in the Moonlight" with no first
person in it at all.
"""

from __future__ import annotations

import pytest

from core.expression.register import Register, distance, held_token, read, toward

COOKE = (
    "I was born by the river in a little tent. Oh, and just like the river "
    "I've been running ever since. It's been a long, a long time coming, but I "
    "know a change gonna come. It's been too hard living but I'm afraid to die. "
    "There's been times that I thought I couldn't last for long, but now I "
    "think I'm able to carry on."
)
MJ = (
    "Do you remember when we fell in love? Do you remember how it all began? "
    "Do you remember back in the fall? Do you remember us holding hands? "
    "Do you remember the time?"
)
MOONLIGHT = (
    "Everybody feel warm and bright. It's such a wild and natural sight. "
    "Everybody dancing in the moonlight. We like our fun and we never fight. "
    "Everybody here is out of sight. They don't bark and they don't bite."
)
SOMEHOW = (
    "Living by myself, paranoid as hell. There's nothing I can do. Yesterday's "
    "tomorrow, my dreams are never true. But somehow I keep on loving you. "
    "I broke out of my shell just to find that I don't belong."
)


def test_a_first_person_report_reads_as_testimony() -> None:
    r = read(COOKE)
    assert r.measured
    assert r.stance() == "testimony"
    assert r.first > 0.8
    assert r.asking == 0.0


def test_testimony_that_holds_on_asks_to_be_witnessed() -> None:
    """The response to it is not assistance."""
    for text in (COOKE, SOMEHOW):
        r = read(text)
        assert r.asks_to_be_witnessed(), r.as_dict()
        assert not r.asks_for_help()


def test_questions_are_what_a_request_looks_like() -> None:
    r = read(MJ)
    assert r.asks_for_help()
    assert not r.asks_to_be_witnessed()
    assert r.asking >= 0.5


def test_the_second_person_is_detected_as_address() -> None:
    r = read(MJ)
    assert r.stance() == "address"
    assert r.second > r.first


def test_communal_joy_has_no_first_person_in_it() -> None:
    """Measured on the record: pron_I is exactly zero."""
    r = read(MOONLIGHT)
    assert r.first == 0.0
    assert r.stance() in {"report", "together"}
    assert not r.asks_to_be_witnessed()


def test_testimony_without_holding_on_is_not_a_call_for_a_witness() -> None:
    """The connectives of persistence are part of the signature."""
    r = read("I went to the shop. I bought bread. I came home. I made toast.")
    assert r.stance() == "testimony"
    assert r.persistence == 0.0
    assert not r.asks_to_be_witnessed()


def test_the_refrain_is_the_phrase_that_comes_back() -> None:
    r = read(MJ)
    assert "do you remember" in r.refrain
    assert r.refrain_times >= 4


def test_a_longer_phrase_repeating_outranks_a_shorter_one() -> None:
    text = "hold on hold on hold on " + "if you love me won't you say " * 4
    r = read(text)
    assert r.refrain_times >= 4
    assert len(r.refrain.split()) >= 4


def test_persistence_is_a_rate_so_length_is_not_insistence() -> None:
    short = read("But still, somehow, I keep on.")
    long = read("But still, somehow, I keep on. " + "The weather was fine. " * 40)
    assert short.persistence > long.persistence


def test_duration_marks_significance() -> None:
    """The loaded word gets the time, not the rhyme."""
    words = [("it's", 0.0, 0.2), ("been", 0.2, 2.4), ("a", 2.4, 2.6), ("long", 2.6, 3.0)]
    word, ratio = held_token(words)
    assert word == "been"
    assert ratio > 3.0


def test_held_token_on_nothing_says_nothing() -> None:
    assert held_token([]) == ("", 0.0)
    assert held_token([("x", 1.0, 1.0)]) == ("", 0.0)


def test_distance_is_zero_to_itself_and_bounded() -> None:
    a, b = read(COOKE), read(MJ)
    assert distance(a, a) == pytest.approx(0.0)
    assert 0.0 < distance(a, b) <= 1.0
    assert distance(a, b) == pytest.approx(distance(b, a))


def test_an_unmeasured_register_has_no_distance() -> None:
    assert distance(Register(), read(COOKE)) == 0.0
    assert toward(Register(), read(COOKE)) == ()


def test_toward_names_the_widest_gaps_first() -> None:
    theirs, mine = read(MJ), read(COOKE)
    moves = toward(theirs, mine)
    assert moves
    assert abs(moves[0].gap) >= abs(moves[-1].gap)
    assert all(move.axis for move in moves)


def test_empty_text_is_not_a_reading() -> None:
    for text in ("", "   ", None):
        r = read(text)  # type: ignore[arg-type]
        assert not r.measured
        assert r.stance() == "impersonal"
        assert not r.asks_for_help()
        assert not r.asks_to_be_witnessed()


def test_the_reading_serialises_what_a_reader_needs() -> None:
    row = read(COOKE).as_dict()
    for key in ("words", "stance", "first", "second", "plural", "third", "asking",
                "persistence", "variety", "refrain", "refrain_times",
                "clause_words", "clauses", "asks_for_help",
                "asks_to_be_witnessed", "measured"):
        assert key in row, key
