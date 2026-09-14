"""A pulse read off somebody's turns, and how far off it she sat.

Entrainment is the first thing a record does to a listener and it happens
before comprehension. These pin the reading: whose turns it comes from, what
it says when there are too few, and that placement is measured rather than
corrected.
"""

from __future__ import annotations

import pytest

from core.expression.entrainment import MIN_TURNS, Cadence, cadence, placement


def _said(role: str, text: str, at: float | None = None) -> dict:
    entry = {"role": role, "content": text}
    if at is not None:
        entry["timestamp"] = at
    return entry


def test_a_pulse_is_read_from_their_turns_and_not_from_hers() -> None:
    history = [
        _said("user", "yes"),
        _said("assistant", "a very much longer reply than anything they wrote"),
        _said("user", "ok"),
        _said("assistant", "another long one"),
        _said("user", "sure"),
    ]
    theirs = cadence(history)
    assert theirs.measured
    assert theirs.turns == 3
    assert theirs.chars == pytest.approx(3.0)


def test_the_gap_between_their_turns_is_read_when_the_turns_carry_a_clock() -> None:
    history = [
        _said("user", "first", at=100.0),
        _said("user", "second", at=110.0),
        _said("user", "third", at=120.0),
    ]
    assert cadence(history).gap == pytest.approx(10.0)


def test_turns_without_a_clock_say_so_rather_than_claiming_a_gap() -> None:
    theirs = cadence([_said("user", "a"), _said("user", "b"), _said("user", "c")])
    assert theirs.gap == 0.0
    assert "no clock" in theirs.why


def test_too_few_turns_is_not_a_pulse() -> None:
    theirs = cadence([_said("user", "hello"), _said("user", "there")])
    assert not theirs.measured
    assert f"of {MIN_TURNS} turns" in theirs.why


def test_only_their_recent_turns_count() -> None:
    history = [_said("user", "x" * 100)] * 3 + [_said("user", "y" * 10)] * 8
    assert cadence(history).chars == pytest.approx(10.0)


def test_placement_says_how_far_off_their_length_she_sat() -> None:
    theirs = cadence([_said("user", "x" * 100)] * 3)
    assert placement("x" * 150, theirs) == pytest.approx(0.5)
    assert placement("x" * 50, theirs) == pytest.approx(-0.5)
    assert placement("x" * 100, theirs) == pytest.approx(0.0)


def test_with_no_pulse_there_is_nothing_to_be_off() -> None:
    assert placement("anything", Cadence()) == 0.0


def test_empty_turns_do_not_make_a_pulse() -> None:
    theirs = cadence([_said("user", "   "), _said("user", ""), _said("user", "real")])
    assert not theirs.measured


def test_the_reading_serialises_what_a_reader_needs() -> None:
    row = cadence([_said("user", "one"), _said("user", "two"), _said("user", "three")]).as_dict()
    for key in ("chars", "gap", "turns", "measured", "why"):
        assert key in row, key
