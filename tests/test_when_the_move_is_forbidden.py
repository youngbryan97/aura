"""Wanting a move she may not make yet, and buying it somewhere else.

Go's ko rule forbids immediately recreating the position that was just there,
and what it produces is one of the strangest useful shapes in any game: when
you cannot make the move you want, you make a different one elsewhere that
they must answer, and while they answer it the move you wanted becomes legal.

The tests are a held lock, because that is the same shape.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.cognition.when_the_move_is_forbidden import a_way_round


@dataclass
class Afterwards:
    """A world, and what it allows once something has been dealt with."""

    freed: bool

    def allows(self, wanted: str) -> bool:
        return self.freed


#: Ignoring these costs them this much. Only some of them free the lock.
THREATS = {
    "ask them to pause": 0.2,          # too small to be answered
    "unblock their build": 3.0,        # they answer, and the lock is released
    "take the whole service down": 90.0,  # answered, frees it, and absurd
    "file a ticket": 5.0,              # answered, and frees nothing
}
FREES = {"unblock their build", "take the whole service down"}


def _way(worth: float = 1.0):
    return a_way_round(
        "take the lock",
        allowed=lambda _one: False,
        elsewhere=list(THREATS),
        they_must_answer=lambda one: THREATS[one],
        after_they_answer=lambda one: Afterwards(one in FREES),
        worth_of_the_fight=worth,
    )


def test_it_finds_something_elsewhere_that_makes_the_move_available() -> None:
    got = _way()
    assert got.found
    assert got.spend_a_turn_on == "unblock their build", got.describe()


def test_a_threat_too_small_to_be_answered_is_not_a_threat() -> None:
    """They will simply take the thing instead."""
    assert _way().spend_a_turn_on != "ask them to pause"


def test_the_smallest_threat_that_still_works_is_the_one() -> None:
    """A threat worth more than the fight is not a threat, it is a move she
    should be making anyway, and spending it here throws away the difference."""
    got = _way()
    assert got.worth == 3.0
    assert got.spend_a_turn_on != "take the whole service down"


def test_something_they_must_answer_that_frees_nothing_is_no_use() -> None:
    assert _way().spend_a_turn_on != "file a ticket"


def test_a_fight_bigger_than_every_threat_has_no_way_round() -> None:
    got = _way(worth=1000.0)
    assert not got.found
    assert "nothing she can do elsewhere" in got.describe()


def test_a_move_already_allowed_needs_no_way_round() -> None:
    got = a_way_round(
        "take the lock",
        allowed=lambda _one: True,
        elsewhere=list(THREATS),
        they_must_answer=lambda one: THREATS[one],
        after_they_answer=lambda one: Afterwards(True),
        worth_of_the_fight=1.0,
    )
    assert got.found
    assert got.spend_a_turn_on == "take the lock"
    assert got.worth == 0.0
