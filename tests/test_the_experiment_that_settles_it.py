"""Choosing the one thing to try next, in any world.

The pairwise form of this could compare two readings of a sequence over an
enumeration written into it. These check the general one: several accounts, any
acts, and a choice made on what the outcome would settle.
"""

from __future__ import annotations

import pytest

from core.cognition.the_experiment_that_settles_it import (
    every_act_that_settles_a_sequence,
    how_many_would_settle_it,
    what_it_ruled_out,
    what_these_acts_cannot_separate,
    what_to_try,
)

# A world with no sequences in it: which of four coins is the heavy one, where
# an act is a pair of pans and the answer is which side goes down.
COINS = ("a", "b", "c", "d")


def _weighs(heavy: str, act: tuple[tuple[str, ...], tuple[str, ...]]) -> str:
    left, right = act
    if heavy in left:
        return "left"
    if heavy in right:
        return "right"
    return "level"


WEIGHINGS = [
    (("a",), ("b",)),
    (("a", "b"), ("c", "d")),
    (("a",), ("b", "c")),
]


def test_it_picks_the_act_that_settles_most_not_merely_one_that_settles_some():
    """A balance answers three ways, and counting the ways is the whole point.

    Two against two splits four coins in half: one bit, every time. One against
    one splits them three ways — this one, that one, or neither — and settles
    one and a half. Anything that scores acts by whether they separate a pair
    calls these equal.
    """
    best = what_to_try({coin: coin for coin in COINS}, WEIGHINGS, predicts=_weighs)
    assert best is not None
    assert best.do == (("a",), ("b",))
    assert best.tells_apart == 3
    assert best.settles == pytest.approx(1.5)

    halving = [one for one in WEIGHINGS if one == (("a", "b"), ("c", "d"))]
    lesser = what_to_try({coin: coin for coin in COINS}, halving, predicts=_weighs)
    assert lesser is not None and lesser.settles == pytest.approx(1.0)


def test_an_outcome_rules_out_what_expected_otherwise():
    left = what_it_ruled_out(
        {coin: coin for coin in COINS}, (("a", "b"), ("c", "d")), "left", predicts=_weighs
    )
    assert sorted(left) == ["a", "b"]


def test_a_second_act_finishes_it():
    after = what_it_ruled_out(
        {coin: coin for coin in COINS}, (("a", "b"), ("c", "d")), "left", predicts=_weighs
    )
    then = what_to_try(after, WEIGHINGS, predicts=_weighs)
    assert then is not None
    assert len(what_it_ruled_out(after, then.do, "left", predicts=_weighs)) == 1


def test_worst_case_depth_is_reported():
    assert (
        how_many_would_settle_it({coin: coin for coin in COINS}, WEIGHINGS, predicts=_weighs)
        == 2
    )


def test_an_account_that_will_not_answer_survives_every_outcome():
    """Silence is not a wrong prediction, and must not be scored as one."""

    def sometimes(one, act):
        return None if one == "mute" else _weighs(one, act)

    field = {"a": "a", "b": "b", "mute": "mute"}
    left = what_it_ruled_out(field, WEIGHINGS[0], "left", predicts=sometimes)
    assert sorted(left) == ["a", "mute"]


def test_an_act_that_silences_the_field_is_never_chosen():
    def mute(_one, act):
        return None if act == "useless" else _weighs(_one, act)

    best = what_to_try(
        {coin: coin for coin in COINS}, ["useless", *WEIGHINGS], predicts=mute
    )
    assert best is not None and best.do != "useless"


def test_a_cheap_act_can_beat_a_dear_one_that_settles_more():
    """The choice is bits per unit cost, so cost has to be able to change it."""
    field = {coin: coin for coin in COINS}
    free = what_to_try(field, WEIGHINGS, predicts=_weighs)
    dear = what_to_try(
        field,
        WEIGHINGS,
        predicts=_weighs,
        costs=lambda act: 100.0 if act == free.do else 1.0,
    )
    assert dear is not None and dear.do != free.do
    assert dear.costs == 1.0
    assert dear.worth_doing > free.settles / 100.0


def test_nothing_to_settle_returns_nothing():
    assert what_to_try({"a": "a"}, WEIGHINGS, predicts=_weighs) is None
    assert what_to_try({coin: coin for coin in COINS}, [], predicts=_weighs) is None


def test_the_first_of_equally_good_acts_wins_so_the_answer_does_not_hash():
    """A set with nothing to choose between its members must not choose by
    whichever address came first out of a dict."""
    twice = [WEIGHINGS[0], WEIGHINGS[0]]
    picks = {
        what_to_try({coin: coin for coin in COINS}, twice, predicts=_weighs).do
        for _ in range(5)
    }
    assert len(picks) == 1


def test_an_equivalence_says_how_hard_she_looked():
    """A thin act space makes two accounts look like one. Measured: two
    readings agreed over lengths two and four and parted on the first act of
    length three, so the answer that carries no act count is a trap."""
    seen = what_these_acts_cannot_separate(
        {"a": "a", "b": "b"}, [WEIGHINGS[1]], predicts=_weighs
    )
    assert seen.groups == (("a", "b"),)
    assert seen.acts_tried == 1
    assert "1 acts" in str(seen)

    wider = what_these_acts_cannot_separate(
        {"a": "a", "b": "b"}, WEIGHINGS, predicts=_weighs
    )
    assert not wider.groups
    assert bool(wider) is False


def test_it_cannot_be_settled_by_acts_that_never_part_them():
    assert (
        how_many_would_settle_it(
            {"a": "a", "b": "b"}, [WEIGHINGS[1]], predicts=_weighs
        )
        is None
    )


def test_shorter_accounts_carry_more_weight_before_any_evidence():
    """Description length is the prior when nothing else is offered, so an act
    that settles the likely part of the field beats one that settles the
    unlikely part."""
    from core.cognition.an_invented_kind import everything_that_fits, addressings

    along = addressings()["one along"]
    shown = [
        (state, tuple(state[along(at, len(state)) % len(state)] for at in range(len(state))))
        for state in [(3, 7), (4, 9), (2, 5)]
    ]
    readings = {one.name: one for one in everything_that_fits(shown)}
    assert len(readings) > 2
    best = what_to_try(
        readings,
        every_act_that_settles_a_sequence(2),
        predicts=lambda reading, state: reading.read(state),
    )
    assert best is not None
    assert best.settles == pytest.approx(best.unsettled_now)
    left = what_it_ruled_out(
        readings,
        best.do,
        readings["take one along"].read(best.do),
        predicts=lambda reading, state: reading.read(state),
    )
    assert list(left) == ["take one along"]


def test_the_sequence_act_space_settles_every_reading_it_admits():
    """The general chooser inherits the pairwise form's completeness rather
    than approximating it."""
    from core.cognition.an_invented_kind import everything_that_fits, addressings

    along = addressings()["one along"]
    shown = [
        (state, tuple(state[along(at, len(state)) % len(state)] for at in range(len(state))))
        for state in [(3, 7), (4, 9), (2, 5)]
    ]
    readings = {one.name: one for one in everything_that_fits(shown)}
    acts = every_act_that_settles_a_sequence(2)
    assert not what_these_acts_cannot_separate(
        readings, acts, predicts=lambda r, s: r.read(s)
    ).groups
    assert (
        how_many_would_settle_it(
            readings, acts, predicts=lambda r, s: r.read(s)
        )
        == 1
    )


def test_a_board_is_a_world_like_any_other():
    """Not sequences: which way a tile board answers a swipe."""
    board = ((2, 2, 0, 0),)

    def slides(_one, way):
        return {"left": (2, 2, 0, 0), "right": (0, 0, 2, 2)}[way]

    def merges(_one, way):
        return {"left": (4, 0, 0, 0), "right": (0, 0, 0, 4)}[way]

    field = {"it only slides": slides, "it merges": merges}
    best = what_to_try(
        field, ["left", "right"], predicts=lambda how, way: how(board, way)
    )
    assert best is not None and best.tells_apart == 2
    left = what_it_ruled_out(
        field, best.do, merges(board, best.do), predicts=lambda how, way: how(board, way)
    )
    assert list(left) == ["it merges"]
