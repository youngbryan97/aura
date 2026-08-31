"""Precious, spendable, and doing as little as possible.

Checked against the position the recorded game actually ended in, and against
something with no board in it at all.
"""

from __future__ import annotations

from core.cognition.what_she_cannot_afford_to_lose import (
    how_to_hold_what_is_already_won,
    what_an_act_risks,
    what_she_cannot_afford_to_lose,
)
from tests.two_sided_game_support import (
    ACTS,
    MINE,
    MINE_MOVES,
    THEIRS,
    THEIR_MOVES,
    back_row_whole,
    board_of,
    parts_of,
    theirs,
    without,
)

# Her back row untouched, her men frozen behind a wall of theirs, and one
# crowned piece loose. This is the shape the recording ended in.
JAMMED = board_of(
    {
        (7, 0): (MINE, False),
        (7, 2): (MINE, False),
        (7, 4): (MINE, False),
        (7, 6): (MINE, False),
        (3, 4): (MINE, True),
        (6, 1): (THEIRS, False),
        (6, 3): (THEIRS, False),
        (6, 5): (THEIRS, False),
        (6, 7): (THEIRS, False),
        (5, 0): (THEIRS, False),
        (5, 2): (THEIRS, False),
        (5, 4): (THEIRS, False),
        (5, 6): (THEIRS, False),
    }
)


def test_the_one_thing_she_cannot_spend_is_found_not_declared() -> None:
    """Nothing tells it a crown is worth more. It takes each piece away and looks."""
    precious = what_she_cannot_afford_to_lose(
        JAMMED,
        holding=back_row_whole,
        parts_of=parts_of,
        without=without,
        acts=ACTS,
        step=MINE_MOVES,
    )
    crowned = [one for one in precious if one[3]]
    assert crowned, precious
    # And her back row men are precious too, for the other reason: the thing
    # she is holding is exactly that they are there.
    assert sum(1 for one in precious if one[0] == 7) == 4
    # Every piece is precious here and that is correct: she has five and needs
    # all five. What matters is that it is not the same set in a position
    # where she has something to spare.
    spare = board_of(dict(zip(
        [(7, 0), (7, 2), (7, 4), (7, 6), (3, 4), (2, 1), (6, 1)],
        [(MINE, False)] * 4 + [(MINE, True), (MINE, False), (THEIRS, False)],
        strict=True,
    )))
    can_spend = [
        one
        for one in parts_of(spare)
        if one[2] == MINE
        and one
        not in what_she_cannot_afford_to_lose(
            spare, holding=back_row_whole, parts_of=parts_of, without=without,
            acts=ACTS, step=MINE_MOVES,
        )
    ]
    assert can_spend, "with a piece to spare, something should be spendable"


def test_she_will_not_put_the_one_she_needs_where_they_can_take_it() -> None:
    # Her men walled in, so the crowned piece is the only thing keeping her
    # able to move at all — which is what makes it the one she must not hang.
    board = board_of(
        {
            (7, 0): (MINE, False),
            (7, 2): (MINE, False),
            (7, 4): (MINE, False),
            (7, 6): (MINE, False),
            (4, 5): (MINE, True),
            (6, 1): (THEIRS, False),
            (6, 3): (THEIRS, False),
            (6, 5): (THEIRS, False),
            (6, 7): (THEIRS, False),
            (5, 0): (THEIRS, False),
            (5, 2): (THEIRS, False),
            (5, 4): (THEIRS, False),
            (5, 6): (THEIRS, False),
            (2, 3): (THEIRS, False),
        }
    )
    ranked = how_to_hold_what_is_already_won(
        board,
        acts=ACTS,
        holding=back_row_whole,
        parts_of=parts_of,
        without=without,
        step=MINE_MOVES,
        their_acts=ACTS,
        their_step=THEIR_MOVES,
        theirs=theirs,
    )
    assert ranked
    assert ranked[0].safe, ranked[0].describe()
    risky = [one for one in ranked if one.exposes]
    assert risky, "a move that hangs the crowned piece exists and should be found"
    assert ranked.index(risky[0]) > ranked.index(ranked[0])
    # Stepping to (3, 4) puts the crowned piece under the man on (2, 3) with an
    # empty square behind it. Stepping to (3, 6) does not.
    hung = next(one for one in ranked if one.act == (4, 5, 3, 4))
    assert not hung.safe and hung.exposes
    fine = next(one for one in ranked if one.act == (4, 5, 3, 6))
    assert fine.safe
    assert ranked.index(fine) < ranked.index(hung)


def test_a_free_taking_is_still_taken() -> None:
    """Caution is not refusing everything. It is refusing what costs her."""
    board = board_of(
        {
            (7, 0): (MINE, False),
            (7, 2): (MINE, False),
            (7, 4): (MINE, False),
            (7, 6): (MINE, False),
            (4, 3): (MINE, True),
            (3, 2): (THEIRS, False),
        }
    )
    ranked = how_to_hold_what_is_already_won(
        board,
        acts=ACTS,
        holding=back_row_whole,
        parts_of=parts_of,
        without=without,
        step=MINE_MOVES,
        their_acts=ACTS,
        their_step=THEIR_MOVES,
        theirs=theirs,
    )
    assert ranked[0].takes_from_them == 1, ranked[0].describe()
    assert ranked[0].act == (4, 3, 2, 1)


def test_with_nothing_to_gain_she_does_the_smallest_thing() -> None:
    board = board_of(
        {
            (7, 0): (MINE, False),
            (7, 2): (MINE, False),
            (7, 4): (MINE, False),
            (7, 6): (MINE, False),
            (4, 3): (MINE, True),
            (0, 7): (THEIRS, False),
        }
    )
    ranked = how_to_hold_what_is_already_won(
        board,
        acts=ACTS,
        holding=back_row_whole,
        parts_of=parts_of,
        without=without,
        step=MINE_MOVES,
        their_acts=ACTS,
        their_step=THEIR_MOVES,
        theirs=theirs,
    )
    best = ranked[0]
    assert best.takes_from_them == 0
    assert best.changes == min(one.changes for one in ranked if one.safe)
    assert best.keeps_it


def test_none_of_this_needs_a_board() -> None:
    """A budget. What she cannot spend is what the promise rests on."""
    kept = {"deposit": 300, "rent": 900, "fun": 120}

    def promise_kept(purse: dict) -> bool:
        return purse.get("rent", 0) >= 900

    precious = what_she_cannot_afford_to_lose(
        kept,
        holding=promise_kept,
        parts_of=lambda purse: list(purse),
        without=lambda purse, name: {k: v for k, v in purse.items() if k != name},
    )
    assert precious == ("rent",)

    def spend(purse: dict, name: str) -> dict | None:
        if name not in purse:
            return None
        return {k: v for k, v in purse.items() if k != name}

    risk = what_an_act_risks(
        kept,
        "fun",
        holding=promise_kept,
        parts_of=lambda purse: list(purse),
        without=lambda purse, name: {k: v for k, v in purse.items() if k != name},
        step=spend,
    )
    assert risk is not None and risk.safe
