"""Finding the configuration somebody described, without being given it.

    using two pieces on either side of a gap as a pathway to "walk" up a line

Nothing below tells it that. It is handed a world, an act that happens to be
safe, and a way of taking things out of the world, and it works out what the
safety was resting on.
"""

from __future__ import annotations

from core.cognition.a_shape_that_makes_it_safe import (
    a_way_through,
    what_makes_it_safe,
    where_else_it_holds,
)
from tests.two_sided_game_support import (
    ACTS,
    MINE,
    MINE_MOVES,
    SIDE,
    THEIRS,
    THEIR_MOVES,
    board_of,
    parts_of,
    without,
)

# Three of hers abreast, one of theirs able to come at the square in front.
ABREAST = board_of(
    {
        (5, 2): (MINE, False),
        (5, 4): (MINE, False),
        (5, 6): (MINE, False),
        (3, 6): (THEIRS, False),
    }
)

WALK = (5, 6, 4, 5)


def _still_there(board, act) -> bool:
    """Whether the thing she just moved survives their reply."""
    after = MINE_MOVES(board, act)
    if after is None:
        return False
    landed = (act[2], act[3])
    for their_act in ACTS:
        theirs_after = THEIR_MOVES(after, their_act)
        if theirs_after is None:
            continue
        if not any((row, col) == landed for row, col, _, _ in theirs_after):
            return False
    return True


def _where(part) -> tuple[int, int]:
    return (part[0], part[1])


def _kind(part):
    return part[2]


def _about(act) -> tuple[int, int]:
    return (act[2], act[3])


def test_it_finds_the_two_pieces_either_side_of_the_gap() -> None:
    assert _still_there(ABREAST, WALK)
    shape = what_makes_it_safe(
        ABREAST,
        WALK,
        safe=_still_there,
        parts_of=parts_of,
        without=without,
        where_of=_where,
        kind_of=_kind,
        about=_about,
    )
    # One either side of the square being stepped into, both hers.
    assert shape.around == (((1, -1), MINE), ((1, 1), MINE)), shape.describe()
    assert shape.established, shape.why
    # The third piece is not part of it, and was not swept in for being nearby.
    assert shape.size == 2


def test_the_shape_is_looked_for_everywhere_else() -> None:
    shape = what_makes_it_safe(
        ABREAST, WALK, safe=_still_there, parts_of=parts_of, without=without,
        where_of=_where, kind_of=_kind, about=_about,
    )
    everywhere = [(row, col) for row in range(SIDE) for col in range(SIDE)]
    holds = where_else_it_holds(
        ABREAST, shape, places=everywhere, parts_of=parts_of, where_of=_where,
        kind_of=_kind,
    )
    assert sorted(holds) == [(4, 3), (4, 5)]


def test_squares_in_a_row_are_a_way_through_and_not_just_squares() -> None:
    shape = what_makes_it_safe(
        ABREAST, WALK, safe=_still_there, parts_of=parts_of, without=without,
        where_of=_where, kind_of=_kind, about=_about,
    )
    everywhere = [(row, col) for row in range(SIDE) for col in range(SIDE)]
    ways = a_way_through(
        ABREAST, shape, places=everywhere, parts_of=parts_of, where_of=_where,
        kind_of=_kind,
        next_to=lambda a, b: abs(a[0] - b[0]) <= 1 and abs(a[1] - b[1]) == 2,
    )
    assert ways == [((4, 3), (4, 5))]


def test_it_refuses_a_shape_it_cannot_stand_behind() -> None:
    """An act that is not safe has no shape making it safe, and says so."""
    exposed = board_of({(5, 6): (MINE, False), (3, 6): (THEIRS, False)})
    shape = what_makes_it_safe(
        exposed, WALK, safe=_still_there, parts_of=parts_of, without=without,
        where_of=_where, kind_of=_kind, about=_about,
    )
    assert not shape.around
    assert "not safe" in shape.why


def test_none_of_this_is_about_a_board() -> None:
    """A rota. What makes a shift safe to take is who else is on."""
    on = {("mon", "ana"), ("mon", "bo"), ("tue", "ana"), ("tue", "cy")}
    days = {"mon": 0, "tue": 1, "wed": 2}

    def covered(rota, shift) -> bool:
        day, _who = shift
        return sum(1 for when, _ in rota if when == day) >= 2

    shape = what_makes_it_safe(
        on,
        ("mon", "dee"),
        safe=covered,
        parts_of=lambda rota: sorted(rota),
        without=lambda rota, one: rota - {one},
        where_of=lambda one: (days[one[0]],),
        kind_of=lambda one: one[1],
        about=lambda shift: (days[shift[0]],),
    )
    assert {what for _, what in shape.around} == {"ana", "bo"}
