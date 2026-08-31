"""New actions from the ones she was given — the other side of the language work.

Everything built before this grows the language she describes with. A7 is the
same question on the action side, and the shape carries over exactly: a word was
a term over places in a thing of some size, an action is a term over states of a
world.
"""

from __future__ import annotations

from core.cognition.an_action_she_composed import (
    Doing,
    World,
    an_action_she_composed,
    every_doing,
    how_many_acts,
    read_back,
    what_it_does,
    written_down,
)


def _slide(row):
    kept = [one for one in row if one]
    out, at = [], 0
    while at < len(kept):
        if at + 1 < len(kept) and kept[at] == kept[at + 1]:
            out.append(kept[at] * 2)
            at += 2
        else:
            out.append(kept[at])
            at += 1
    return tuple(out + [0] * (len(row) - len(out)))


def _flipped(how):
    return lambda row: tuple(reversed(how(tuple(reversed(row)))))


def _a_board():
    return World(
        can_do={"left": _slide, "right": _flipped(_slide)},
        can_tell={"nothing on the left": lambda row: row[0] == 0},
    )


def _a_line():
    def step(by):
        return lambda at: max(0, min(9, at + by))

    return World(
        can_do={"one right": step(1), "one left": step(-1)},
        can_tell={"at the left wall": lambda at: at == 0},
    )


def _moves_somehow(row):
    return _slide(row) if _slide(row) != row else _flipped(_slide)(row)


PACKED = [(2, 4, 0, 0), (8, 2, 0, 0), (4, 8, 0, 0)]
LOOSE = [(0, 0, 2, 4), (0, 2, 0, 4), (0, 0, 8, 2), (2, 0, 0, 4)]


def test_she_arrives_at_repeating_something_until_it_settles():
    """The count comes from the world. "Three times" needs a three from
    somewhere and there is nowhere honest to get one."""
    line = _a_line()
    found = an_action_she_composed(
        line, [(3, 9), (7, 9), (0, 9)], held_out=[(1, 9), (5, 9), (8, 9)]
    )
    assert found is not None
    doing, worth = found
    assert doing.head == "until"
    assert what_it_does(doing, 2, line) == 9
    assert worth.keep_it


def test_she_arrives_at_a_recovery_for_an_act_that_did_nothing():
    """An action that does nothing is the commonest thing on a screen and the
    hardest to see. Making the recovery part of the grammar means she can build
    it rather than have it written for her."""
    board = _a_board()
    shown = [(one, _moves_somehow(one)) for one in [*PACKED[:2], *LOOSE[:2]]]
    held = [(one, _moves_somehow(one)) for one in [PACKED[2], LOOSE[2], LOOSE[3]]]
    found = an_action_she_composed(board, shown, held_out=held)
    assert found is not None
    doing, worth = found
    assert doing.head == "instead"
    assert worth.keep_it
    for state, wanted in held:
        assert what_it_does(doing, state, board) == wanted


def test_nothing_is_composed_where_one_key_already_does_it():
    """A composed action that stands in for nothing is a branch bought for
    nothing."""
    board = _a_board()
    assert (
        an_action_she_composed(
            board, [((0, 0, 2, 4), (2, 4, 0, 0))], held_out=[((0, 0, 8, 2), (8, 2, 0, 0))]
        )
        is None
    )


def test_the_same_algebra_serves_two_worlds_with_nothing_in_common():
    line, board = _a_line(), _a_board()
    on_a_line = an_action_she_composed(line, [(3, 9), (7, 9)], held_out=[(1, 9), (5, 9)])
    on_a_board = an_action_she_composed(
        board,
        [((2, 2, 2, 2), (8, 0, 0, 0)), ((4, 4, 4, 4), (16, 0, 0, 0))],
        held_out=[((8, 8, 8, 8), (32, 0, 0, 0))],
    )
    assert on_a_line is not None and on_a_board is not None
    assert on_a_line[0].head == on_a_board[0].head == "until"


def test_a_primitive_is_not_a_new_action():
    board = _a_board()
    for doing in every_doing(board, deepest=1):
        if doing.head == "do":
            assert doing.value in board.can_do


def test_repetition_that_never_settles_is_refused_rather_than_hung():
    """A state that keeps changing forever cannot be reached by waiting."""
    forever = World(can_do={"on": lambda at: at + 1}, can_tell={})
    doing = Doing(head="until", parts=(Doing(head="do", value="on"),))
    try:
        what_it_does(doing, 0, forever)
    except ValueError as exc:
        assert "never stopped" in str(exc)
    else:  # pragma: no cover - the guard is the point
        raise AssertionError("it should have refused")


def test_acts_are_counted_in_key_presses():
    line = _a_line()
    doing = Doing(head="until", parts=(Doing(head="do", value="one right"),))
    assert how_many_acts(doing, 7, line) == 3  # two steps and the one that proves it


def test_an_action_she_composed_survives_the_process():
    board = _a_board()
    doing = Doing(
        head="instead",
        parts=(Doing(head="do", value="left"), Doing(head="do", value="right")),
    )
    back = read_back(written_down(doing))
    assert back == doing
    assert what_it_does(back, (2, 4, 0, 0), board) == (0, 0, 2, 4)


def test_nothing_is_composed_from_nothing():
    assert an_action_she_composed(_a_board(), []) is None
