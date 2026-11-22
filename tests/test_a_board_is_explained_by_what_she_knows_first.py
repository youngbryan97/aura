"""What she reads of a board changing asks her to believe as little as it can.

LIVE-like 22 November 2026 she learned draughts from scratch on a board drawn for tests. Each rule below fixes one way
that learning went wrong: a board read once wrongly, and explained by a move nobody made, taught her men to step
sideways; a man of hers stepping forward and taken over at once was read as a two-square step taking a piece beside
where it landed; a take the game refused was put down to the piece it left open; the two sides' men, drawn a little
differently by colour, never learned from each other; a dark man crowned was read as a light man; and her own man
crowned was held a man because her board already had it there.
"""
from __future__ import annotations

from dataclasses import replace

import core.agency.playing_pieces_by_kind as playing
from core.agency.how_the_pieces_move import HowThePiecesMove, a_move_her_rules_allow, moves_seen
from core.agency.pieces_by_their_kind import FIRST, SECOND, Family, Kind, Move, Position, Step
from tests.fixtures.a_game_of_pieces_to_learn import CHESS, DRAUGHTS


def _board(cells: dict, turn: int = FIRST) -> Position:
    return Position(cells=cells, turn=turn, unmoved=frozenset())


def _steps(learned: HowThePiecesMove, kind: str) -> set:
    return {(s.offsets, s.may) for s in learned.family().kinds[kind].steps}


def test_a_move_her_rules_do_not_allow_teaches_her_only_once_seen_again() -> None:
    learned = HowThePiecesMove.from_the_start(8, 8, DRAUGHTS.start)
    start = Position(cells=DRAUGHTS.start, turn=SECOND, unmoved=frozenset())
    learned.watched(start, Move((5, 1), (4, 0)), allowed=False)
    assert not learned.family().kinds["M"].steps, "one sighting of a move nobody may have made teaches nothing"
    learned.watched(start, Move((5, 3), (4, 2)), allowed=False)
    assert (((1, 1),), "either") in _steps(learned, "M")


def test_a_take_refused_by_a_kind_never_seen_taking_says_it_does_not_take() -> None:
    learned = HowThePiecesMove.from_the_start(8, 8, DRAUGHTS.start)
    learned.offered(_board({(2, 2): (FIRST, "M")}), (2, 2), {(3, 1), (3, 3)})
    assert (((1, 1),), "either") in _steps(learned, "M")
    # Taking on the next square leaves her man where their man beside could take it: left open, but never seen taking.
    position = _board({(2, 2): (FIRST, "M"), (3, 3): (SECOND, "M"), (4, 2): (SECOND, "M")})
    learned.refused(position, Move((2, 2), (3, 3), taken_at=(3, 3)), family=learned.family())
    assert learned.refusals == 0
    assert (((1, 1),), "moves") in _steps(learned, "M")


def test_a_move_her_rules_allow_takes_what_her_rule_takes() -> None:
    position = _board({(1, 7): (FIRST, "M"), (2, 6): (SECOND, "M"), (3, 7): (SECOND, "M")})
    assert a_move_her_rules_allow(DRAUGHTS, position, Move((1, 7), (3, 5), taken_at=(2, 6)))
    assert not a_move_her_rules_allow(DRAUGHTS, position, Move((1, 7), (3, 5), taken_at=(3, 7)))


def _kings_that_take_over_only_forward() -> Family:
    king = Kind("K", steps=(Step(((1, 1), (1, -1), (-1, 1), (-1, -1)), may="moves"),
                            Step(((2, 2), (2, -2)), may="takes over")))
    return Family(name="kings", rows=8, columns=8, kinds={"K": king})


def test_a_board_is_explained_by_moves_of_the_sorts_she_knows() -> None:
    family = _kings_that_take_over_only_forward()
    before = _board({(6, 2): (FIRST, "K"), (4, 2): (SECOND, "K")})
    # Her king stepped, and theirs took it over backwards, which her rules do not have; or her king stepped the other
    # way and theirs took it from beside where it landed, which no kind does.
    seen = moves_seen(before, {(6, 4): (SECOND, "K")}, FIRST, family)
    assert seen is not None
    mine, theirs = seen
    assert (mine.start, mine.end) == ((6, 2), (5, 3))
    assert (theirs.start, theirs.end, theirs.taken()) == ((4, 2), (6, 4), ((5, 3),))


def test_her_move_takes_the_kind_the_board_shows_where_it_landed() -> None:
    men = replace(DRAUGHTS, kinds={"M": replace(DRAUGHTS.kinds["M"], becomes=()), "K": DRAUGHTS.kinds["K"]})
    before = _board({(6, 0): (FIRST, "M"), (4, 4): (SECOND, "M")})
    seen = moves_seen(before, {(7, 1): (FIRST, "K"), (3, 3): (SECOND, "M")}, FIRST, men)
    assert seen is not None and seen[0].becomes == "K", seen


def _drawn_by_colour() -> dict:
    """Draughts as a board drawing each side's men a little differently: a look of each side's own."""
    return {square: (side, "a" if side == FIRST else "b") for square, (side, _kind) in DRAUGHTS.start.items()}


def test_two_sides_that_begin_as_images_of_each_other_learn_from_each_other() -> None:
    learned = HowThePiecesMove.from_the_start(8, 8, _drawn_by_colour())
    assert learned.counterparts() == {"a": ("b", False), "b": ("a", False)}
    position = _board({(5, 5): (SECOND, "b"), (4, 4): (FIRST, "a"), (2, 2): (FIRST, "a")}, turn=SECOND)
    learned.watched(position, Move((5, 5), (1, 1), taken_at=(4, 4), via=((3, 3),), also_taken=((2, 2),)))
    family = learned.family()
    assert family.kinds["a"].takes_again
    assert any(step.may == "takes over" for step in family.kinds["a"].steps)


def test_in_chess_no_kind_is_the_image_of_another() -> None:
    # Turned round, the board puts one side's queen on the other's king: both sides have both looks, so neither is
    # the other's image.
    assert HowThePiecesMove.from_the_start(8, 8, CHESS.start).counterparts() == {}


def test_a_look_only_the_other_side_has_held_is_read_as_unread() -> None:
    learned = HowThePiecesMove.from_the_start(8, 8, _drawn_by_colour())
    others = playing._only_the_others(learned)
    shown = {(0, 4): (SECOND, "a"), (1, 1): (FIRST, "a")}
    read = playing._read_against(shown, {(1, 1): (FIRST, "a")}, others)
    assert read[(0, 4)] == (SECOND, None) and read[(1, 1)] == (FIRST, "a")


def test_her_piece_shown_changed_where_her_move_landed_became_that_kind() -> None:
    learned = HowThePiecesMove.from_the_start(8, 8, DRAUGHTS.start)
    game = playing._Game(learned=learned, believed=_board({(6, 0): (FIRST, "M"), (5, 5): (SECOND, "M")}))
    game.follow(Move((6, 0), (7, 1)))
    game.became_on_landing({(7, 1): (FIRST, "K"), (5, 5): (SECOND, "M")})
    assert game.believed.cells[(7, 1)] == (FIRST, "K")
    assert learned.becomes["M"] == ["K"]


def _castled(fen: str, mover: tuple, partner: tuple) -> tuple[Position, dict]:
    from tests.fixtures.a_game_of_pieces_to_learn import from_fen

    before = from_fen(fen)
    cells = dict(before.cells)
    for start, end in (mover, partner):
        cells[end] = cells.pop(start)
    return before, cells


def test_a_long_crossing_is_read_with_the_royal_kind_as_the_one_that_moved() -> None:
    # LIVE 22 November 2026 her long crossing, e1-c1 with a1-d1, went unexplained: the rook went three squares to the
    # king's two, was taken for the mover, and failed a crossing's shape.
    from core.agency.how_the_pieces_move import a_move_seen

    before, cells = _castled("r3k2r/8/8/8/8/8/8/R3K2R w KQkq -", ((0, 4), (0, 2)), ((0, 0), (0, 3)))
    assert moves_seen(before, cells, FIRST, CHESS) == [Move((0, 4), (0, 2), partner=((0, 0), (0, 3)))]
    assert a_move_seen(before, cells, FIRST) == Move((0, 4), (0, 2), partner=((0, 0), (0, 3)))
    # The short one fits either piece as the mover: the royal kind is the one that moved.
    before, cells = _castled("r3k2r/8/8/8/8/8/8/R3K2R w KQkq -", ((0, 4), (0, 6)), ((0, 7), (0, 5)))
    assert moves_seen(before, cells, FIRST, CHESS) == [Move((0, 4), (0, 6), partner=((0, 7), (0, 5)))]
    assert a_move_seen(before, cells, FIRST, royal=frozenset({"K"})) == Move((0, 4), (0, 6), partner=((0, 7), (0, 5)))
