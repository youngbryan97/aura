"""A board in an application she has not played that begins as one she has played starts from what she learned there.

Every application draws its pieces its own way, and she names a kind by its look, so at a new application's board all
she had learned was lost and her first game there was played knowing nothing. What a board starts with says which kind
stands where (core/cognition/the_same_start_under_other_names.py): where two boards begin alike under other names, the
rules and worths she learned at one are her first guess at the other, each kind by the kind standing where it stands.

Her rows count from her own back row, so the side she sits at turns the board. A chess board turned half round begins
as it does unturned with the king and queen called by each other's names, so the start alone cannot say which way
round two boards are; who moved first can.
"""
from __future__ import annotations

import asyncio

import core.agency.playing_pieces_by_kind as playing
import core.runtime.what_she_learned as what_she_learned
import core.skills.screen_pursuit_board_of_pieces as board_of_pieces
from core.agency.how_the_pieces_move import HowThePiecesMove, turned_round_from
from core.agency.pieces_by_their_kind import FIRST, SECOND
from core.agency.searching_by_kind import Weights
from core.agency.what_a_kind_is_worth import Rehearsal
from core.cognition.the_same_start_under_other_names import the_same_start
from tests.fixtures.a_board_application import BoardApplication
from tests.fixtures.a_game_of_pieces_to_learn import CHESS, NOBODYS_GAME

#: Chess's kinds as another application's looks might name them.
DIGITS = {"P": "0", "R": "1", "N": "2", "B": "3", "Q": "4", "K": "5"}


def _renamed(start: dict, names: dict) -> dict:
    return {square: (side, names[kind]) for square, (side, kind) in start.items()}


def test_one_beginning_under_two_sets_of_names_is_lined_up() -> None:
    assert the_same_start(_renamed(CHESS.start, DIGITS), CHESS.start) == {v: k for k, v in DIGITS.items()}


def test_a_different_beginning_is_not() -> None:
    moved = dict(CHESS.start)
    moved[(2, 4)] = moved.pop((1, 4))
    assert the_same_start(moved, CHESS.start) is None
    # Two kinds in one beginning standing where one kind stands in the other.
    assert the_same_start(_renamed(CHESS.start, {**DIGITS, "B": "2"}), CHESS.start) is None
    sides_swapped = {square: (SECOND if side == FIRST else FIRST, kind) for square, (side, kind) in CHESS.start.items()}
    assert the_same_start(sides_swapped, CHESS.start) is None


def test_chess_turned_round_lines_up_with_its_king_and_queen_swapped() -> None:
    turned = turned_round_from(CHESS.start, 8, 8)
    assert turned[(0, 3)] == (FIRST, "K") and turned[(7, 3)] == (SECOND, "K")
    swapped = the_same_start(turned, CHESS.start)
    assert swapped is not None and swapped["K"] == "Q" and swapped["Q"] == "K"


def _kept_chess(first_was_hers: bool | None) -> tuple[HowThePiecesMove, Rehearsal]:
    learned = HowThePiecesMove.from_the_start(8, 8, CHESS.start)
    learned.first_was_hers = first_was_hers
    learned.kept_off_attacked = {"K": 6}
    steps = learned._of("K", (1, 1))
    steps.offered, steps.moved_to_empty = {1}, 4
    table = [[0.0] * 8 for _ in range(8)]
    table[0][6] = 30.0  # her king tucked away on her right
    rehearsal = Rehearsal(weights=Weights(worth={"Q": 900.0, "K": 0.0}, squares={"K": table}), earned=["worth"],
                          lengths=[80], royal=["K"])
    return learned, rehearsal


def _kept_in(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(what_she_learned, "_kept_in", lambda: tmp_path)


def test_a_board_she_sits_at_the_other_side_of_is_turned_round(monkeypatch, tmp_path) -> None:
    _kept_in(monkeypatch, tmp_path)
    learned, rehearsal = _kept_chess(first_was_hers=True)
    played = asyncio.run(_a_seat(BoardApplication(NOBODYS_GAME, seed=1)))
    board_of_pieces._keep_what_she_learned("First Board", played.looks, learned, rehearsal)
    # At the new board she moves second: in her own rows and columns the board is turned half round.
    new = _renamed(turned_round_from(CHESS.start, 8, 8), DIGITS)
    lent = board_of_pieces._borrowed_from_a_board_like_it("Second Board", new, 8, 8, first_is_hers=False)
    assert lent is not None
    rules, measured, where = lent
    assert where == "First Board"
    assert rules.kept_off_attacked == {"5": 6} and rules.seen["5"][(1, 1)].moved_to_empty == 4
    assert measured.royal == ["5"] and measured.weights.worth == {"4": 900.0, "5": 0.0}
    # Seen from the other side, her right is the first side's left.
    assert measured.weights.squares["5"][0][1] == 30.0
    assert rules.first_was_hers is False
    # Sitting where she sat, the same board lines up unturned.
    straight = board_of_pieces._borrowed_from_a_board_like_it("Second Board", _renamed(CHESS.start, DIGITS), 8, 8,
                                                              first_is_hers=True)
    assert straight is not None and straight[0].kept_off_attacked == {"5": 6}


def test_a_record_that_does_not_say_where_she_sat_lends_nothing_it_cannot_tell(monkeypatch, tmp_path) -> None:
    _kept_in(monkeypatch, tmp_path)
    learned, rehearsal = _kept_chess(first_was_hers=None)
    played = asyncio.run(_a_seat(BoardApplication(NOBODYS_GAME, seed=1)))
    board_of_pieces._keep_what_she_learned("First Board", played.looks, learned, rehearsal)
    assert board_of_pieces._borrowed_from_a_board_like_it("Second Board", _renamed(CHESS.start, DIGITS), 8, 8,
                                                          first_is_hers=True) is None


async def _a_seat(app: BoardApplication):
    return playing.a_board_of_two_sides_at_its_start(app.draw())


def test_a_new_application_borrows_from_the_one_she_played(monkeypatch, tmp_path) -> None:
    _kept_in(monkeypatch, tmp_path)
    played = asyncio.run(_a_seat(BoardApplication(NOBODYS_GAME, seed=1)))
    new = asyncio.run(_a_seat(BoardApplication(NOBODYS_GAME, seed=4)))
    assert played is not None and new is not None
    leaper, royal = played.start[(0, 1)][1], played.start[(0, 3)][1]
    learned = HowThePiecesMove.from_the_start(played.rows, played.columns, played.start)
    learned.first_was_hers = True
    leap = learned._of(leaper, (1, 1))
    leap.offered, leap.jumped, leap.moved_to_empty = {2}, 3, 3
    rehearsal = Rehearsal(weights=Weights(worth={leaper: 310.0}), earned=["worth"], lengths=[44, 52], royal=[royal])
    board_of_pieces._keep_what_she_learned("First Board", played.looks, learned, rehearsal)

    lent = board_of_pieces._borrowed_from_a_board_like_it("Second Board", new.start, new.rows, new.columns, True)
    assert lent is not None
    rules, measured, where = lent
    here_leaper, here_royal = new.start[(0, 1)][1], new.start[(0, 3)][1]
    assert where == "First Board" and rules.seen[here_leaper][(1, 1)].jumped == 3
    assert measured.weights.worth == {here_leaper: 310.0} and measured.royal == [here_royal]
    assert measured.lengths == [44, 52]
    # Its own record is never its own lender.
    assert board_of_pieces._borrowed_from_a_board_like_it("First Board", played.start, played.rows, played.columns,
                                                          True) is None
    # A board that begins otherwise borrows nothing.
    otherwise = dict(new.start)
    otherwise[(2, 3)] = otherwise.pop((1, 3))
    assert board_of_pieces._borrowed_from_a_board_like_it("Second Board", otherwise, new.rows, new.columns,
                                                          True) is None


def test_the_player_asks_once_it_knows_who_moves_first(monkeypatch) -> None:
    monkeypatch.setattr(playing, "STILL_S", 0.02)
    monkeypatch.setattr(playing, "OFFERS_SHOW_S", 0.01)
    monkeypatch.setattr(playing, "WAITS_FOR_THE_FIRST_MOVE_S", 0.5)
    monkeypatch.setattr(playing._Rehearsing, "start", lambda self, pool, family: None)
    app = BoardApplication(NOBODYS_GAME, seed=5)
    said: list[str] = []
    asked: list[bool] = []

    def borrow(start, first_is_hers):
        asked.append(first_is_hers)
        lent = HowThePiecesMove.from_the_start(7, 7, start)
        return lent, None, "First Board"

    async def run() -> dict:
        seat = playing.a_board_of_two_sides_at_its_start(app.draw())
        stop = asyncio.Event()
        answering = asyncio.create_task(app.answers(stop))
        try:
            return await playing.play_pieces_by_kind(seat, app.look, app.click, seconds=1.5, say=said.append,
                                                     thinks_s=0.05, borrow=borrow)
        finally:
            stop.set()
            await answering

    came = asyncio.run(run())
    assert asked == [True]
    assert "This board begins as the one in First Board does, so I'll start from what I learned there." in said
    assert came["learned"].first_was_hers is True
