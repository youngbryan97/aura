"""A game that offers her a move is not over, whatever her rules say, and what it offers tells her what does not attack.

LIVE 23 November 2026, the first game after far squares read once were kept as leaps (164f17a72): her royal kind had
been offered six and seven squares forward in one misread look, and never seen to pass over a piece. Kept as leaps,
their royal kind leapt the length of the board and stood attacking hers through both rows of pieces at the start. Her
rules left her three moves, the game refused all three, and she said "I have no move left, with my piece that must be
kept safe attacked: I lost." In the same turn the game had offered her pawns' moves, each leaving her royal kind so
attacked by her rules, and none of those offers counted against the attack.
"""
from __future__ import annotations

import asyncio

import core.agency.playing_pieces_by_kind as playing
import core.agency.searching_by_kind as searching
from core.agency.how_the_pieces_move import ROYAL_AFTER, HowThePiecesMove
from core.agency.pieces_by_their_kind import FIRST, Position, in_check, moves
from tests.fixtures.a_board_application import BoardApplication
from tests.fixtures.a_game_of_pieces_to_learn import CHESS, NOBODYS_GAME, from_fen

E1 = (0, 4)
AROUND = {(1, 4), (1, 3), (1, 5), (0, 3), (0, 5)}
OPEN_FILE = from_fen("k7/8/8/8/8/8/8/4K3 w - -")


def _royal_read_far_once() -> HowThePiecesMove:
    rules = HowThePiecesMove.from_the_start(CHESS.rows, CHESS.columns, CHESS.start)
    for _ in range(3):
        rules.offered(OPEN_FILE, E1, AROUND, family=CHESS)
    rules.offered(OPEN_FILE, E1, AROUND | {(6, 4), (7, 4)}, family=CHESS)  # a misreading: e7 and e8 taken for offered
    rules.kept_off_attacked["K"] = ROYAL_AFTER + 1
    return rules


def test_a_far_square_never_reached_over_a_piece_attacks_nothing_through_pieces() -> None:
    rules = _royal_read_far_once()
    start = Position(cells=dict(CHESS.start), turn=FIRST, unmoved=frozenset(CHESS.start))
    assert not in_check(rules.family(), start, FIRST)
    # Along a clear file it still may: the evidence is that far, and the look that settles it is still to come.
    facing = from_fen("4k3/8/8/8/8/8/P7/4K3 w - -")
    assert in_check(rules.family(), facing, FIRST)


def test_a_move_the_game_offers_counts_against_an_attack_her_rules_are_not_sure_of() -> None:
    rules = _royal_read_far_once()
    facing = from_fen("4k3/8/8/8/8/8/P7/4K3 w - -")
    # The pawn picked up: the game offers its steps, though by her rules each leaves her king attacked from e8.
    rules.offered(facing, (1, 0), {(2, 0), (3, 0)}, family=rules.family())
    assert not in_check(rules.family(), facing, FIRST)
    assert moves(rules.family(), facing)
    # Going that far is not ruled out by it: only taking.
    assert [s.may for s in rules.family().kinds["K"].steps if s.offsets == ((1, 0),) and s.rides == 7] == ["moves"]


def test_an_offer_of_nothing_withholds_nothing() -> None:
    rules = HowThePiecesMove.from_the_start(CHESS.rows, CHESS.columns, CHESS.start)
    start = Position(cells=dict(CHESS.start), turn=FIRST, unmoved=frozenset(CHESS.start))
    rules.offered(start, (1, 0), {(2, 0), (3, 0)}, family=CHESS)
    before = {d: (set(w.withheld), set(w.withheld_unmoved)) for d, w in rules.seen["P"].items()}
    # A board not yet taking moves: every piece picked up is offered nothing.
    rules.offered(start, (1, 0), set(), family=CHESS)
    assert {d: (set(w.withheld), set(w.withheld_unmoved)) for d, w in rules.seen["P"].items()} == before


def test_her_rules_leaving_her_no_move_never_end_a_game_that_offers_one(monkeypatch) -> None:
    monkeypatch.setattr(playing, "STILL_S", 0.02)
    monkeypatch.setattr(playing, "OFFERS_SHOW_S", 0.01)
    monkeypatch.setattr(playing, "WAITS_FOR_THE_FIRST_MOVE_S", 0.5)
    monkeypatch.setattr(playing._Rehearsing, "start", lambda self, pool, family: None)
    real_search = searching.search
    searches = {"n": 0}

    def nothing_on_her_first_look(*args, **kwargs):
        # Her rules wrong about the board: the first look finds her no move, as every move she had was refused.
        searches["n"] += 1
        if searches["n"] == 1:
            return searching.Found(None, 0, 0, 0, [])
        return real_search(*args, **kwargs)

    monkeypatch.setattr(searching, "search", nothing_on_her_first_look)
    app = BoardApplication(NOBODYS_GAME, seed=5)
    said: list[str] = []

    async def run() -> dict:
        seat = playing.a_board_of_two_sides_at_its_start(app.draw())
        stop = asyncio.Event()
        answering = asyncio.create_task(app.answers(stop))
        try:
            return await playing.play_pieces_by_kind(seat, app.look, app.click, seconds=4.0, say=said.append,
                                                     thinks_s=0.1)
        finally:
            stop.set()
            await answering

    came = asyncio.run(run())
    assert searches["n"] >= 1
    assert came["ended"] not in ("lost", "over"), (came["ended"], said)
    assert came["moves"] >= 1, said
    assert any(line.endswith("the move the game offers me.") for line in said), said


def _seen_one_square_forward_often() -> HowThePiecesMove:
    rules = HowThePiecesMove.from_the_start(CHESS.rows, CHESS.columns, CHESS.start)
    for _ in range(20):
        rules.offered(OPEN_FILE, E1, AROUND, family=CHESS)  # one square forward, the squares beyond withheld
    return rules


#: Her royal kind boxed in on its file, one of hers in front and one of theirs before that, and a look misread as
#: offering the square of theirs and the one beyond it: a take and a jump that never happened.
CROWDED = from_fen("k7/8/8/8/8/4p3/4P3/4K3 w - -")
MISREAD_OVER_A_PIECE = (AROUND - {(1, 4)}) | {(2, 4), (3, 4)}


def test_one_look_against_many_does_not_make_a_way_sure() -> None:
    """LIVE 23 November 2026, the game after: one misread look had her royal kind take two and three squares forward,
    and its twenty sightings one square forward made the three-square way one she was sure of."""
    rules = _seen_one_square_forward_often()
    rules.offered(CROWDED, E1, MISREAD_OVER_A_PIECE, family=CHESS)
    forward = [(s.rides, s.sure) for s in rules.family().kinds["K"].steps if s.offsets == ((1, 0),)]
    assert (1, True) in forward and not any(sure for rides, sure in forward if rides != 1), forward


def test_a_far_square_is_a_leap_only_where_a_jump_was_seen_that_far() -> None:
    """The same game: the jump seen at three made its far squares, six and seven, each read once with nothing between,
    leaps over every piece again, and their royal kind attacked hers at the start."""
    from core.agency.pieces_by_their_kind import FIRST as HER_SIDE

    rules = _seen_one_square_forward_often()
    rules.offered(OPEN_FILE, E1, AROUND | {(6, 4), (7, 4)}, family=CHESS)  # far squares read once, nothing between
    rules.offered(CROWDED, E1, MISREAD_OVER_A_PIECE, family=CHESS)
    assert not [s for s in rules.family().kinds["K"].steps if s.offsets in (((6, 0),), ((7, 0),))]
    start = Position(cells=dict(CHESS.start), turn=HER_SIDE, unmoved=frozenset(CHESS.start))
    assert not in_check(rules.family(), start, HER_SIDE)


def test_an_answer_is_read_once_the_board_holds_still_and_never_from_another_pieces_marks() -> None:
    """LIVE 23 November 2026, game 31: of 55 answers read, 31 squares were ones the game never offered, most of them
    the marks of the piece looked at before. A board that redraws a look late is read again until the answer holds."""
    app = BoardApplication(CHESS, seed=3, lags_looks=1)
    seat = playing.a_board_of_two_sides_at_its_start(app.draw())
    picture = app.draw()

    async def probe(square):
        return await playing._offers_for(seat, app.look, app.click, picture, square)

    for square in ((1, 0), (0, 1), (1, 4), (0, 6)):
        read = asyncio.run(probe(square))
        true = {m.end for m in moves(CHESS, app.position) if m.start == square}
        assert read is None or read == true, (square, read, true)
