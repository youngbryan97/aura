"""She plays a game of pieces she was never told the rules of: she sees the board, learns how the pieces move from what
the game offers and what she watches, follows every move, and plays to the end.

The games here are hidden worlds (tests/fixtures/a_game_of_pieces_to_learn.py): chess, because published move counts
check the vocabulary can hold a real game exactly, and a game nobody plays, so a learner that has quietly come to know
chess shows it by failing there.
"""
from __future__ import annotations

import asyncio
import random

import pytest

import core.agency.playing_pieces_by_kind as playing
from core.agency.how_the_pieces_move import HowThePiecesMove
from core.agency.pieces_by_their_kind import FIRST, after, moves, outcome
from core.agency.searching_by_kind import described, placed
from core.perception.pieces_by_their_look import Looks, squares_of_one_ground
from tests.fixtures.a_board_application import BoardApplication
from tests.fixtures.a_game_of_pieces_to_learn import CHESS, NOBODYS_GAME, from_fen

#: Published move counts (perft): the position, and the number of move sequences of each length from it.
PUBLISHED = (
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -", (20, 400, 8902)),
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -", (48, 2039)),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -", (14, 191, 2812)),
    ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", (6, 264, 9467)),
    ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", (44, 1486)),
)


def _perft(family, position, depth: int) -> int:
    legal = moves(family, position)
    if depth == 1:
        return len(legal)
    return sum(_perft(family, after(family, position, m), depth - 1) for m in legal)


@pytest.mark.parametrize("fen,counts", PUBLISHED)
def test_the_vocabulary_holds_a_real_game_exactly(fen: str, counts: tuple[int, ...]) -> None:
    position = from_fen(fen)
    assert tuple(_perft(CHESS, position, d) for d in range(1, len(counts) + 1)) == counts


@pytest.mark.parametrize("fen,counts", PUBLISHED)
def test_the_compiled_search_makes_the_same_moves(fen: str, counts: tuple[int, ...]) -> None:
    engine = pytest.importorskip("aura_board_search")
    position = from_fen(fen)
    assert engine.perft(described(CHESS), placed(position), len(counts)) == counts[-1]


def _learned_by_playing(hidden, plies: int, seed: int) -> HowThePiecesMove:
    rng = random.Random(seed)
    position = hidden.the_start()
    learned = HowThePiecesMove.from_the_start(hidden.rows, hidden.columns, hidden.start)
    for _ in range(plies):
        legal = moves(hidden, position)
        if not legal or outcome(hidden, position, legal):
            position = hidden.the_start()
            continue
        if position.turn == FIRST:
            family = learned.family()
            for square, (side, _kind) in list(position.cells.items()):
                if side == FIRST:
                    learned.offered(position, square, {m.end for m in legal if m.start == square}, family=family)
        move = rng.choice(legal)
        learned.watched(position, move)
        position = after(hidden, position, move)
    return learned


def _agreement(hidden, learned_family, positions: int, seed: int) -> float:
    rng = random.Random(seed)
    position, exact, tested = hidden.the_start(), 0, 0
    for _ in range(positions):
        legal = moves(hidden, position)
        if not legal or outcome(hidden, position, legal):
            position = hidden.the_start()
            continue
        tested += 1
        exact += {(m.start, m.end, m.becomes) for m in legal} == {
            (m.start, m.end, m.becomes) for m in moves(learned_family, position)}
        position = after(hidden, position, rng.choice(legal))
    return exact / max(1, tested)


@pytest.mark.parametrize("hidden", [CHESS, NOBODYS_GAME], ids=["chess", "nobodys_game"])
def test_she_learns_how_the_pieces_move_from_the_game_alone(hidden) -> None:
    learned = _learned_by_playing(hidden, plies=300, seed=1)
    family = learned.family()
    assert _agreement(hidden, family, positions=200, seed=7) >= 0.9
    royal = [k for k, kind in family.kinds.items() if kind.royal]
    assert royal == [k for k, kind in hidden.kinds.items() if kind.royal]


def test_she_holds_no_game_of_her_own() -> None:
    import core.agency.pieces_by_their_kind as vocabulary

    assert not hasattr(vocabulary, "FAMILIES")
    assert not hasattr(vocabulary, "a_family_whose_start_this_is")


def test_she_sees_the_board_the_application_draws() -> None:
    app = BoardApplication(CHESS, seed=3)
    picture = app.draw()
    lattice = squares_of_one_ground(picture)
    assert lattice is not None and (lattice.rows, lattice.columns) == (8, 8)
    looks = Looks()
    seen = looks.read(picture, lattice)
    assert len(looks.kinds) == len(CHESS.kinds)
    assert len(seen.occupied()) == len(CHESS.start)


def _played(hidden, seconds: float, she_goes_first: bool = True) -> tuple[dict, BoardApplication, list[str]]:
    playing.STILL_S, playing.OFFERS_SHOW_S, playing.WAITS_FOR_THE_FIRST_MOVE_S = 0.02, 0.01, 1.0
    app = BoardApplication(hidden, she_goes_first=she_goes_first, seed=5)
    said: list[str] = []

    async def run() -> dict:
        seat = playing.a_board_of_two_sides_at_its_start(app.draw())
        assert seat is not None
        stop = asyncio.Event()
        answering = asyncio.create_task(app.answers(stop))
        try:
            return await playing.play_pieces_by_kind(seat, app.look, app.click, seconds=seconds, say=said.append,
                                                     thinks_s=0.1)
        finally:
            stop.set()
            await answering

    return asyncio.run(run()), app, said


def test_she_follows_and_plays_a_game_she_was_never_told(monkeypatch: pytest.MonkeyPatch) -> None:
    came, app, said = _played(NOBODYS_GAME, seconds=25.0)
    assert came["ended"] != "lost track of the board", said[-3:]
    assert came["moves"] >= 5
    # Every move she made is one the application made: she never moved what the game would not move.
    assert len(app.made) >= 2 * came["moves"] - 1


def test_she_finds_out_who_goes_first(monkeypatch: pytest.MonkeyPatch) -> None:
    came, app, said = _played(NOBODYS_GAME, seconds=8.0, she_goes_first=False)
    assert "They went first." in said
    assert came["ended"] != "lost track of the board"
