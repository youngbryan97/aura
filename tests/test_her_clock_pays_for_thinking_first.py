"""Her clock goes to thinking before it goes to looking.

LIVE 21 November 2026, game 10 against Stockfish on a three-minute clock: of the 232 s her turns took, thinking had 60.
Picking the chosen piece up to see the game offer the move before making it took 47 s, though her rules had said what
the game would offer for turn after turn; and four late turns that read no clock were given four seconds of thought
where the last reading left her one or two.
"""
from __future__ import annotations

import asyncio
import logging

import core.agency.playing_pieces_by_kind as playing
from core.agency.how_the_pieces_move import HowThePiecesMove
from core.agency.pieces_by_their_kind import FIRST, Position
from tests.fixtures.a_board_application import BoardApplication
from tests.fixtures.a_game_of_pieces_to_learn import NOBODYS_GAME


def _a_game() -> playing._Game:
    start = dict(NOBODYS_GAME.start)
    return playing._Game(learned=HowThePiecesMove.from_the_start(7, 7, start),
                         believed=Position(cells=start, turn=FIRST, unmoved=frozenset(start)))


def test_she_looks_before_every_move_until_her_rules_earn_going_without() -> None:
    game = _a_game()
    assert all(playing._looks_first(game) for _ in range(5))
    for _ in range(6):
        game.far.it_was_where_she_said(1)
    looked = [playing._looks_first(game) for _ in range(12)]
    assert looked.count(False) > looked.count(True), looked
    # The first look that finds the game offering other than her rules said has her looking every move again.
    game.far.it_was_not(1)
    assert all(playing._looks_first(game) for _ in range(3))


def test_a_turn_whose_clock_cannot_be_read_has_the_last_share_it_could(monkeypatch, caplog) -> None:
    monkeypatch.setattr(playing, "STILL_S", 0.02)
    monkeypatch.setattr(playing, "OFFERS_SHOW_S", 0.01)
    monkeypatch.setattr(playing, "WAITS_FOR_THE_FIRST_MOVE_S", 0.5)
    monkeypatch.setattr(playing._Rehearsing, "start", lambda self, pool, family: None)
    monkeypatch.setattr(playing._Rehearsing, "moves_left", lambda self, made: 20)
    readings = {"n": 0}

    async def clock():
        # Read once for whose turn it is, twice more, and then never again.
        readings["n"] += 1
        return 4.0 - 0.1 * readings["n"] if readings["n"] <= 3 else None

    app = BoardApplication(NOBODYS_GAME, seed=5)

    async def run() -> dict:
        seat = playing.a_board_of_two_sides_at_its_start(app.draw())
        stop = asyncio.Event()
        answering = asyncio.create_task(app.answers(stop))
        try:
            return await playing.play_pieces_by_kind(seat, app.look, app.click, seconds=4.0, thinks_s=3.0,
                                                     clock=clock)
        finally:
            stop.set()
            await answering

    with caplog.at_level(logging.INFO, logger="Aura.PlayingPiecesByKind"):
        asyncio.run(run())
    shares = [record.getMessage().rsplit("was ", 1)[-1] for record in caplog.records
              if record.getMessage().startswith("her turn ")]
    assert len(shares) >= 2, shares
    assert "not on a clock" not in shares, shares


def test_the_first_piece_of_a_turn_waits_for_room_once_a_pick_up_is_timed() -> None:
    game = _a_game()
    assert playing._room_to_learn(game, spent=0.0, share=2.4, probes=0)
    game.timed_probe(1.6)
    assert not playing._room_to_learn(game, spent=0.0, share=2.4, probes=0)
    assert playing._room_to_learn(game, spent=0.0, share=5.0, probes=0)
    assert not playing._room_to_learn(game, spent=1.6, share=5.0, probes=1)


def test_one_look_that_shows_her_move_made_is_enough() -> None:
    from core.agency.pieces_by_their_kind import moves

    app = BoardApplication(NOBODYS_GAME, seed=5)

    async def run() -> tuple[bool, int]:
        picture = app.draw()
        seat = playing.a_board_of_two_sides_at_its_start(picture)
        game = playing._Game(learned=HowThePiecesMove.from_the_start(seat.rows, seat.columns, seat.start),
                             believed=Position(cells=dict(seat.start), turn=FIRST, unmoved=frozenset(seat.start)))
        move = next(one for one in moves(NOBODYS_GAME, NOBODYS_GAME.the_start()) if one.taken_at is None)
        await app.click(*seat.centre_share(move.start, picture))
        await app.click(*seat.centre_share(move.end, picture))
        looks = {"n": 0}

        async def look():
            looks["n"] += 1
            return await app.look()

        return await playing._comes_to_show(seat, look, game, move), looks["n"]

    shown, looks = asyncio.run(run())
    assert shown and looks == 1, looks


def test_she_always_looks_while_her_piece_that_must_be_kept_safe_is_attacked() -> None:
    game = _a_game()
    for _ in range(8):
        game.far.it_was_where_she_said(1)
    assert not all(playing._looks_first(game) for _ in range(4))
    assert all(playing._looks_first(game, pressed=True) for _ in range(4))


def test_her_clock_is_read_from_one_picture(monkeypatch) -> None:
    import core.perception.what_the_pixels_show as pixels
    import core.skills.screen_pursuit_board_of_pieces as board_of_pieces

    asked: dict = {}

    async def look_at_window(app, over=None, *, wait_for_stillness=True, **_kwargs):
        asked["wait"] = wait_for_stillness
        return {"ok": True, "layout": [{"text": "2:41", "center_y": 0.1}, {"text": "0:19.4", "center_y": 0.9}]}

    monkeypatch.setattr(pixels, "look_at_window", look_at_window)
    assert asyncio.run(board_of_pieces._her_clock("en-croissant")) == 19.4
    assert asked["wait"] is False
