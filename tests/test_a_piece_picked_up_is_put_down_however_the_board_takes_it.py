"""A piece she picks up to see its offers is put down again, however the board application puts pieces down.

Many board applications leave a piece picked up on a second click, since a click there may begin a drag. LIVE 20
November 2026 her record of En Croissant's board held sixteen kinds of piece where it has six, most of them the marks a
picked-up piece leaves; a piece left picked up shows its marks to the next piece she picks up, as offers it never had.
"""
from __future__ import annotations

import asyncio

import core.agency.playing_pieces_by_kind as playing
from tests.fixtures.a_board_application import BoardApplication
from tests.fixtures.a_game_of_pieces_to_learn import NOBODYS_GAME


def _played(monkeypatch, keeps_it_picked: bool) -> tuple[dict, BoardApplication, list[str], int]:
    monkeypatch.setattr(playing, "STILL_S", 0.02)
    monkeypatch.setattr(playing, "OFFERS_SHOW_S", 0.01)
    monkeypatch.setattr(playing, "WAITS_FOR_THE_FIRST_MOVE_S", 0.5)
    monkeypatch.setattr(playing._Rehearsing, "start", lambda self, pool, family: None)
    # Searched by a count of positions, not by the clock: the same game every time, however loaded the machine. A
    # search by time looked less deep under load, played another game, and in it met a move this game really refuses.
    import core.agency.searching_by_kind as searching

    by_time = searching.search
    monkeypatch.setattr(searching, "search", lambda family, position, **how: by_time(
        family, position, **{**how, "nodes": 20_000, "threads": 1, "soft": None}))
    app = BoardApplication(NOBODYS_GAME, seed=5, keeps_it_picked=keeps_it_picked)
    said: list[str] = []
    seat_holder: list = []

    async def run() -> dict:
        seat = playing.a_board_of_two_sides_at_its_start(app.draw())
        assert seat is not None
        seat_holder.append(seat)
        stop = asyncio.Event()
        answering = asyncio.create_task(app.answers(stop))
        try:
            return await playing.play_pieces_by_kind(seat, app.look, app.click, seconds=12.0, say=said.append,
                                                     thinks_s=0.1)
        finally:
            stop.set()
            await answering

    came = asyncio.run(run())
    return came, app, said, len(seat_holder[0].looks.kinds)


def test_a_board_that_keeps_a_piece_picked_up_teaches_her_no_false_offers(monkeypatch) -> None:
    from core.agency.pieces_by_their_kind import moves

    came, app, said, kinds = _played(monkeypatch, keeps_it_picked=True)
    assert came["moves"] >= 4, said[-4:]
    # A move the game would not take is no false offer where the game's own rules do not allow it there: a move that
    # leaves a royal piece attacked, the game ending in it. Only one the game allows, refused, is a false lesson.
    allowed = {f"{'abcdefgh'[m.start[1]]}{m.start[0] + 1}{'abcdefgh'[m.end[1]]}{m.end[0] + 1}"
               for m in moves(app.family, app.position)}
    refused = {name[:2] + name[-2:] for name in came["acts"]["refused"]}
    assert not refused & allowed, (refused & allowed, [line for line in said if "didn't" in line])
    assert kinds == len(NOBODYS_GAME.kinds), f"{kinds} kinds of piece read on a board of {len(NOBODYS_GAME.kinds)}"
