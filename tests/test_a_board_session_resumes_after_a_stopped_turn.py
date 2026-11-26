from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from core.agency import playing_pieces_by_kind as playing
from core.agency.pieces_by_their_kind import SECOND, moves
from core.skills import screen_pursuit as pursuit
from core.skills import screen_pursuit_board_of_pieces as boards
from tests.fixtures.a_board_application import BoardApplication
from tests.fixtures.a_game_of_pieces_to_learn import NOBODYS_GAME
from tests.test_she_learns_a_game_of_pieces import _learned_by_playing


@pytest.mark.asyncio
async def test_a_stopped_turn_resumes_only_on_the_same_observed_board(monkeypatch) -> None:
    app = BoardApplication(NOBODYS_GAME, seed=5)
    rules = _learned_by_playing(NOBODYS_GAME, plies=180, seed=1)
    window = SimpleNamespace(owner="Fixture Board", pid=40, number=50, bounds=(0, 0, app.wide, app.tall))
    monkeypatch.setattr(boards, "_SUSPENDED_GAMES", {})
    monkeypatch.setattr(boards, "_ACTIVE_GAMES", set())
    monkeypatch.setattr(boards, "_what_she_kept", lambda _app: {"rules": rules})
    monkeypatch.setattr(boards, "_keep_what_she_learned", lambda *args: None)
    monkeypatch.setattr(playing, "STILL_S", 0.02)
    monkeypatch.setattr(playing, "OFFERS_SHOW_S", 0.01)
    monkeypatch.setattr(playing, "WAITS_FOR_THE_FIRST_MOVE_S", 0.25)
    monkeypatch.setattr(playing, "_the_most_a_look_may_take", lambda *_args: 0.25)
    monkeypatch.setattr(playing._Rehearsing, "start", lambda *args: None)
    monkeypatch.setattr("core.capabilities.window_server.window_of", lambda _app: window)

    async def pixels(_window):
        return await app.look()

    async def click(x, y, **_kwargs):
        return await app.click(x, y)

    async def ready(*_args):
        return None

    async def words(_app):
        return "Fixture board"

    async def no_result(*_args, **_kwargs):
        return "", ""

    monkeypatch.setattr("core.perception.what_the_pixels_show._the_pixels_of", pixels)
    monkeypatch.setattr("core.skills.screen_pursuit_surface.click_normalized", click)
    monkeypatch.setattr("core.capabilities.host_automation.get_host_automation",
                        lambda: SimpleNamespace(_ensure_input_target=ready))
    monkeypatch.setattr(boards, "_her_clock", ready)
    monkeypatch.setattr(boards, "the_words_in", words)
    monkeypatch.setattr(boards, "what_the_window_says_of_the_end", no_result)

    confirmed = asyncio.Event()

    def tell(line, **_kwargs):
        if line.startswith("I played"):
            confirmed.set()

    monkeypatch.setattr(pursuit, "_tell", tell)

    async def interrupt_after_a_move(*, resume: bool = False):
        confirmed.clear()
        turn = asyncio.create_task(boards.played_a_board_of_pieces(
            window.owner, time.monotonic() + 20, {}, resume=resume,
        ))
        witness = asyncio.create_task(confirmed.wait())
        try:
            done, _ = await asyncio.wait((turn, witness), timeout=12, return_when=asyncio.FIRST_COMPLETED)
            assert witness in done, turn.result() if turn.done() else app.made
        finally:
            witness.cancel()
            turn.cancel()
            await asyncio.gather(turn, witness, return_exceptions=True)

    await interrupt_after_a_move()
    retained = boards._SUSPENDED_GAMES[window.owner.casefold()].session.game
    assert retained.made == 1 and len(retained.played) == 1 and retained.believed.turn == SECOND
    assert retained.did.as_record()["done"] == 1

    clicks = app.clicks
    window.number += 1
    assert await boards.played_a_board_of_pieces(window.owner, time.monotonic() + 3, {}, resume=True) is None
    window.number -= 1
    held = app.position
    app.position = app.family.the_start()
    rejected = await boards.played_a_board_of_pieces(window.owner, time.monotonic() + 3, {}, resume=True)
    assert rejected["ended"] == "resume unavailable" and app.clicks == clicks

    app.position = held
    app.make(moves(app.family, app.position)[0])
    await interrupt_after_a_move(resume=True)
    assert retained.made == 2 and retained.theirs == 1
    assert len(retained.played) == len(app.made) == 3
    assert retained.did.as_record()["done"] == 2
