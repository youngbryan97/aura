"""Her account of a game she played is of the whole game, and does not stop at a list item it has begun.

LIVE 21 November 2026 she played 33 moves a side against Stockfish and lost. Her record of it was what she had said as
she played, and only its first 25 lines reached her account, which numbered each side's move as a move of its own
("2....d5 | 3. e3") and stopped at "24. ...Nb6 | 25." with the room of a remark on a film, before it said she had lost.
"""
from __future__ import annotations

import asyncio
import json
import re
from types import SimpleNamespace

import pytest

import core.agency.playing_pieces_by_kind as playing
import core.skills.screen_pursuit as pursuit
from core.skills.computer_use import ComputerUseSkill, _in_pairs
from core.skills.watching_along import her_turn, whole_paragraphs
from tests.fixtures.a_board_application import BoardApplication
from tests.fixtures.a_game_of_pieces_to_learn import NOBODYS_GAME


def test_moves_are_numbered_a_pair_to_each_number() -> None:
    assert _in_pairs(["g1-f3", "d7-d5", "e2-e3"]) == "1. g1-f3 d7-d5 2. e2-e3"


@pytest.mark.asyncio
async def test_the_record_holds_the_whole_game_before_what_she_said(monkeypatch) -> None:
    async def pursued(**_kwargs):
        return {"completed": True, "outcome": "goal_reached", "moves": [], "attempts": [],
                "handed_over": {"ended": "lost", "moves": 2, "theirs": 2, "closed": True, "first_was_hers": False,
                                "the_moves": ["e2-e4", "e7-e5", "g1-f3", "b8-c6"]},
                "said_along_the_way": ["They went first."]}

    monkeypatch.setattr(pursuit, "pursue_on_screen", pursued)
    report = await ComputerUseSkill()._pursue_on_screen(json.dumps({"goal": "play a game", "success_when": "x"}))
    said = report["said"]
    assert "I lost" in said
    assert "The moves, theirs first in each pair: 1. e2-e4 e7-e5 2. g1-f3 b8-c6." in said
    assert said.index("The moves") < said.index("What I said as I played")


def test_the_player_keeps_every_move_it_sees_made(monkeypatch) -> None:
    monkeypatch.setattr(playing, "STILL_S", 0.02)
    monkeypatch.setattr(playing, "OFFERS_SHOW_S", 0.01)
    monkeypatch.setattr(playing, "WAITS_FOR_THE_FIRST_MOVE_S", 0.5)
    monkeypatch.setattr(playing._Rehearsing, "start", lambda self, pool, family: None)
    app = BoardApplication(NOBODYS_GAME, seed=5)

    async def run() -> dict:
        seat = playing.a_board_of_two_sides_at_its_start(app.draw())
        stop = asyncio.Event()
        answering = asyncio.create_task(app.answers(stop))
        try:
            return await playing.play_pieces_by_kind(seat, app.look, app.click, seconds=3.0, thinks_s=0.05)
        finally:
            stop.set()
            await answering

    came = asyncio.run(run())
    assert came["moves"] >= 1 and came["first_was_hers"] is True
    assert len(came["the_moves"]) >= came["moves"] + came["theirs"] - 1
    assert all(re.fullmatch(r"[a-z]\d+[-x][a-z]\d+", move) for move in came["the_moves"]), came["the_moves"]


def test_a_list_cut_at_its_next_number_ends_at_its_last_item() -> None:
    cut = "Done.\n\n**The moves:**\n\n1. **Nf3** (knight g1→f3) | 2. ...d5\n24. ...Nb6 | 25. "
    assert whole_paragraphs(cut).endswith("24. ...Nb6")
    assert whole_paragraphs("It was long. I lost on move 25.") == "It was long. I lost on move 25."


@pytest.mark.asyncio
async def test_an_answer_telling_something_long_has_room_and_time_for_it(monkeypatch) -> None:
    handed: dict = {}

    async def think(objective, *, context, mode, origin, timeout_s):
        handed.update(context=context, timeout_s=timeout_s)
        return SimpleNamespace(content="I lost.")

    monkeypatch.setattr("core.container.ServiceContainer.get",
                        lambda name, default=None: SimpleNamespace(think=think) if name == "cognitive_engine"
                        else default)
    await her_turn("play a game", "the game", [])
    usual_ceiling, usual_time = handed["context"]["hard_output_token_ceiling"], handed["timeout_s"]
    usual_thinking = handed["context"]["private_channel_budget"]
    await her_turn("play a game", "the game", [], room_to_say=800)
    assert handed["context"]["hard_output_token_ceiling"] > usual_ceiling
    assert handed["timeout_s"] == pytest.approx(usual_time * handed["context"]["hard_output_token_ceiling"]
                                                / usual_ceiling, rel=0.01)
    # Room to work it out grows with room to say it: LIVE 21 November 2026 her account of game 13, given more to say
    # and the same room to think, worked its moves out aloud ("Nxe5? — Wait, no: Ne5").
    grown = handed["context"]["private_channel_budget"] / usual_thinking
    assert grown == pytest.approx(800 / 352, rel=0.01)
