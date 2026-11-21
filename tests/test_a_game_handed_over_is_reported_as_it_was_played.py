"""A game played by its own player is reported as it was played, not as a count of the pursuit's steps.

LIVE 20 November 2026 she played ten moves a side against Stockfish in En Croissant, lost on time, and left the
application open. The pursuit reported "Reached it ... after 2 move(s)", and her reply said she had opened with e4,
that the game ended after two moves, and that she had closed the application. None of it was true.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import core.skills.screen_pursuit as pursuit
from core.skills.computer_use import ComputerUseSkill
from core.skills.screen_pursuit_steps import _pursue_on_screen_result

LINES = ["I'll move my piece that runs straight from a2 to a3. It looks about even.",
         "They played their piece that runs straight from c7 to c5."]


def _receipt(completed: bool) -> SimpleNamespace:
    return SimpleNamespace(completed=completed,
                           to_dict=lambda: {"completed": completed, "outcome": "goal_reached" if completed else ""})


def _result_of(played: dict) -> dict:
    return _pursue_on_screen_result({"value": False}, {"count": 0, "last": ""}, [], [{"key": "x"}] * 2,
                                    {"choice": "", "because": "", "waits": 0}, _receipt(True),
                                    {"count": 0, "because": ""}, {"value": False}, "a game is played",
                                    drawn={"board_of_pieces": played})


def test_a_game_that_stopped_before_its_end_is_not_reported_reached() -> None:
    result = _result_of({"ended": "out of time", "moves": 10, "theirs": 9})
    assert result["completed"] is False and result["outcome"] == "stopped"
    assert result["handed_over"] == {"ended": "out of time", "moves": 10, "theirs": 9}


def test_a_game_played_to_its_end_is_reached() -> None:
    result = _result_of({"ended": "lost", "moves": 31, "theirs": 31, "closed": True})
    assert result["completed"] is True
    assert result["handed_over"]["closed"] is True


@pytest.mark.asyncio
async def test_the_report_says_the_game_as_it_was_played(monkeypatch) -> None:
    async def pursued(**_kwargs):
        return {"completed": False, "outcome": "stopped", "moves": [{"key": "x"}] * 2, "attempts": [],
                "stopped_because": "the game stopped before its end: out of time",
                "handed_over": {"ended": "out of time", "moves": 10, "theirs": 9},
                "said_along_the_way": LINES}

    monkeypatch.setattr(pursuit, "pursue_on_screen", pursued)
    report = await ComputerUseSkill()._pursue_on_screen(json.dumps({"goal": "play a game", "success_when": "x"}))
    said = report["said"]
    assert "10 move(s) of mine and 9 of theirs" in said and "ended out of time" in said
    assert "still open" in said and "2 move(s)" not in said
    assert all(line in said for line in LINES)


def test_what_she_says_in_a_run_is_kept_for_its_report(monkeypatch) -> None:
    monkeypatch.setattr("core.agency.narrator.Narrator.say_everywhere", staticmethod(lambda _said: None))
    kept: list[str] = []
    token = pursuit._SAID_THIS_RUN.set(kept)
    try:
        pursuit._tell("They played their piece that leaps two-and-one from b8 to c6.")
    finally:
        pursuit._SAID_THIS_RUN.reset(token)
    assert kept and "b8 to c6" in kept[0]
