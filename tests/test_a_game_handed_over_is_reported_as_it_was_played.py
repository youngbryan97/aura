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


def test_a_game_played_on_their_screen_is_not_a_watch_with_nothing_playing(monkeypatch) -> None:
    """LIVE 21 November 2026 the game was taken for a request to watch with nothing playing, and her reply was written
    from "what her browsing found, with nothing opened or playing yet": "Let me look at the board to see what's on it."""
    from core.skills.desktop_task import a_game_was_played
    from interface.routes import chat_desktop_objective

    receipts = [{"action": "computer_use", "result": {"ok": True, "handed_over": {"ended": "lost", "moves": 13}}}]
    assert a_game_was_played(receipts)
    assert not a_game_was_played([{"action": "computer_use", "result": {"ok": True}}])
    monkeypatch.setattr(chat_desktop_objective, "_asks_to_watch", lambda objective: True)
    assert not chat_desktop_objective._asked_to_watch_and_nothing_plays({"ok": True, "played": True}, "play a game")


def test_a_game_step_is_verified_with_the_moves_made_in_the_game() -> None:
    """LIVE 21 November 2026 eleven moves a side were verified as "moves=0", and her reply doubted the game was played."""
    import json

    from core.skills.desktop_task import DesktopTaskSkill, DesktopTaskStep

    step = DesktopTaskStep(action="pursue_on_screen", target=json.dumps({"goal": "play a game"}), reason="asked")
    ok, why = DesktopTaskSkill._verify_step_effect(step, {"ok": True, "moves": [], "outcome": "goal_reached",
                                                          "handed_over": {"ended": "lost", "moves": 11, "theirs": 11}})
    assert ok and "moves=11" in why


@pytest.mark.asyncio
async def test_how_the_game_went_is_what_her_call_came_back_with(monkeypatch) -> None:
    """The game goes to her turn as hers to tell, and what her call came back with is the game's record, not "done"."""
    import json as _json

    from core.conversation.surface_disposition import her_calls_this_turn, record_tool_receipt
    from core.conversation.turn_evidence_custody import bind_turn_evidence_custody
    from core.skills.desktop_task import how_the_game_went
    from interface.routes import chat as chat_routes

    record = "Played a game: 20 move(s) of mine and 20 of theirs; it ended lost."
    receipts = [{"action": "computer_use", "result": {"ok": True, "said": record,
                                                      "handed_over": {"ended": "lost", "moves": 20}}}]
    assert how_the_game_went(receipts) == record
    played = {"ok": True, "status": "completed", "played": True, "evidence": record, "receipts": receipts,
              "steps_requested": 1, "steps_completed": 1}

    class _Engine:
        async def execute(self, skill_name, params, context=None):
            record_tool_receipt(skill_name, ok=True, evidence=str(played["evidence"]))
            return dict(played)

    monkeypatch.setattr(chat_routes.ServiceContainer, "get", staticmethod(
        lambda name, default=None: _Engine() if name == "capability_engine" else default))
    with bind_turn_evidence_custody(session_id="s", turn_id="t"):
        result = await chat_routes._execute_desktop_objective_from_chat(
            "Open En Croissant and play a game of chess against Stockfish.", cognitive_reply="")
        calls = her_calls_this_turn()
    assert result["status"] == "desktop_objective_game_read" and result["response"] == ""
    assert _json.loads(calls[1]["content"])["came_to"] == record
