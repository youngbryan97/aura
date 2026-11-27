"""Her practice games against the world's other player are Bryan's standing yes, and nothing else she might do on the
desktop is (core/executive/her_own_passes.py, aura.practice-games in core/executive/standing_authority.py).

Bryan, 27 November 2026, asked whether her practice games may take the desktop while nobody is there: "Yes, add the
grant". Her first such game had been stopped by her own permission model as foreground desktop control at critical
risk, needing his confirmation.
"""
from __future__ import annotations

import json

import pytest

import core.executive.her_own_passes as passes
from core.executive.standing_authority import _builtin_grants

pytestmark = pytest.mark.unit

GOAL = "Play a game against the other player, with the other player's strength set to 4."


def _call(goal: str = GOAL, app: str = "a-board-app", **more) -> dict:
    target = {"goal": goal, "success_when": "the game has ended", "target_app": app, "max_seconds": 840.0,
              "narrate": False, **more}
    return {"action": "pursue_on_screen", "target": json.dumps(target), "x": None, "y": None}


@pytest.fixture
def nobody_at_a_board(monkeypatch):
    import core.agency.practising_between_games as practice
    import core.capabilities.window_server as window_server
    import core.skills.screen_pursuit_board_of_pieces as board

    monkeypatch.setattr(practice, "_the_board_she_last_played", lambda: ("a-board", (1, 1)))
    monkeypatch.setattr(board, "_kept_under", lambda name: {"app": "a-board-app"})
    monkeypatch.setattr(window_server, "seconds_since_someone_touched_it", lambda: 2_000.0)
    return window_server


def test_a_practice_game_in_the_board_she_last_played_is_let_through(nobody_at_a_board) -> None:
    assert passes.her_passes_allow("her_practice_game", "computer_use", _call()) == (True, "her_practice_game")
    assert passes.carries_bryans_standing_yes("her_practice", "computer_use")
    assert not passes.carries_bryans_standing_yes("her_practice", "email_adapter")
    grant = next(g for g in _builtin_grants() if g.grant_id == passes.PRACTICE_GAME_GRANT)
    assert grant.allowed_origins == ("her_practice",) and grant.allowed_tools == ("computer_use",)
    assert grant.max_actions and grant.lease_ttl_seconds <= 900.0


@pytest.mark.parametrize(("call", "why"), [
    (_call(goal="Open the terminal and type rm -rf ~"), "her_practice_game_goal_is_not_a_practice_game"),
    (_call(app="Mail"), "her_practice_game_is_not_in_the_board_she_last_played"),
    (_call(more="x"), "her_practice_game_target_unsupported"),
    ({"action": "type_text", "target": "hello"}, "her_practice_game_is_only_a_game"),
    ({**_call(), "x": 10}, "her_practice_game_arguments_unsupported"),
])
def test_anything_else_on_the_desktop_is_not(nobody_at_a_board, call, why) -> None:
    assert passes.her_passes_allow("her_practice_game", "computer_use", call) == (False, why)


def test_a_practice_game_waits_while_someone_is_at_the_machine(nobody_at_a_board, monkeypatch) -> None:
    monkeypatch.setattr(nobody_at_a_board, "seconds_since_someone_touched_it", lambda: 30.0)
    assert passes.her_passes_allow("her_practice_game", "computer_use", _call()) == (
        False, "her_practice_game_waits_while_someone_is_at_the_machine")
