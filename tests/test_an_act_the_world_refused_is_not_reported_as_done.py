"""An act she proposed and the world refused is reported as refused, never as done.

LIVE 23 November 2026 her account of a game named moves the game had refused: what she said as she played held
"I'll move ..." before every attempt, and nothing kept the attempts the game would not take apart from the moves it
made. Each act now goes proposed, attempted, then accepted or refused (core/agency/what_she_did.py), and her report
of the game says which of her tries the game refused.
"""
from __future__ import annotations

import json

import pytest

import core.skills.screen_pursuit as pursuit
from core.agency.what_she_did import ACCEPTED, REFUSED, WhatSheDid
from core.skills.computer_use import ComputerUseSkill


def test_each_act_is_kept_at_the_stage_the_world_left_it() -> None:
    did = WhatSheDid()
    made, refused, pending = did.proposed("e2-e4"), did.proposed("e2xa8"), did.proposed("g1-f3")
    for act in (made, refused, pending):
        did.attempted(act)
    assert not did.settled(pending, ACCEPTED, known_by="the board changed")
    did.dispatched(made)
    did.dispatched(refused)
    did.settled(made, ACCEPTED, known_by="the board showed it made")
    did.settled(refused, REFUSED, known_by="the game did not take it")
    assert did.done() == ["e2-e4"] and did.refused() == ["e2xa8"] and did.unsettled() == ["g1-f3"]
    assert did.as_record() == {"done": 1, "refused": ["e2xa8"], "unsettled": ["g1-f3"]}


@pytest.mark.asyncio
async def test_her_report_says_which_tries_the_game_refused(monkeypatch) -> None:
    async def pursued(**_kwargs):
        return {"completed": True, "outcome": "goal_reached", "moves": [], "attempts": [],
                "handed_over": {"ended": "lost", "moves": 1, "theirs": 1, "closed": True, "first_was_hers": True,
                                "the_moves": ["e2-e4", "e7-e5"],
                                "acts": {"done": 1, "refused": ["e2xa8"], "unsettled": []}},
                "said_along_the_way": ["I'll move my slant-runner from e2 to a8.", "I'll move e2 to e4."]}

    monkeypatch.setattr(pursuit, "pursue_on_screen", pursued)
    report = await ComputerUseSkill()._pursue_on_screen(json.dumps({"goal": "play a game", "success_when": "x"}))
    said = report["said"]
    assert "Moves I tried that the game refused, so never made: e2xa8." in said
    assert said.index("refused") < said.index("What I said as I played")
