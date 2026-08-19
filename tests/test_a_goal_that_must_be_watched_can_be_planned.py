"""Every desktop action happened once, so any request with a condition died.

LIVE, 2026-08-19. "I have 2048 open in Chrome. Play it for me — keep going
until you get a 128 tile" was answered "Done — opened Google Chrome" in under
two seconds, and the turn reported the objective completed.

Nothing was broken. The desktop vocabulary held twenty-four one-shot verbs
and no way to say "keep going until", so a planner had to reduce the request
to its first action. The same hole swallows "wait for the build and tell me
how it ended", "step through the wizard", "watch for the download".

The primitive that was missing takes a goal and the text that means it is
finished, and the run ends on what actually happened rather than on ok.
"""
from __future__ import annotations

import json

import pytest

from core.runtime.desktop_task_contract import (
    DESKTOP_TASK_ALLOWED_ACTIONS,
    DESKTOP_TASK_IRREVERSIBLE_ACTIONS,
    DESKTOP_TASK_RETRY_SAFE_ACTIONS,
)
from core.skills.computer_use import ComputerUseSkill
from core.skills.desktop_task import DesktopTaskSkill


class _Step:
    def __init__(self, action, target="{}"):
        self.action = action
        self.target = target


def test_the_vocabulary_can_express_a_goal_that_has_to_be_watched():
    assert "pursue_on_screen" in DESKTOP_TASK_ALLOWED_ACTIONS


def test_watching_again_is_safe_and_changes_nothing_by_itself():
    assert "pursue_on_screen" in DESKTOP_TASK_RETRY_SAFE_ACTIONS
    assert "pursue_on_screen" not in DESKTOP_TASK_IRREVERSIBLE_ACTIONS


def test_the_action_is_advertised_to_whoever_plans():
    from core.skills.computer_use import ComputerUseParams

    described = ComputerUseParams.model_fields["action"].description
    assert "pursue_on_screen" in described


@pytest.mark.asyncio
async def test_a_pursuit_with_no_finishing_condition_is_refused():
    """A run that cannot end is not a run."""
    skill = ComputerUseSkill()
    result = await skill._pursue_on_screen(json.dumps({"goal": "play the game"}))
    assert result["ok"] is False
    assert "could never end" in result["error"]


@pytest.mark.asyncio
async def test_a_pursuit_with_no_goal_is_refused():
    skill = ComputerUseSkill()
    result = await skill._pursue_on_screen(json.dumps({"success_when": "128"}))
    assert result["ok"] is False
    assert "no goal" in result["error"]


@pytest.mark.asyncio
async def test_a_pursuit_runs_the_loop_with_what_the_plan_asked_for(monkeypatch):
    seen = {}

    async def fake_pursue(**kwargs):
        seen.update(kwargs)
        return {"completed": True, "outcome": "goal_reached", "cycles": 9, "moves": [{"key": "up"}]}

    import core.skills.screen_pursuit as sp

    monkeypatch.setattr(sp, "pursue_on_screen", fake_pursue)
    skill = ComputerUseSkill()
    result = await skill._pursue_on_screen(
        json.dumps(
            {
                "goal": "reach a 128 tile",
                "success_when": "128",
                "target_app": "Google Chrome",
                "expect_page": "play2048",
                "move_keys": ["up", "left"],
                "region_top": 0.25,
                "region_bottom": 0.85,
            }
        )
    )
    assert result["ok"] is True
    assert seen["goal"] == "reach a 128 tile"
    assert seen["success_when"] == "128"
    assert seen["target_app"] == "Google Chrome"
    assert seen["expect_page"] == "play2048"
    assert seen["move_keys"] == ("up", "left")
    assert seen["region_top"] == 0.25


@pytest.mark.asyncio
async def test_a_pursuit_that_ended_blocked_says_what_blocked_it(monkeypatch):
    """"child action reported failure" tells the person nothing."""
    import core.skills.screen_pursuit as sp

    async def blocked(**_kw):
        return {"completed": False, "outcome": "blocked_by_overlay", "blocked_by": "a cookie wall", "moves": []}

    monkeypatch.setattr(sp, "pursue_on_screen", blocked)
    result = await ComputerUseSkill()._pursue_on_screen(
        json.dumps({"goal": "play", "success_when": "128"})
    )
    verified, evidence = DesktopTaskSkill._verify_step_effect(_Step("pursue_on_screen"), result)
    assert verified is False
    assert "cookie wall" in evidence


@pytest.mark.asyncio
async def test_a_pursuit_that_could_not_decide_says_so(monkeypatch):
    import core.skills.screen_pursuit as sp

    async def undecided(**_kw):
        return {
            "completed": False,
            "outcome": "cannot_decide",
            "cannot_decide": "her reasoning could not be reached",
            "moves": [{"key": "up"}],
        }

    monkeypatch.setattr(sp, "pursue_on_screen", undecided)
    result = await ComputerUseSkill()._pursue_on_screen(
        json.dumps({"goal": "play", "success_when": "128"})
    )
    verified, evidence = DesktopTaskSkill._verify_step_effect(_Step("pursue_on_screen"), result)
    assert verified is False
    assert "reasoning could not be reached" in evidence
    assert "1 move(s)" in evidence


@pytest.mark.asyncio
async def test_running_out_of_moves_is_named_in_words_not_as_a_status_code(monkeypatch):
    import core.skills.screen_pursuit as sp

    async def spent(**_kw):
        return {"completed": False, "outcome": "out_of_cycles", "moves": [{"key": "up"}] * 40}

    monkeypatch.setattr(sp, "pursue_on_screen", spent)
    result = await ComputerUseSkill()._pursue_on_screen(
        json.dumps({"goal": "play", "success_when": "128"})
    )
    assert result["error"] == "ran out of moves before reaching the goal (after 40 move(s))"


def test_a_pursuit_that_reached_its_goal_counts_the_moves_it_took():
    verified, evidence = DesktopTaskSkill._verify_step_effect(
        _Step("pursue_on_screen"),
        {"ok": True, "outcome": "goal_reached", "moves": [{"key": "up"}, {"key": "left"}]},
    )
    assert verified is True
    assert "moves=2" in evidence
    assert "goal_reached" in evidence


def test_a_watching_action_is_bounded_by_its_goal_not_by_a_keystroke():
    """LIVE: every run was cut off at thirty seconds mid-game.

    The pursuit carries its own limit and stops itself. Wrapping it in the
    budget sized for one keystroke reported "Operation took too long" for a
    loop that was working.
    """
    from core.skills.computer_use import ComputerUseSkill

    click = ComputerUseSkill.timeout_for({"action": "click"})
    watching = ComputerUseSkill.timeout_for({"action": "pursue_on_screen", "target": "{}"})
    assert watching > click * 10


def test_the_budget_comes_from_what_the_pursuit_asked_for():
    from core.skills.computer_use import ComputerUseSkill

    short = ComputerUseSkill.timeout_for(
        {"action": "pursue_on_screen", "target": json.dumps({"max_seconds": 120})}
    )
    long = ComputerUseSkill.timeout_for(
        {"action": "pursue_on_screen", "target": json.dumps({"max_seconds": 600})}
    )
    assert long > short
    assert short > 120, "a run killed on its own deadline loses the receipt saying what it did"


def test_a_malformed_target_still_gets_a_workable_budget():
    from core.skills.computer_use import ComputerUseSkill

    assert ComputerUseSkill.timeout_for({"action": "pursue_on_screen", "target": "not json"}) > 30.0
