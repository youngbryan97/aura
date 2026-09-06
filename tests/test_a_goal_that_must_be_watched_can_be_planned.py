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
async def test_a_pursuit_with_no_finishing_condition_runs_to_its_bounds(monkeypatch):
    """It ends on the cycle count and the clock, which it always had.

    This used to be a refusal, guarding against a loop that could never stop.
    The loop could always stop — the bounds are arguments to it. What the
    refusal blocked was every request that names a process without naming an
    end, which is most of the ways a person asks for one. LIVE 2026-08-27: a
    correctly parsed goal, with the page to open and the keys to press, turned
    away in 417ms.
    """
    import core.skills.screen_pursuit as sp

    asked: dict[str, object] = {}

    async def ran(**kwargs):
        asked.update(kwargs)
        return {"completed": False, "outcome": "out_of_cycles", "moves": []}

    monkeypatch.setattr(sp, "pursue_on_screen", ran)
    skill = ComputerUseSkill()
    result = await skill._pursue_on_screen(json.dumps({"goal": "play the game"}))
    assert asked, "the run was refused before it started"
    assert asked["goal"] == "play the game"
    assert asked["success_when"] == "", "no condition was named, and none was invented"
    assert asked["max_cycles"] > 0 and asked["max_seconds"] > 0, "the bounds it ends on"
    # It ran and ran out, which is an honest ending rather than a refusal.
    assert result["outcome"] == "out_of_cycles"
    assert "could never end" not in str(result.get("error", ""))


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


def test_every_layer_agrees_on_how_long_a_watched_goal_gets():
    """LIVE: she found the site, opened it, played to a score of 744, and the
    turn reported "Operation took too long. Completed 0/0 steps."

    Three layers each had their own idea of the budget and the outermost one
    was smallest, so the run that was working got cancelled. They read one
    number now, and each wraps the one below it.
    """
    from core.runtime.watched_goal import PURSUIT_SECONDS, read_watched_goal
    from core.skills.computer_use import ComputerUseSkill
    from core.skills.desktop_task import DesktopTaskSkill

    goal = read_watched_goal("Go find a 2048 game online and play it until you get a 128 tile.")
    assert goal is not None
    action = ComputerUseSkill.timeout_for(
        {"action": "pursue_on_screen", "target": json.dumps(goal.as_target())}
    )
    task = DesktopTaskSkill.timeout_for(
        {"objective": "Go find a 2048 game online and play it until you get a 128 tile."}
    )
    assert PURSUIT_SECONDS < action < task


def test_an_ordinary_desktop_request_keeps_its_ordinary_budget():
    from core.skills.desktop_task import DesktopTaskSkill

    assert DesktopTaskSkill.timeout_for({"objective": "open Chrome"}) == DesktopTaskSkill.timeout_seconds


def test_the_pursuit_declares_its_own_limit_so_the_layers_can_read_it():
    """The limit is the cycles it may take, at the speed a cycle really runs.

    It was a flat number, chosen when a cycle was a keystroke and a glance. A
    cycle now reads the screen, grades the last prediction and often thinks in
    words, so the flat budget bought a sixth of the play it was written for.
    What the layers read is still one number they can plan against; it is just
    no longer a constant.
    """
    from core.runtime.watched_goal import (
        PURSUIT_CEILING_S,
        PURSUIT_SECONDS,
        read_watched_goal,
        time_for,
    )

    goal = read_watched_goal("play 2048 until you get 128")
    declared = goal.as_target()["max_seconds"]
    assert declared == time_for()
    assert PURSUIT_SECONDS <= declared <= PURSUIT_CEILING_S


@pytest.mark.asyncio
async def test_a_pursuit_reports_what_it_did_in_words(monkeypatch):
    """A step count is bookkeeping. What she did is the answer."""
    import core.skills.screen_pursuit as sp

    async def finished(**_kw):
        return {
            "completed": True,
            "outcome": "goal_reached",
            "moves": [{"key": "up"}, {"key": "left"}, {"key": "down"}],
            "attempts": [{"held": True}, {"held": True}, {"held": False}],
            "restarts": 1,
            "pacing": {"chose": "slow down"},
        }

    monkeypatch.setattr(sp, "pursue_on_screen", finished)
    result = await ComputerUseSkill()._pursue_on_screen(
        json.dumps({"goal": "play until 128", "success_when": "128"})
    )
    said = result["said"]
    assert "Reached it" in said and "128" in said
    assert "3 move(s)" in said
    assert "2 of them did what I expected" in said
    assert "Began again 1 time(s)" in said
    assert "slow down" in said


@pytest.mark.asyncio
async def test_an_unfinished_pursuit_says_how_far_it_got(monkeypatch):
    import core.skills.screen_pursuit as sp

    async def gave_up(**_kw):
        return {
            "completed": False,
            "outcome": "out_of_cycles",
            "moves": [{"key": "up"}] * 40,
            "attempts": [{"held": True}] * 30,
        }

    monkeypatch.setattr(sp, "pursue_on_screen", gave_up)
    result = await ComputerUseSkill()._pursue_on_screen(
        json.dumps({"goal": "play until 128", "success_when": "128"})
    )
    assert "Made 40 move(s)" in result["said"]
    assert "ran out of moves" in result["said"]
    assert "30 of them did what I expected" in result["said"]


def test_what_a_step_said_about_itself_is_the_answer():
    """LIVE: she played a game to a score of 996 and the turn replied "I
    couldn't get to an answer I'd stand behind."

    The work was done and reported in words. Reaching past that for a screen
    buffer, a step count, or an apology is the same mistake in three
    different directions.
    """
    from interface.routes.chat_desktop_objective import _desktop_task_observation

    result = {
        "ok": True,
        "steps_requested": 1,
        "steps_completed": 1,
        "receipts": [
            {
                "action": "pursue_on_screen",
                "ok": True,
                "result": {
                    "said": "Reached it: '128' appeared after 47 move(s). 31 of them did what I expected.",
                    "text": "2048 SCORE 996 BEST 6068 New Game",
                },
            }
        ],
    }
    observed = _desktop_task_observation(result)
    assert "Reached it" in observed
    assert "SCORE 996 BEST" not in observed, "the screen buffer stood in for the answer again"
