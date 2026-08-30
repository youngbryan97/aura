"""An objective needing feedback between acts cannot be one generated script.

LIVE 2026-08-30, asked to play 2048 in the browser: planned correctly as one
pursue_on_screen step sized at 3088s, then escalated to OS automation, which
asked the model to write a script for the whole game. The reasoning did not fit
(2601s > 480s), the skill died 35.6s in, and the turn reported "Completed 0/1
steps" without a key ever being pressed.
"""

from __future__ import annotations

import pytest

from core.runtime.watched_goal import read_watched_goal
from core.skills.desktop_task import DesktopTaskSkill

PLAYING = (
    "Open https://play2048.co in the browser and play it. Press arrow keys, "
    "read the board after each press, and keep going until you cannot."
)
WATCHING = "keep refreshing the page until the build goes green"
ONE_SHOT = "create a folder on my desktop called Reports"


@pytest.fixture
def skill():
    return DesktopTaskSkill()


def test_a_goal_to_keep_at_plans_as_a_pursuit(skill):
    steps = skill._derive_single_objective_steps(PLAYING, {})
    assert [one.action for one in steps] == ["pursue_on_screen"]


def test_the_pursuit_is_not_then_replaced_by_a_script(skill):
    steps = skill._derive_single_objective_steps(PLAYING, {})
    assert DesktopTaskSkill._should_escalate_to_os_automation(PLAYING, steps, {}) is False


def test_nothing_about_it_is_particular_to_a_game(skill):
    """Any objective carrying a condition to keep at, not this one."""
    assert read_watched_goal(WATCHING) is not None
    steps = skill._derive_single_objective_steps(WATCHING, {})
    assert (
        DesktopTaskSkill._should_escalate_to_os_automation(WATCHING, steps, {}) is False
    )


def test_a_one_shot_objective_is_untouched_by_this(skill):
    """The change may only ever affect goals that are watched."""
    assert read_watched_goal(ONE_SHOT) is None
    steps = skill._derive_single_objective_steps(ONE_SHOT, {})
    before = DesktopTaskSkill._should_escalate_to_os_automation(ONE_SHOT, steps, {})
    assert before is False  # covered by its own primitives, as it always was


def test_the_budget_still_follows_the_watched_goal(skill):
    """The pursuit runs to its own clock, not to a flat per-skill number."""
    watched = read_watched_goal(PLAYING)
    assert watched is not None
    assert DesktopTaskSkill.timeout_for({"objective": PLAYING}) > float(
        watched.max_seconds
    )
