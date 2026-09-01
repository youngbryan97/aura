"""Getting there is part of the task, for a page and for an application alike.

LIVE 2026-09-01, asked "Find 2048 online, play it, and get to a 256 tile":

  this run belongs to '2048 Game' on 'https://careers-amd.icims.com/jobs/...'
  what she came for is 0 screenful(s) down: 25x36 with 58 thing(s) in it
  I am choosing **right** because the "Did you finish applying?" prompt ...
  pursue_on_screen failed: the page it was working on was replaced (after 0 move(s))

Two independent reasons, and neither was about 2048.

The word "online" was trimmed off the name before the name was looked up, so a
request that said "the web" resolved to the 2048.app sitting in /Applications
and the place was cleared in favour of it. And an application named as the
world was never brought to the front before the first look, because the arrival
step was written for pages only — so the run read a job posting that happened
to be behind everything, and reasoned about it for 176 seconds.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from core.runtime.watched_goal import read_watched_goal

pytestmark = pytest.mark.unit


def _target(said: str) -> dict:
    goal = read_watched_goal(said)
    return goal.as_target() if goal is not None else {}


def test_saying_online_keeps_it_on_the_web() -> None:
    """Even when a local application answers to the same name."""

    target = _target("Find 2048 online, play it, and get to a 256 tile.")
    assert target.get("open_page") == "2048"
    assert not target.get("target_app")


@pytest.mark.parametrize(
    "said",
    [
        "Find 2048 online and get to 256.",
        "Open 2048 in my browser and get to 256.",
        "Play the web version of 2048 until 256.",
        "Pull up 2048 on the web and play to 256.",
    ],
)
def test_every_way_of_saying_the_web(said: str) -> None:
    assert _target(said).get("open_page"), said


def test_a_bare_name_still_prefers_what_is_installed() -> None:
    """The qualifier is the person choosing; without it the choice is open."""

    target = _target("Play 2048 and get to 256.")
    assert target.get("target_app")
    assert not target.get("open_page")


def test_an_application_is_reached_before_anything_looks_at_it() -> None:
    """The arrival `reach` gives a page, given to an application too."""

    source = Path("core/skills/screen_pursuit.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        said = ast.unparse(node)
        if "reach(open_page" not in said:
            continue
        assert node.orelse, "the page arrival has no application branch"
        arrival = ast.unparse(node.orelse[0] if node.orelse else node)
        assert "_bring_the_thing_back_to_the_front" in arrival
        assert "could_not_get_there" in arrival
        return
    raise AssertionError("the arrival step is gone")


def test_failing_to_arrive_is_reported_rather_than_pursued_anyway() -> None:
    """Zero moves against the wrong window is worse than saying it could not."""

    source = Path("core/skills/screen_pursuit.py").read_text(encoding="utf-8")
    assert source.count('"outcome": "could_not_get_there"') >= 2
