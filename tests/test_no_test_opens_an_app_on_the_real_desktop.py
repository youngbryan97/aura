"""A test never opens Notes, or a browser, on the Mac that runs it.

Bryan, 2026-10-01: "when we run computer skills, for some reason my notes app
always opens? and sometimes a browser?" Two causes, both closed here: a test
of a desktop skill could reach `open` and AppleScript on the real machine, and
the fallback planner opened Notes for any goal with the letters "notes" in it.
"""
from __future__ import annotations

import subprocess
import webbrowser

import pytest

from core.agency.autonomous_task_engine import AutonomousTaskEngine
from tests.host_actuation_guard import would_act_on_the_desktop

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "args",
    [
        ["open", "-a", "Notes"],
        ["/usr/bin/open", "https://www.google.com/search?q=robot"],
        ["osascript", "-e", 'tell application "Notes" to make new note'],
        ["osascript", "-e", 'tell application "Safari" to open location "https://x.test"'],
        ["osascript", "-e", 'tell application "System Events" to set frontmost of process "Notes" to true'],
        ["/bin/sh", "-c", "sleep 0; open -a Notes"],
    ],
)
def test_what_would_show_something_is_recognised(args):
    assert would_act_on_the_desktop(args)


@pytest.mark.parametrize(
    "args",
    [
        ["osascript", "-e", 'tell application "System Events" to get name of first process whose frontmost is true'],
        ["git", "status"],
        ["ls", "-la"],
        ["/bin/sh", "-c", "echo opened"],
    ],
)
def test_reading_and_ordinary_commands_are_left_alone(args):
    assert not would_act_on_the_desktop(args)


def test_a_refused_launch_fails_the_way_a_failed_command_does(_no_test_opens_an_app_on_the_real_desktop):
    done = subprocess.run(["open", "-a", "Notes"], capture_output=True, text=True)
    assert done.returncode == 1
    assert "real desktop" in done.stderr
    assert _no_test_opens_an_app_on_the_real_desktop, "the refusal was not recorded"


def test_a_browser_is_not_opened(_no_test_opens_an_app_on_the_real_desktop):
    assert webbrowser.open("https://www.google.com/search?q=robot") is False
    assert any("robot" in item for item in _no_test_opens_an_app_on_the_real_desktop)


@pytest.mark.parametrize(
    "goal",
    [
        "take notes on the psych test",
        "summarize the footnotes in this paper",
        "what this denotes about her",
        "the browser tab I left open",
    ],
)
def test_a_word_inside_a_goal_is_not_a_request_to_open_an_app(goal):
    assert AutonomousTaskEngine._extract_app_name(goal) == ""


@pytest.mark.parametrize(
    ("goal", "app"),
    [
        ("open a browser and search for cats", "Browser"),
        ("open the Notes app", "Notes"),
    ],
)
def test_an_app_asked_for_is_still_opened(goal, app, monkeypatch):
    from core.runtime import app_target_resolution
    from core.runtime.app_target_resolution import InstalledApp

    monkeypatch.setattr(
        app_target_resolution,
        "installed_app_inventory",
        lambda **_k: (InstalledApp(name="Notes", path="/System/Applications/Notes.app"),),
    )
    assert AutonomousTaskEngine._extract_app_name(goal) == app


def test_an_app_that_is_not_installed_is_not_guessed(monkeypatch):
    from core.runtime import app_target_resolution

    monkeypatch.setattr(app_target_resolution, "installed_app_inventory", lambda **_k: ())
    assert AutonomousTaskEngine._extract_app_name("start Frobnicator please") == ""


def test_the_planner_is_not_shown_one_app_as_the_way_to_begin():
    """Its only worked example opened Notes, so a plan that copied it did."""
    from core.planning import task_decomposer

    assert '"Notes"' not in task_decomposer.DECOMPOSE_PROMPT
