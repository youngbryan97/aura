"""A goal with a condition on it is not the first action in it.

LIVE 2026-08-19. "2048 is open in Chrome. Play it — keep going until you get
a 128 tile" was planned as a single open_app step, and the turn reported the
objective completed in two seconds. The heuristics that plan a desktop
objective each answer "what single thing is being asked for", which is the
wrong question about a request carrying a condition.

What separates the two is structural: something to keep doing, and a
condition that ends it. Both must be present. "Open Chrome until it opens" is
one action; "keep going" with no end is a request nobody can accept.
"""
from __future__ import annotations

import json

import pytest

from core.agency.watched_goal import BOARD_KEYS, FORM_KEYS, read_watched_goal
from core.skills.desktop_task import DesktopTaskSkill


def test_something_to_keep_doing_and_a_condition_makes_a_watched_goal():
    goal = read_watched_goal("Play it — keep going until you get a 128 tile.")
    assert goal is not None
    assert goal.success_when == "128"


def test_an_action_with_no_condition_is_not_a_watched_goal():
    assert read_watched_goal("open Chrome") is None
    assert read_watched_goal("write a note about otters") is None


def test_a_condition_with_nothing_to_keep_doing_is_not_one_either():
    assert read_watched_goal("open Chrome until it opens") is None


def test_keeping_going_with_no_end_is_refused():
    """A run that cannot finish is not a run."""
    assert read_watched_goal("keep going") is None
    assert read_watched_goal("just keep playing forever") is None


@pytest.mark.parametrize(
    "request_text,expected",
    [
        ("keep refreshing the build page until it says passed", "passed"),
        ("step through the installer until it says Finished", "Finished"),
        ("watch for the download until it is done", "done"),
        ("play the game in Safari until you hit 512", "512"),
        ('keep trying until the screen says "all tests green"', "all tests green"),
        ("keep going until you reach 4,096", "4096"),
    ],
)
def test_the_condition_is_read_out_of_the_request(request_text, expected):
    goal = read_watched_goal(request_text)
    assert goal is not None, request_text
    assert goal.success_when == expected


def test_a_number_is_preferred_over_a_word_because_a_screen_can_match_it():
    goal = read_watched_goal("keep going until you get a 128 tile")
    assert goal.success_when == "128"


def test_quoted_words_are_taken_as_written():
    goal = read_watched_goal('keep waiting until it says "Build succeeded"')
    assert goal.success_when == "Build succeeded"


def test_the_app_named_in_the_request_is_the_one_watched():
    assert read_watched_goal("play it in Chrome until you get 128").target_app == "Google Chrome"
    assert read_watched_goal("keep playing in Safari until 256").target_app == "Safari"


def test_a_browser_is_watched_below_its_own_chrome():
    """A page title would otherwise satisfy the condition before anything happens."""
    in_browser = read_watched_goal("play 2048 in Chrome until you get 128")
    assert in_browser.region_top == pytest.approx(0.12)
    elsewhere = read_watched_goal("keep stepping through the installer until it says Finished")
    assert elsewhere.region_top == 0.0


def test_the_moves_match_what_the_task_is():
    assert read_watched_goal("play it until 128").move_keys == BOARD_KEYS
    assert read_watched_goal("step through the wizard until it says Finished").move_keys == FORM_KEYS


def test_the_planner_makes_it_one_pursuit_and_not_one_action():
    steps = DesktopTaskSkill()._derive_single_objective_steps(
        "2048 is open in Chrome. Play it — keep going until you get a 128 tile.", {}
    )
    assert [step.action for step in steps] == ["pursue_on_screen"]
    payload = json.loads(steps[0].target)
    assert payload["success_when"] == "128"
    assert payload["target_app"] == "Google Chrome"
    assert payload["region_top"] == pytest.approx(0.12)
    assert steps[0].critical is True


def test_an_ordinary_request_still_plans_the_way_it_did():
    steps = DesktopTaskSkill()._derive_single_objective_steps("open Chrome", {})
    assert [step.action for step in steps] == ["open_app"]


def test_the_pursuit_payload_is_what_the_action_accepts():
    goal = read_watched_goal("play 2048 in Chrome until you get 128")
    payload = goal.as_target()
    assert set(payload) >= {"goal", "success_when", "move_keys", "region_top", "region_bottom", "target_app"}
    assert payload["move_keys"] == list(BOARD_KEYS)
