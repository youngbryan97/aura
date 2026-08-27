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

from core.runtime.watched_goal import BOARD_KEYS, FORM_KEYS, read_watched_goal
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


def test_keeping_going_with_no_end_is_bounded_rather_than_refused():
    """A run that names no finish is bounded by the budget instead.

    This asserted None. Requiring a named finish made "find a sliding puzzle
    and work out how it moves by playing it" structurally unplannable — no
    number, no quoted phrase, nothing a screen could be matched against — so it
    fell through to the one-shot verbs and was answered "Done — opened Safari,
    and opened Safari." (LIVE 2026-08-27, and the same shape recorded against
    this module on 2026-08-19 with a different phrasing.)

    Not naming an end does not make a request one-shot. What has to stay true
    is that it cannot run forever, and that is the budget's job — so the
    property is asserted here rather than the refusal.
    """
    for asked in ("keep going", "just keep playing forever"):
        goal = read_watched_goal(asked)
        assert goal is not None, asked
        assert not goal.success_when, "no finish was named, so none may be claimed"
        assert 0 < goal.max_seconds < 24 * 3600, (
            f"{asked!r} was accepted with no bound on how long it may run"
        )


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


def test_a_watched_goal_is_a_desktop_objective():
    """LIVE: the request never reached the lane that could do it.

    "Go find a 2048 game online and play it until you get a 128 tile. Tell me
    what you are doing as you go." was classified as not needing the desktop,
    routed to identity grounding by the rider on the end, and answered "I'm
    Aura. I'm a local stateful cognitive-agent runtime" while the browser sat
    on a blank page.

    The recogniser that plans these is now the one that identifies them, so
    there is one definition rather than two.
    """
    from core.runtime.desktop_objective_intent import looks_like_desktop_objective

    assert looks_like_desktop_objective(
        "Go find a 2048 game online and play it until you get a 128 tile. "
        "Tell me what you are doing as you go."
    )
    assert looks_like_desktop_objective("2048 is open in Chrome. Keep playing it until you get a 128 tile.")


def test_ordinary_conversation_is_still_not_a_desktop_objective():
    from core.runtime.desktop_objective_intent import looks_like_desktop_objective

    assert not looks_like_desktop_objective("who are you?")
    assert not looks_like_desktop_objective("tell me about mycelial networks")
    assert not looks_like_desktop_objective("what did you do earlier today?")
