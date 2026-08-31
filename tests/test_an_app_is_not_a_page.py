"""Somewhere to go that is an application is an application.

LIVE 2026-08-31: "open the 2048 Game app on my Mac and play it" was planned as
a browser goal and stopped with "the browser is on nothing, not 2048.app". The
application was installed the whole time. Anything named as somewhere to go
counted as a page.
"""

from __future__ import annotations

import pathlib

import pytest

from core.runtime.watched_goal import _an_application_here, read_watched_goal

HAS_2048 = pathlib.Path("/Applications/2048.app").exists()


def test_a_url_is_still_a_page():
    goal = read_watched_goal("Open https://play2048.co and play it, keep going")
    assert goal is not None
    assert goal.where == "https://play2048.co"
    assert not goal.target_app


@pytest.mark.skipif(not HAS_2048, reason="that application is not installed here")
def test_an_installed_application_is_named_as_one():
    goal = read_watched_goal("play the 2048 app until you cannot")
    assert goal is not None
    assert goal.target_app
    assert not goal.where


def test_nothing_installed_by_that_name_stays_a_place():
    """It asks the machine, so a name nothing answers to is left as it was."""
    assert _an_application_here("Wuthering Heights Simulator 9000") == ""


def test_words_around_the_name_do_not_stop_it_being_found():
    if not HAS_2048:
        pytest.skip("that application is not installed here")
    assert _an_application_here("the 2048 Game app") == _an_application_here("2048")


def test_a_goal_with_no_place_at_all_is_untouched():
    goal = read_watched_goal("keep refreshing the page until the build goes green")
    assert goal is not None
    assert not goal.target_app


@pytest.mark.skipif(not HAS_2048, reason="that application is not installed here")
def test_it_answers_with_the_name_the_window_system_uses():
    """2048.app runs as a process called "2048 Game". Answering with the
    filename meant window_bounds matched nothing, so the reading was never
    cropped to the window — and she read a Finder window and her own panels
    alongside the board."""
    assert _an_application_here("the 2048 app") == "2048 Game"
    assert _an_application_here("2048 Game") == "2048 Game"


def test_an_application_named_the_same_either_way_is_unaffected():
    assert _an_application_here("Safari") in {"Safari", ""}
