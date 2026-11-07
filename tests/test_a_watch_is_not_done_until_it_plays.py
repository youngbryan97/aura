"""Asked to find something and watch it together, she is not done until it plays with them.

LIVE 7 November 2026: asked "Let's find a funny stand-up comedy clip on YouTube ... and watch it together", she declared
it done on the results page, a clip found and none opened, and nothing was watched.
"""

from __future__ import annotations

import pytest

from core.skills.sovereign_browser import WATCH_DONE_HANDED_BACK_AT_MOST, _a_watching_still_owed

pytestmark = pytest.mark.unit

GOAL = "Let's find a funny stand-up comedy clip on YouTube, something under ten minutes, and watch it together."


def test_done_on_the_results_page_is_handed_back_a_few_times():
    steps: list[dict] = [{"landed": True}]
    owed = [_a_watching_still_owed(GOAL, steps) for _ in range(WATCH_DONE_HANDED_BACK_AT_MOST + 1)]
    assert owed == [True] * WATCH_DONE_HANDED_BACK_AT_MOST + [False]
    assert steps[1]["note"] == "a_watching_is_still_owed" and "open what was found" in steps[1]["why"]


def test_done_once_it_plays_with_them_is_done():
    assert not _a_watching_still_owed(GOAL, [{"landed": True}, {"beside": True, "moved": True}])


def test_a_goal_that_is_not_a_watch_is_done_when_she_says():
    assert not _a_watching_still_owed("Find the opening hours of the library and tell me.", [{"landed": True}])
