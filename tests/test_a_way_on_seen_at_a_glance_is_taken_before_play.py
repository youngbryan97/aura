"""A moving screen whose way on her eyes saw at a glance (SKIP INTRO drawn small over an animation) is not play: its
way on is taken first. Offline 2026-10-13 a game's intro was played as it happened for five minutes, its skip button
found by the pointer and named by her eyes, and the game behind it was never begun."""
from __future__ import annotations

import time
from types import SimpleNamespace

from core.cognition.what_this_place_is import GLIMPSE_HOLDS_S, a_way_on_glimpsed


def _guide_that_glimpsed(*things: str, ago: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(reading=SimpleNamespace(last_glimpsed=(time.monotonic() - ago, things)))


def test_a_way_on_seen_lately_counts():
    assert a_way_on_glimpsed(_guide_that_glimpsed("blue character", "red monster", "skip intro"))
    assert a_way_on_glimpsed(_guide_that_glimpsed("night sky", "start"))


def test_scenery_alone_or_a_glance_long_past_does_not():
    assert not a_way_on_glimpsed(_guide_that_glimpsed("blue character", "red monster", "dark background"))
    assert not a_way_on_glimpsed(_guide_that_glimpsed("skip intro", ago=GLIMPSE_HOLDS_S + 5))
    assert not a_way_on_glimpsed(SimpleNamespace(reading=SimpleNamespace(last_glimpsed=None)))


def test_a_glance_is_kept_with_when_it_was_taken():
    from core.cognition.what_this_place_is import WhatThisPlaceIs

    reading = WhatThisPlaceIs()
    reading.glimpsed("a moonlit street", ["blue blob", "skip intro"])
    assert reading.last_glimpsed is not None and reading.last_glimpsed[1] == ("blue blob", "skip intro")


def test_a_menu_is_gone_through_first_where_one_was_glimpsed():
    """A moving scene whose words offer no way on is gone through first once her eyes saw SKIP INTRO on it."""
    from core.cognition.a_guide_to_a_place import THE_GUIDE
    from core.skills.screen_pursuit_as_it_happens import PlayingAsItHappens

    def first(*glimpsed: str) -> bool:
        playing = PlayingAsItHappens.__new__(PlayingAsItHappens)
        playing.words, playing.way_on_shown, playing.held_for_a_way_on = ["a moonlit street at night"], False, {}
        token = THE_GUIDE.set(_guide_that_glimpsed(*glimpsed))
        try:
            return playing._a_menu_first()
        finally:
            THE_GUIDE.reset(token)

    assert first("blue character", "skip intro")
    assert not first("blue character", "dark background")
