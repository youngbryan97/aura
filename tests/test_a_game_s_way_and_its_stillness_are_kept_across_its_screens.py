"""The way a place said it is played holds on its later screens, and play that keeps ending on a still screen with
nothing to show for it hands the screen over, though the place goes on in time.

Offline 2026-10-12 "Click on your ball and hold down the mouse button" was read six screens before the course, was no
longer among the words she went by there, and the course was played as a world going by itself, three seconds at a time,
a hundred times over, with not one putt.
"""
from __future__ import annotations

import time

import pytest

from core.agency.the_way_it_is_played import SEND
from core.skills.screen_pursuit_as_it_happens import (
    STILL_STRETCHES_AT_MOST,
    PlayingAsItHappens,
    _left_alone_after,
    _stuck_still,
    _ways_asked_in_this_game,
)

pytestmark = pytest.mark.unit


def _playing() -> PlayingAsItHappens:
    return PlayingAsItHappens(page=None, band=(0, 0, 1, 1), goal="play it", ends_at=time.monotonic() + 60)


def test_the_way_a_place_said_holds_on_a_screen_that_says_none():
    playing = _playing()
    assert _ways_asked_in_this_game(playing, []) == []
    assert _ways_asked_in_this_game(playing, [SEND]) == [SEND]          # the putter's lesson
    assert _ways_asked_in_this_game(playing, []) == [SEND]              # the course, its words only a name and a score


def test_stretches_ending_still_with_nothing_to_show_hand_the_screen_over():
    playing = _playing()
    still = {"hers": "", "gains": 0, "losses": 0, "ended": "nothing on the screen has moved for a while"}
    for _ in range(STILL_STRETCHES_AT_MOST - 1):
        _left_alone_after(playing, still)
    assert not _stuck_still(playing)
    _left_alone_after(playing, still)
    assert _stuck_still(playing)
    _left_alone_after(playing, {**still, "gains": 1, "ended": "the screen says the round is over"})
    assert not _stuck_still(playing), "a stretch that got somewhere starts the count again"


def test_a_way_on_needs_pixel_evidence_to_correct_a_later_reading():
    """A nearby old label supplies no word strokes for the current reading."""
    from core.skills.screen_pursuit_as_it_happens import _ways_on_read_right_before

    playing = _playing()
    first = {"layout": [{"text": "Start", "center_x": 0.74, "center_y": 0.84}, {"text": "ChOY", "center_x": 0.35, "center_y": 0.56}]}
    _ways_on_read_right_before(playing, first)
    later = {"layout": [{"text": "tarit", "center_x": 0.745, "center_y": 0.842}, {"text": "BLEADELS", "center_x": 0.37, "center_y": 0.36},
                        {"text": "Score 120", "center_x": 0.74, "center_y": 0.85}]}
    _ways_on_read_right_before(playing, later)
    assert [region["text"] for region in later["layout"]] == ["tarit", "BLEADELS", "Score 120"]


def test_a_start_over_read_in_two_and_a_warning_of_loss_are_read_as_what_they_are():
    """Offline 2026-10-12 an alien builder's START OVER, its two words set one above the other, was read as START and
    pressed twenty times, each time asking "ARE YOU SURE? YOU WILL LOSE ALL CURRENT PROGRESS ON YOUR ALIEN"."""
    from core.language.a_way_on import how_much_it_leads_on
    from core.skills.screen_pursuit_as_it_happens import _starting_over_read_whole

    builder = {"text": "VIEW GALLERY START OVER I'M DONE!",
               "layout": [{"text": "START", "center_x": 0.74, "center_y": 0.9}, {"text": "OVER", "center_x": 0.74, "center_y": 0.94},
                          {"text": "I'M DONE!", "center_x": 0.88, "center_y": 0.9}]}
    _starting_over_read_whole(builder)
    assert [region["text"] for region in builder["layout"]] == ["START OVER", "OVER", "I'M DONE!"]
    assert how_much_it_leads_on("I'M DONE!") > 1.0 > how_much_it_leads_on("START OVER")
    warned = {"text": "ARE YOU SURE? YOU WILL LOSE ALL CURRENT PROGRESS ON YOUR ALIEN. CANCEL OK",
              "layout": [{"text": "CANCEL", "center_x": 0.55, "center_y": 0.6}, {"text": "OK", "center_x": 0.66, "center_y": 0.6}]}
    _starting_over_read_whole(warned)
    assert [region["text"] for region in warned["layout"]] == ["CANCEL"]
    plain = {"text": "LEVEL COMPLETE OK", "layout": [{"text": "OK", "center_x": 0.5, "center_y": 0.6}]}
    _starting_over_read_whole(plain)
    assert [region["text"] for region in plain["layout"]] == ["OK"]
