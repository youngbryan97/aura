"""A screen that draws things beside words about them is a legend: each thing is taken in play for what it said.

LIVE 2026-10-09 a game's rules screen drew its candy, its heart and its enemies over what each was for; she read the
words aloud and then met every one of them in play as a stranger. Nothing here is that game: a panel, two drawings,
two captions.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.perception.what_a_legend_shows import most_like, stance_of, what_a_legend_shows
from core.perception.what_moves_in_the_picture import _look_of

pytestmark = pytest.mark.unit

PANEL, SWEET, ENEMY = (120, 90, 200), (250, 80, 150), (150, 150, 160)


def _screen():
    picture = np.full((400, 600, 3), PANEL, np.uint8)
    picture[20:50, 150:450] = (255, 160, 20)                    # a heading
    picture[90:140, 120:180] = SWEET                            # a sweet over its caption
    picture[100:130, 135:165] = (255, 255, 255)
    picture[230:300, 380:470] = ENEMY                           # an enemy over its caption
    picture[250:270, 400:420] = (200, 30, 30)
    regions = [
        {"text": "HOW TO PLAY", "x": 0.25, "y": 0.05, "width": 0.5, "height": 0.08},
        {"text": "CANDY", "x": 0.2, "y": 0.37, "width": 0.1, "height": 0.04},
        {"text": "Gives you invincibility", "x": 0.15, "y": 0.41, "width": 0.22, "height": 0.04},
        {"text": "Destroy the robots!", "x": 0.6, "y": 0.78, "width": 0.25, "height": 0.045},
        {"text": "PLAY", "x": 0.8, "y": 0.9, "width": 0.1, "height": 0.06},
    ]
    return picture, regions


@pytest.mark.parametrize(("words", "stance"), [
    ("Gives you invincibility", "meet"), ("Restores your health", "meet"), ("+500 pts", "meet"),
    ("Destroy the robot dogs to free the puppies!", "shoot"), ("Watch out for the falling rocks", "avoid"),
    ("Don't touch the jellyfish", "avoid"), ("PLAY", ""), ("HOW TO PLAY", ""),
])
def test_a_caption_says_what_to_do_about_its_thing(words, stance):
    assert stance_of(words) == stance


def test_each_drawing_is_found_by_its_caption_and_a_heading_is_not_a_thing():
    picture, regions = _screen()
    shown = what_a_legend_shows(picture, regions)
    assert sorted((s.stance, s.words) for s in shown) == [("meet", "CANDY: Gives you invincibility"),
                                                         ("shoot", "Destroy the robots!")]
    sweet = next(s for s in shown if s.stance == "meet")
    assert sweet.where[1] < 0.37 and abs(sweet.where[0] - 0.2) < 0.02


def test_a_thing_in_play_that_looks_like_a_drawing_is_taken_for_it_and_one_that_does_not_is_not():
    picture, regions = _screen()
    shown = what_a_legend_shows(picture, regions)
    small_sweet = np.full((6, 6, 3), SWEET, np.uint8)
    small_sweet[2:4, 2:4] = (255, 255, 255)                     # drawn smaller in play: size is not compared
    assert most_like(_look_of(small_sweet.reshape(-1, 3)), shown).stance == "meet"
    leaf = np.full((6, 6, 3), (40, 160, 40), np.uint8)
    assert most_like(_look_of(leaf.reshape(-1, 3)), shown) is None


def test_play_takes_a_kind_for_what_the_legend_said_until_play_says_otherwise():
    from types import SimpleNamespace

    from core.agency.playing_as_it_happens import _Run, _what_the_rules_said_of
    from core.agency.what_meeting_things_does import SHOOT, WhatMeetingDoes

    picture, regions = _screen()
    run = _Run(keys=["left", "right"], began=0.0, last_moving=0.0, legend=tuple(what_a_legend_shows(picture, regions)))
    enemy = np.full((8, 8, 3), ENEMY, np.uint8)
    enemy[3:5, 3:5] = (200, 30, 30)
    kinds = [SimpleNamespace(number=4, look=_look_of(enemy.reshape(-1, 3)), colour=ENEMY)]
    meeting, said = WhatMeetingDoes(), []
    _what_the_rules_said_of(None, SimpleNamespace(kinds=kinds), meeting, run, said.append, 1.0)
    assert meeting.told == {4: SHOOT}
    assert said and "Destroy the robots" in said[0]
    assert meeting.stance(4) == SHOOT


def test_a_read_out_of_a_number_and_its_unit_is_not_a_caption():
    """LIVE 2026-10-09 a launch game's distance read-out ("+ 12 ft.") was read as a legend of things to get."""
    picture, _regions = _screen()
    regions = [{"text": "+ 12 ft.", "x": 0.2, "y": 0.41, "width": 0.1, "height": 0.04},
               {"text": "24 ft.: + 0 ft.", "x": 0.6, "y": 0.78, "width": 0.2, "height": 0.045}]
    assert what_a_legend_shows(picture, regions) == []


@pytest.mark.unit
def test_a_paragraph_of_rules_wrapped_over_lines_is_not_read_as_captions():
    """LIVE 2026-10-10 "You have two" / "minutes to collect" was read as a thing called "You have two", to be got."""
    from core.perception.what_a_legend_shows import _captions

    def line(text, y):
        return {"text": text, "x": 0.2, "y": y, "width": 0.4, "height": 0.04}

    wrapped = [line("You have two", 0.30), line("minutes to collect", 0.35), line("as many points as possible.", 0.40)]
    assert _captions(wrapped) == []
    legend = [line("Gem", 0.30), line("Collect these for points", 0.35)]
    assert [c[0] for c in _captions(legend)] == ["Gem: Collect these for points"]
