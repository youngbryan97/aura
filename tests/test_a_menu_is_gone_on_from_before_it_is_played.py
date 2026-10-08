"""A screen that offers a way on is a menu, whatever moves on it: it is gone on from before it is played.

LIVE 2026-10-08 a title screen's fireflies were played as a game for fifty-three
seconds, with START and HOW TO PLAY written under them.
"""
from __future__ import annotations

import pytest

from core.language.a_way_on import offers_a_way_on
from core.skills.screen_pursuit_as_it_happens import PlayingAsItHappens


@pytest.mark.unit
def test_a_screen_with_a_way_on_is_a_menu_and_a_scoreboard_is_not():
    assert offers_a_way_on("REGULAR SHOW ALL NIGHTER START HOW TO PLAY CREDITS")
    assert offers_a_way_on("Press space to play")
    assert not offers_a_way_on("SCORE 120 LIVES 3")


@pytest.mark.unit
def test_play_waits_for_the_first_reading_and_twice_at_most_for_a_menu():
    playing = PlayingAsItHappens(page=None, band=(0, 0, 1, 1), goal="", ends_at=0.0)
    assert playing._a_menu_first()  # nothing read yet
    playing.read({"text": "ALL NIGHTER START HOW TO PLAY"})
    assert playing._a_menu_first() and playing._a_menu_first()
    assert not playing._a_menu_first()  # its way on did nothing twice: what moves on it is played
    playing.read({"text": "SCORE 0 FIREFLIES 3"})
    assert not playing._a_menu_first()
