"""A round that reaches screens the game had not shown her is getting somewhere, scored or not.

LIVE 2026-10-08 a game's menus, story and player choice took most of a round;
nothing was scored on them, and the game was left just as its play began.
"""
from __future__ import annotations

import pytest

from core.skills.screen_pursuit_as_it_happens import PlayingAsItHappens


@pytest.mark.unit
def test_new_screens_are_counted_once_each_across_rounds():
    keep: dict = {}
    first = PlayingAsItHappens(page=None, band=(0, 0, 1, 1), goal="", ends_at=0.0, keep=keep)
    for said in ("LEVEL SELECT Strike Em Out Road Rage", "LEVEL SELECT Strike Em Out Road Rage Backyard", "PICK A PLAYER DAMAGE"):
        first.read({"text": said})
    assert first.new_screens == 2
    again = PlayingAsItHappens(page=None, band=(0, 0, 1, 1), goal="", ends_at=0.0, keep=keep)
    again.read({"text": "PICK A PLAYER DAMAGE"})
    assert again.new_screens == 0
