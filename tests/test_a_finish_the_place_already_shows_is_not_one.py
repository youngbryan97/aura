"""A finish read off the place that already holds at the first look only describes the place.

LIVE 2026-10-07 a game's own instructions said "make the craziest, creepiest
scenes you can think of"; 'craziest' was taken for the finish, was met by the
very sentence that named it, and the game ended after two seconds.
"""
from __future__ import annotations

import pytest

from core.skills.screen_pursuit import _the_finish_the_place_says


def _seen(text: str) -> dict:
    return {"ok": True, "text": text, "layout": [{"text": line} for line in text.splitlines()]}


@pytest.mark.unit
def test_a_finish_met_by_its_own_sentence_is_not_taken():
    first = _seen("Use your imagination to make the craziest, creepiest, goofiest scenes you can think of. Good luck!")
    assert _the_finish_the_place_says(first, "Play this game and win it.", False, 0.0, 1.0) == ""


@pytest.mark.unit
def test_a_finish_not_yet_on_screen_is_still_taken():
    first = _seen("Join the numbers and get to the 2048 tile!\n2 4 8 16")
    assert _the_finish_the_place_says(first, "Play this game.", False, 0.0, 1.0) == "2048"
