"""A key the screen asks for in its own words is one of her moves, ahead of the rest.

Offline 2026-10-03 20:23, on a Pong whose title read "Press SPACE to play", the
screen pursuit made 199 moves with the arrow keys and never pressed space:
space and return press whatever has focus, so they are never tried to find out
what they do. A screen that asks for one is not an experiment.
"""
from __future__ import annotations

import pytest

from core.agency.what_i_can_do_here import WhatWorksHere, keys_a_screen_asks_for

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("words", "keys"),
    [
        ("Press SPACE to play", ("space",)),
        ("Hit the space bar to start", ("space",)),
        ("Press any key", ("space",)),
        ("press enter to continue", ("return",)),
        ("Press P to play", ("p",)),
        ("Press a button to begin", ()),
        ("Click START", ()),
        ("PONG Press SPAÇE to play", ("space",)),
    ],
)
def test_what_a_screen_asks_for(words, keys):
    assert keys_a_screen_asks_for(words) == keys


def test_a_key_the_screen_asks_for_comes_first_while_it_asks():
    can_do = WhatWorksHere(told=("up", "down", "left", "right"))
    can_do.asked_for_by("PONG Up and down arrows move your paddle. Press SPACE to play")
    assert can_do.available()[0] == "space"
    for _ in range(6):
        can_do.tried("space", changed=False)
    assert can_do.available()[0] == "space"
    can_do.asked_for_by("")
    assert "space" not in can_do.available()
