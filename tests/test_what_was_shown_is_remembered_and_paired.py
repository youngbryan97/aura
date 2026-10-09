"""What each place showed when she turned it over is remembered, and two that showed alike are paired.

Cards turned over to find the pairs, each customer given what they ask for, each
word put with its picture: remember what was where, and bring together what is
alike.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.agency.things_that_go_together import a_match_of, what_is_matched
from core.agency.what_i_can_do_here import WhatWorksHere, a_click_on
from core.perception.how_a_place_looks import alike, look_of

pytestmark = pytest.mark.unit


def _picture_with(*faces: tuple[int, int, int]) -> np.ndarray:
    picture = np.full((100, 400, 3), 30, np.uint8)
    for at, colour in enumerate(faces):
        picture[20:80, 20 + at * 90:80 + at * 90] = colour
    return picture


def _place(at: int) -> dict:
    return {"x": (20 + at * 90) / 400, "y": 0.2, "width": 60 / 400, "height": 0.6}


def test_a_places_look_is_alike_for_the_same_picture_and_unlike_for_another():
    picture = _picture_with((200, 40, 40), (40, 200, 40), (200, 40, 40))
    first, second, third = (look_of(picture, _place(at)) for at in range(3))
    assert alike(first, third) and not alike(first, second)


def test_two_places_that_showed_alike_are_paired_first_and_not_again_once_paired():
    places = [f"the shape at {at * 25}% across, 50% down" for at in range(4)]
    screen = tuple(a_click_on(place) for place in places)
    shown = {places[0]: (0.8, 0.1, 0.1), places[1]: (0.1, 0.8, 0.1), places[2]: (0.8, 0.12, 0.1), places[3]: (0.1, 0.1, 0.8)}
    back = (0.3, 0.3, 0.3)
    backs = {move: back for move in screen}
    here = WhatWorksHere()
    here.looked_at(screen, looks=backs)
    for place in places:
        here.tried(a_click_on(place), changed=True)
        here.looked_at(screen, looks={**backs, a_click_on(place): shown[place]})
        here.looked_at(screen, looks=backs)
    pair = a_match_of(places[0], places[2])
    assert here.pairs(here.on_screen) == (pair,)
    assert here.available()[0] == pair
    assert what_is_matched(pair) == (places[0], places[2])
    here.tried(pair, changed=True)
    assert here.pairs(here.on_screen) == ()


def test_a_click_that_leads_elsewhere_shows_nothing_about_the_place():
    here = WhatWorksHere()
    menu = (a_click_on("PLAY"), a_click_on("HELP"))
    here.looked_at(menu, looks={menu[0]: (0.5, 0.5, 0.5), menu[1]: (0.5, 0.5, 0.5)})
    here.tried(menu[0], changed=True)
    here.looked_at(menu, looks={menu[0]: (0.5, 0.5, 0.5), menu[1]: (0.5, 0.5, 0.5)})
    assert here.showed == {}
