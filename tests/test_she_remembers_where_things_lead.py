"""A place she has been through before is gone through again by what led on there, not by trying everything.

LIVE-like 2026-10-05, an archived game: title, then a level select, then play.
Every sitting found the way through again from nothing.
"""
from __future__ import annotations

from core.agency.what_i_can_do_here import WhatWorksHere, a_click_on
from core.agency.where_things_lead import WhereThingsLead, screen_words

_TITLE = [a_click_on("Battle Bowlers"), a_click_on("START")]
_LEVELS = [a_click_on("LEVEL SELECT"), a_click_on("Back"), a_click_on("Strike Em Out"), a_click_on("Road Rage")]
_PLAYING = [a_click_on("SCORE 0015250")]


def _a_sitting(here: WhatWorksHere) -> None:
    here.looked_at(_TITLE)
    here.looked_at(_TITLE)
    here.tried("up", False)  # pressed at the title, nothing
    here.looked_at(_TITLE)
    here.tried("down", False)
    here.looked_at(_LEVELS)
    here.tried(a_click_on("START"), True)
    here.looked_at(_TITLE)
    here.tried(a_click_on("Back"), True)  # back where she was
    here.looked_at(_LEVELS)
    here.tried(a_click_on("START"), True)
    here.looked_at(_PLAYING)
    here.tried(a_click_on("Strike Em Out"), True)
    here.looked_at(_PLAYING)
    here.tried("left", True)  # her keys move things here: this is where the playing is


def test_the_next_sitting_goes_the_way_that_led_on():
    first = WhatWorksHere(told=("up", "down", "left", "right"))
    _a_sitting(first)
    again = WhatWorksHere.from_memory(first.as_memory(), told=("up", "down", "left", "right"))
    again.looked_at(_LEVELS)
    again.looked_at(_LEVELS)
    assert again.leads.how_it_led(a_click_on("Strike Em Out")) > 1.0
    again.looked_at(_TITLE)
    assert again.leads.how_it_led(a_click_on("START")) > 1.0  # taken again after going back, still the way on
    again.looked_at(_LEVELS)
    assert again.leads.how_it_led(a_click_on("Back")) < 1.0  # it led back, not on: below what is untried here
    assert again.leads.how_it_led(a_click_on("Back")) < again.leads.how_it_led(a_click_on("Road Rage"))
    assert again.leads.in_order([a_click_on("Back"), a_click_on("Road Rage"), a_click_on("Strike Em Out")])[0] == a_click_on("Strike Em Out")


def test_what_did_nothing_on_a_screen_is_tried_last_there():
    leads = WhereThingsLead()
    for _ in range(2):
        leads.looked(_TITLE)
        leads.looked(_TITLE)
        leads.acted("up", False)
    leads.looked(_TITLE)
    assert leads.how_it_led("up") == 0.5
    assert leads.in_order(["up", a_click_on("START")]) == (a_click_on("START"), "up")


def test_a_screen_read_a_little_differently_is_the_same_screen():
    leads = WhereThingsLead()
    one = leads.which(screen_words(_LEVELS))
    misread = leads.which(screen_words([a_click_on("LEUEL SELECT"), a_click_on("Strike Em Out"), a_click_on("Roed Rage"), a_click_on("Back")]))
    other = leads.which(screen_words([a_click_on("GAME OVER"), a_click_on("Play Again"), a_click_on("Main Menu")]))
    assert one == misread and other != one
    assert screen_words(['click "the shape at 90% across, 90% down"', 'click "0015250"']) == frozenset()


def test_what_led_on_from_one_screen_is_tried_first_on_the_next():
    """The arrow that turned a comic's first page is pressed first on its second, a page she has not seen."""
    arrow = 'click "the shape at 95% across, 90% down"'
    leads = WhereThingsLead()
    page_one = [a_click_on("DAD BOWLING IS SO BORING"), a_click_on("WHEN ARE WE GOING HOME"), arrow]
    page_two = [a_click_on("THE PINS ARE COMING TO LIFE"), a_click_on("LET'S ROLL"), arrow]
    leads.looked(page_one)
    leads.looked(page_two)
    leads.acted(arrow, True)
    leads.looked(page_two)
    assert leads.in_order(["up", a_click_on("LET'S ROLL"), arrow])[0] == arrow
