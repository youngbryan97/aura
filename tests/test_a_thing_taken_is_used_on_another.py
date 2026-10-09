"""A thing she takes by clicking it is hers to use on the other things on the screen: a click on it, then on them.

A key for a door, glasses for a robot, a tool for a picture, a file for a folder:
take, then use. Taken, a thing goes from where it was and turns up elsewhere, or
stays and is marked as chosen.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.agency.taking_and_using import a_use_of, what_is_used
from core.agency.what_i_can_do_here import WhatWorksHere, a_click_on

pytestmark = pytest.mark.unit

ROOM = (a_click_on("DOOR"), a_click_on("GLASSES"), a_click_on("LAMP"))


def test_a_thing_gone_from_where_it_was_and_come_again_elsewhere_is_taken():
    here = WhatWorksHere()
    here.looked_at(ROOM)
    here.looked_at(ROOM)
    here.tried(a_click_on("GLASSES"), changed=True)
    after = (a_click_on("DOOR"), a_click_on("LAMP"), a_click_on("the shape at 90% across, 95% down"))
    here.looked_at(after)
    here.looked_at(after)
    assert here.taken == {"GLASSES": "the shape at 90% across, 95% down"}
    uses = here.uses(here.on_screen)
    assert a_use_of("the shape at 90% across, 95% down", "DOOR") in uses
    assert uses[-1] in here.available()


def test_a_thing_chosen_where_it_stands_is_taken_and_a_click_that_did_nothing_takes_nothing():
    here = WhatWorksHere()
    here.looked_at(ROOM)
    here.tried(a_click_on("LAMP"), changed=False)
    here.looked_at(ROOM)
    assert here.taken == {}
    here.tried(a_click_on("LAMP"), changed=True)
    here.looked_at(ROOM)
    assert here.taken == {"LAMP": "LAMP"}
    assert what_is_used(a_use_of("LAMP", "DOOR")) == ("LAMP", "DOOR")


def test_a_use_is_carried_out_as_two_clicks_in_order(monkeypatch):
    from core.skills import screen_pursuit_acting as acting

    clicked: list[str] = []

    async def _click(run, label):
        clicked.append(label)
        return True

    monkeypatch.setattr(acting, "_click_what_she_named", _click)
    monkeypatch.setattr(acting, "BETWEEN_CLICKS_S", 0.0)
    landed = asyncio.run(acting._click_what_she_named(SimpleNamespace(), "GLASSES")) and asyncio.run(
        acting._after_a_moment(acting._click_what_she_named(SimpleNamespace(), "DOOR")))
    assert landed and clicked == ["GLASSES", "DOOR"]
