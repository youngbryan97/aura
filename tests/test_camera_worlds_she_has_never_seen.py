"""Generated camera worlds, each with its own keys, mouse and layout: she does what she is asked.

`tools/measure_in_camera_worlds.py` is the full measurement (30 worlds per
category: 120 of 120, against 0 of 120 for random acts). These are its cheap
end, and the null that says the rig can fail at all.
"""

from __future__ import annotations

import random

import pytest

from tools.measure_in_camera_worlds import one

CATEGORIES = ("navigation", "using", "finding", "asking")


@pytest.mark.asyncio
@pytest.mark.parametrize("category", CATEGORIES)
async def test_she_does_what_she_is_asked(category):
    assert all([await one(category, seed) for seed in range(3)])


@pytest.mark.asyncio
@pytest.mark.parametrize("category", CATEGORIES)
async def test_random_acts_do_not(category, monkeypatch):
    """A suite nothing can fail measures nothing."""
    from core.agency import going_to_what_she_sees
    from core.agency.what_hands_do import Chunk, Slot

    roll = random.Random(0)

    def at_random(self, layout, *, slot_s, slots):
        pick = roll.random()
        if pick < 0.3:
            return Chunk((Slot(moved=(roll.randint(-200, 200), 0)),), slot_s)
        if pick < 0.8:
            return Chunk((Slot(frozenset({self.walks})),), slot_s)
        return Chunk((Slot(frozenset({roll.choice(["e", "f", "r", "space"])})),), slot_s)

    monkeypatch.setattr(going_to_what_she_sees.GoingTo, "next_chunk", at_random)
    assert not any([await one(category, seed) for seed in range(3)])
