"""A key the words give to hitting that sends nothing out is a blow: she strikes with it what comes close.

"S to punch, D to kick", "Press the Z key to melee attack close enemies": nothing
flies, so there is no shot to aim. She presses when something to hit is within
reach, and learns how far that is from the presses that paid.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.agency.how_far_her_blow_reaches import ARM, FINDING, HerBlows, the_gap
from core.agency.what_meeting_things_does import AVOID, MEET
from core.agency.what_the_rules_said import WhatTheRulesSaid


def _box(number, x, y=100.0, w=20.0, h=30.0, moved=True):
    return SimpleNamespace(number=number, x=x, y=y, w=w, h=h, moved=moved, kind=number)


@pytest.mark.parametrize(("words", "keys"), [
    ("PRESS THE 'S' KEY TO PUNCH 'D' TO KICK, 'A' FOR BATARANG", ("s", "d")),
    ("Press the Z key to melee attack close enemies.", ("z",)),
    ("Press Z to punch the robots.", ("z",)),
    ("Use the arrow keys to move. Press space to punch the robots.", ("space",)),
])
def test_the_words_give_a_blow_its_key(words, keys):
    assert WhatTheRulesSaid.read(words).fire_keys == keys


def test_a_letter_that_is_a_word_is_not_a_key():
    assert WhatTheRulesSaid.read("Get a life by collecting hearts. Press A to start.").fire_keys == ()


def test_she_strikes_what_comes_within_reach_and_not_what_pays_on_touch():
    blows, mine = HerBlows(), _box(0, 100.0)
    near, far, coin = _box(1, 100.0 + 20 + 10), _box(2, 100.0 + 20 + 200), _box(3, 100.0 - 25)
    stance = {1: AVOID, 2: AVOID, 3: MEET}
    assert the_gap(mine, near) == pytest.approx(10 / 30)
    struck = blows.to_strike("z", mine, [mine, near, far, coin], lambda t: stance[t.kind], at=1.0)
    assert struck is near
    blows.pressed("z", 1.0, mine, [mine, near, far, coin])
    assert blows.to_strike("z", mine, [mine, near], lambda t: AVOID, at=1.1) is None   # not again so soon
    assert blows.to_strike("z", mine, [mine, coin], lambda t: MEET, at=2.0) is None     # met, not hit


def test_her_reach_is_learned_from_the_presses_that_paid():
    blows, mine = HerBlows(), _box(0, 100.0)
    assert blows.reach("z") == pytest.approx(ARM * FINDING)
    for at, gap in ((1.0, 15.0), (3.0, 12.0)):
        blows.pressed("z", at, mine, [mine, _box(1, 100.0 + 20 + gap)])
        blows.gained(at + 0.4)
    blows.pressed("z", 5.0, mine, [mine, _box(1, 100.0 + 20 + 34)])                   # nothing came of this one
    assert blows.reach("z") == pytest.approx(15 / 30 * 1.15)
    assert blows.keys(("z", "space"), {"space": object()}) == ["z"]                   # space sends something out


@pytest.mark.asyncio
async def test_she_punches_the_robots_that_walk_up_to_her():
    playwright_api = pytest.importorskip("playwright.async_api")
    from core.agency.playing_as_it_happens import controls_named_in, play_as_it_happens
    from core.capabilities.phantom_browser import _chromium_graphics_arguments
    from tools.measure_playing_as_it_happens import WORLDS, _Page

    async with playwright_api.async_playwright() as playwright:
        try:
            browser = await playwright.chromium.launch(headless=True, args=_chromium_graphics_arguments())
        except Exception as why:  # noqa: BLE001 - no engine installed here
            pytest.skip(f"no browser engine: {why}")
        try:
            page = await browser.new_page(viewport={"width": 800, "height": 600})
            await page.goto(f"file://{WORLDS / 'brawl.html'}?start=1&seed=5")
            box = await page.locator("canvas").bounding_box()
            # Her counters are read from what the page draws, so that no platform's text recognition is needed.
            eyes = _Page(page, (box["x"], box["y"], box["width"], box["height"]), observations="drawing")
            rules = " ".join(await page.evaluate("__world.game.rules"))
            keys, pointer_first = controls_named_in(rules)
            came_to = await play_as_it_happens(eyes.look, eyes, keys=keys, seconds=25.0,
                                               pointer_first=pointer_first, told=rules)
            score, lives, punches = await page.evaluate("[__world.score, __world.lives, __world.game.punches]")
        finally:
            await browser.close()
    # Offline 2026-10-09 in a Linux container: 0 points and every life lost before blows; 60-70 points after.
    assert score >= 30 and lives >= 2, (score, lives, punches, came_to)
