"""A key the words say to hold and let go is held while she plays on, and let go after the time that has paid.

"Hold down the X key to charge up, then release it to fire": the act is in the
letting go, and what it does is in how long the key was held.
"""
from __future__ import annotations

import pytest

from core.agency.holding_to_charge import HOLDS_S, PAID_WITHIN_S, Charging
from core.agency.the_controls_a_game_names import controls_named_in
from core.agency.what_the_rules_said import WhatTheRulesSaid


@pytest.mark.parametrize(("words", "keys"), [
    ("Hold down the X key to charge up, then release it to fire a devastating super attack!", ("x",)),
    ("Press and hold Z to power up, release to shoot.", ("z",)),
    ("Hold space to charge your jump and let go to leap.", ("space",)),
])
def test_the_words_give_the_key_to_hold_and_let_go(words, keys):
    assert WhatTheRulesSaid.read(words).charge_keys == keys


def test_holding_down_another_key_is_not_the_down_key():
    assert controls_named_in("Hold down the X key to charge up.", keys_without_words=())[0] == ["x"]
    assert controls_named_in("Press down to duck.", keys_without_words=())[0] == ["down"]


def test_the_lengths_are_tried_shortest_first_and_the_one_that_paid_is_held_again():
    charging, at = Charging(), 0.0
    for hold in HOLDS_S:
        assert charging.ready(at)
        charging.begin("x", at)
        assert charging.hold_for == hold and not charging.due(at + hold - 0.01) and charging.due(at + hold)
        at += hold
        charging.released(at)
        if hold == 1.5:
            charging.gained(at + 0.3)                                    # only the hold of 1.5 s brought a gain
        at += PAID_WITHIN_S + 0.1
    charging.begin("x", at)
    assert charging.hold_for == 1.5


@pytest.mark.asyncio
async def test_she_charges_the_beam_and_lets_it_go_under_what_comes_down():
    playwright_api = pytest.importorskip("playwright.async_api")
    from core.agency.playing_as_it_happens import play_as_it_happens
    from core.capabilities.phantom_browser import _chromium_graphics_arguments
    from tools.measure_playing_as_it_happens import WORLDS, _Page

    async with playwright_api.async_playwright() as playwright:
        try:
            browser = await playwright.chromium.launch(headless=True, args=_chromium_graphics_arguments())
        except Exception as why:  # noqa: BLE001 - no engine installed here
            pytest.skip(f"no browser engine: {why}")
        try:
            page = await browser.new_page(viewport={"width": 800, "height": 600})
            await page.goto(f"file://{WORLDS / 'charge.html'}?start=1&seed=5")
            box = await page.locator("canvas").bounding_box()
            # Her counters are read from what the page draws, so that no platform's text recognition is needed.
            eyes = _Page(page, (box["x"], box["y"], box["width"], box["height"]), observations="drawing")
            rules = " ".join(await page.evaluate("__world.game.rules"))
            keys, pointer_first = controls_named_in(rules)
            came_to = await play_as_it_happens(eyes.look, eyes, keys=keys, seconds=30.0,
                                               pointer_first=pointer_first, told=rules)
            score, lives, beams = await page.evaluate("[__world.score, __world.lives, __world.game.beams]")
        finally:
            await browser.close()
    # Offline 2026-10-09 in a Linux container: 0 points and every life lost before; 110 points, no life lost, after.
    assert score >= 40 and lives >= 3 and beams >= 3, (score, lives, beams, came_to)
