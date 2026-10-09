"""Where no body answers her keys and something keeps moving, a press is timed to where it is, and learned by what paid.

A needle swings and a press counts only while it is over the green; a meter rises
and falls and a press sets the power; a hook swings and a press drops what it
holds. Nothing moves when she presses: what is asked is when.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.agency.when_a_press_pays import PLACES, WhenAPressPays

pytestmark = pytest.mark.unit


def _needle(x: float, vx: float) -> SimpleNamespace:
    return SimpleNamespace(kind=1, x=x, y=50.0, vx=vx, vy=0.0, w=4.0, h=20.0, moved=True)


def test_a_place_that_pays_is_pressed_for_and_one_that_costs_is_not():
    timing = WhenAPressPays()
    for x in range(0, 201, 10):
        timing.saw([_needle(float(x), 100.0)])
    gains = losses = 0
    at = 0.0
    for _ in range(80):
        for x in range(0, 201, 4):
            at += 0.04
            needle = [_needle(float(x), 100.0)]
            if timing.press_now(needle, at, 0.0):
                timing.credit(gains, losses)
                timing.pressed(needle, at, 0.0)
                if 120 <= x <= 140:
                    gains += 1
                else:
                    losses += 1
    way = timing.ways[1]
    tries = [int(tries) for tries, _paid in way.places]
    most = max(range(PLACES), key=lambda place: tries[place])
    assert way.places[most][1] > 0 and way.place(120.0) <= most <= way.place(140.0), (tries, way.places)
    assert gains > losses


@pytest.mark.asyncio
async def test_a_needle_over_the_green_is_learned_by_what_paid():
    playwright_api = pytest.importorskip("playwright.async_api")
    from core.agency.playing_as_it_happens import play_as_it_happens
    from core.capabilities.phantom_browser import _chromium_graphics_arguments
    from core.perception.what_the_pixels_show import recognize_text
    from tools.measure_playing_as_it_happens import WORLDS, _Page

    async with playwright_api.async_playwright() as playwright:
        try:
            browser = await playwright.chromium.launch(headless=True, args=_chromium_graphics_arguments())
        except Exception as why:  # noqa: BLE001 - no engine installed here
            pytest.skip(f"no browser engine: {why}")
        try:
            page = await browser.new_page(viewport={"width": 800, "height": 600})
            await page.goto(f"file://{WORLDS / 'timing.html'}?start=1&seed=3")
            box = await page.locator("canvas").bounding_box()
            eyes = _Page(page, (box["x"], box["y"], box["width"], box["height"]))
            said: list[str] = []
            rules = " ".join(await page.evaluate("__world.game.rules"))
            await play_as_it_happens(eyes.look, eyes, keys=["space"], seconds=60.0, say=said.append, told=rules,
                                     read_words=recognize_text)
            points, misses = await page.evaluate("[__world.points || 0, __world.misses || 0]")
        finally:
            await browser.close()
    assert any("timing each press" in line for line in said), said
    assert points >= 5, (points, misses, said)


@pytest.mark.asyncio
async def test_where_the_mouse_is_how_it_is_played_and_nothing_follows_it_a_click_is_the_press_that_is_timed():
    """LIVE 2026-10-09 "when the hamster lines up with the pillow, click again to launch the hamster": with the hamster
    moving she held every click back, and the hamster was never launched."""
    from types import SimpleNamespace

    from core.agency.playing_as_it_happens import _press_in_time, _Run

    clicks, said = [], []
    hands = SimpleNamespace(click=lambda x, y: _record(clicks, (x, y)))
    run = _Run(keys=[], began=0.0)
    run.timing = SimpleNamespace(press_now=lambda things, at, ahead: True, credit=lambda gains, losses: None,
                                 pressed=lambda things, at, ahead: None, has_something_to_time=lambda things: True)
    moves = SimpleNamespace(things={})
    assert await _press_in_time(hands, run, moves, 3.0, said.append)
    assert clicks == [(0.5, 0.5)] and run.last_click == 3.0
    assert said and "click" in said[0]


async def _record(into, value):
    into.append(value)
