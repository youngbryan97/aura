"""She plays a world that runs on its own clock from its pixels, at the speed it goes.

The paddle world (tests/fixtures/worlds_that_move/paddle.html) is served to a
headless browser. She is given pictures of its canvas, the keys a player is
told about and nothing else. In fifteen seconds she has to find which thing is
hers, learn that up and down move it and the other keys do not, and get in the
ball's way.
"""
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_she_finds_her_paddle_and_returns_the_ball():
    playwright_api = pytest.importorskip("playwright.async_api")
    from core.agency.playing_as_it_happens import play_as_it_happens
    from tools.measure_playing_as_it_happens import TOLD_KEYS, WORLDS, _Page

    async with playwright_api.async_playwright() as playwright:
        try:
            browser = await playwright.chromium.launch(headless=True)
        except Exception as why:  # noqa: BLE001 - no engine installed here
            pytest.skip(f"no browser engine: {why}")
        page = await browser.new_page(viewport={"width": 800, "height": 600})
        await page.goto(f"file://{WORLDS / 'paddle.html'}?start=1&seed=5")
        box = await page.evaluate(
            "(() => { const r = document.querySelector('canvas').getBoundingClientRect();"
            " return [r.left, r.top, r.width, r.height]; })()"
        )
        eyes = _Page(page, box)
        came_to = await play_as_it_happens(eyes.look, eyes, keys=TOLD_KEYS, seconds=15.0)
        returns = await page.evaluate("__world.returns")
        await browser.close()
    assert came_to["keys_that_move_her"] == ["down", "up"]
    assert came_to["hers"].endswith("bar")
    assert came_to["pictures_a_second"] > 10
    assert returns >= 1, came_to


async def test_an_instruction_to_click_targets_does_not_require_an_avatar():
    playwright_api = pytest.importorskip("playwright.async_api")
    from core.agency.playing_as_it_happens import controls_named_in, play_as_it_happens
    from tools.measure_playing_as_it_happens import WORLDS, _Page

    async with playwright_api.async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport={"width": 800, "height": 600})
            await page.goto(f"file://{WORLDS / 'gallery.html'}?start=1&seed=5")
            box = await page.locator("canvas").bounding_box()
            eyes = _Page(page, (box["x"], box["y"], box["width"], box["height"]))
            rules = " ".join(await page.evaluate("__world.game.rules"))
            keys, pointer_first = controls_named_in(rules)
            result = await play_as_it_happens(eyes.look, eyes, keys=keys, seconds=6.0,
                                             pointer_first=pointer_first, told=rules)
            score = await page.evaluate("__world.score")
            assert score >= 2, result
            assert await page.evaluate("__world.lives") == 3
        finally:
            await browser.close()
