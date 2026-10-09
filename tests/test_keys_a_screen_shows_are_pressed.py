"""Keys a screen shows as pictures mid-play are read, and pressed while it shows them: in turn, and fast.

LIVE 2026-10-08 a game caught her character and drew ← and → over it, lit by
turns: tap them one after the other to break free. Her reading of writing made
the ← a "+" and missed the →, and she steered a character that could not move.
"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from core.agency.pressing_what_is_shown import KeysShown
from core.perception.keys_drawn_on_screen import keys_drawn, which_way_it_points

pytestmark = pytest.mark.unit


def _arrow(way: str, length: int, thick: int, head: int) -> np.ndarray:
    from PIL import Image, ImageDraw

    canvas = Image.new("L", (80, 80), 0)
    draw, m = ImageDraw.Draw(canvas), 40
    draw.rectangle([m - length // 2, m - thick // 2, m + length // 2 - head, m + thick // 2], fill=255)
    draw.polygon([(m + length // 2 - head, m - head), (m + length // 2, m), (m + length // 2 - head, m + head)], fill=255)
    return np.asarray(canvas.rotate({"right": 0, "up": 90, "left": 180, "down": 270}[way])) > 128


@pytest.mark.parametrize("way", ["left", "right", "up", "down"])
@pytest.mark.parametrize("length,thick,head", [(40, 4, 10), (16, 4, 7), (20, 2, 10), (50, 8, 16)])
def test_an_arrow_is_read_by_its_shape_whatever_its_proportions(way, length, thick, head):
    assert which_way_it_points(_arrow(way, length, thick, head)) == way


def test_what_is_no_arrow_points_nowhere():
    square = np.zeros((30, 30), bool)
    square[5:25, 5:25] = True
    plus = np.zeros((30, 30), bool)
    plus[13:17, 3:27] = True
    plus[3:27, 13:17] = True
    rows, cols = np.mgrid[:30, :30]
    assert which_way_it_points(square) == which_way_it_points(plus) == which_way_it_points((rows - 15) ** 2 + (cols - 15) ** 2 < 100) == ""


def _a_screen_with_keys(*ways: str) -> np.ndarray:
    picture = np.full((320, 480, 3), 90, np.uint8)
    for at, way in enumerate(ways):
        x = 200 + at * 32
        picture[100:126, x:x + 26] = 235
        mark = _arrow(way, 18, 4, 7)[31:49, 31:49]
        picture[104:122, x + 4:x + 22][mark] = 20
    return picture


def test_key_caps_on_a_screen_are_read_as_the_keys_they_show_and_nothing_else_is():
    assert [key["key"] for key in keys_drawn(_a_screen_with_keys("left", "right"))] == ["left", "right"]
    assert keys_drawn(np.full((320, 480, 3), 90, np.uint8)) == []


async def _looked(shown: KeysShown, picture: np.ndarray, at: float) -> None:
    """One look at ``picture`` at ``at``, taken in: a look runs off the loop and is taken in at the next one."""
    if shown.looking is not None:
        await shown.looking
    shown.look(picture, at)
    await shown.looking
    shown.look(picture, at + 0.01)


class _Hands:
    def __init__(self) -> None:
        self.did: list[str] = []

    async def down(self, key: str) -> None:
        self.did.append("down " + key)

    async def up(self, key: str) -> None:
        self.did.append("up " + key)


def test_keys_that_appear_in_play_are_asked_for_and_pressed_in_turn_and_the_screens_furniture_is_not():
    async def run() -> tuple[KeysShown, KeysShown, _Hands]:
        furniture, appearing = KeysShown(), KeysShown()
        for at in (0.0, 2.0, 4.0):
            await _looked(furniture, _a_screen_with_keys("up"), at)
        await _looked(appearing, _a_screen_with_keys(), 0.0)
        await _looked(appearing, _a_screen_with_keys("left", "right"), 2.0)
        hands, said = _Hands(), []
        for _ in range(3):
            await appearing.press(hands, say=said.append)
        assert said == ["It's showing left and right: pressing them in turn, fast."]
        return furniture, appearing, hands

    furniture, appearing, hands = asyncio.run(run())
    assert not furniture.asks(4.5)
    assert appearing.asks(2.2) and not appearing.asks(3.0)
    assert hands.did == ["down left", "up left", "down right", "up right"] * 3


@pytest.mark.asyncio
async def test_held_in_a_world_she_breaks_free_by_pressing_what_it_shows():
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
            await page.goto(f"file://{WORLDS / 'held.html'}?start=1&seed=3")
            box = await page.locator("canvas").bounding_box()
            eyes = _Page(page, (box["x"], box["y"], box["width"], box["height"]))
            said: list[str] = []
            await play_as_it_happens(eyes.look, eyes, keys=["up", "down", "left", "right"], seconds=15.0, say=said.append)
            caught, freed = await page.evaluate("[__world.caught || 0, __world.freed || 0]")
        finally:
            await browser.close()
    assert caught >= 1 and freed >= 1, (caught, freed, said)
    assert "It's showing left and right: pressing them in turn, fast." in said
