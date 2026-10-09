"""What one press does over the moment after it is learned by watching, and a press that takes her clear is made.

A key held to go somewhere is a way. A key pressed to jump is a path: up, and
back down. Offline 2026-10-09 a runner whose only key was a jump was never found
to answer to it (a press that rises and falls averages to going nowhere), was
lost by the tracker each time it set off, and ran into every block.
"""
from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from core.agency.what_a_press_does import WhatAPressDoes

pytestmark = pytest.mark.unit


def _thing(number: int, kind: int, x: float, y: float, vx: float = 0.0, vy: float = 0.0, path=()) -> SimpleNamespace:
    return SimpleNamespace(number=number, kind=kind, x=x, y=y, vx=vx, vy=vy, w=10.0, h=10.0, path=deque(path))


def _pressed(presses: WhatAPressDoes, key: str, began: float, heights, hers: int | None = 1, step: float = 0.05) -> None:
    """One press of ``key`` at ``began``: thing 1 (kind 3) goes up and down by ``heights`` (pixels up, per picture)."""
    presses.pressed(key, began, [_thing(1, 3, 50.0, 100.0), _thing(2, 4, 200.0, 100.0, vx=-80.0)], hers)
    for index, height in enumerate(heights, 1):
        at = began + index * step
        if index == 4:
            presses.released(key, at)
        presses.saw({1: _thing(1, 3, 50.0, 100.0 - height), 2: _thing(2, 4, 200.0 - 80.0 * index * step, 100.0, vx=-80.0)}, at)


JUMP = [10, 18, 24, 28, 30, 30, 28, 24, 18, 10, 0, 0, 0, 0, 0, 0]


def test_a_press_that_rises_and_comes_back_down_lifts_her():
    presses = WhatAPressDoes()
    for began in (0.0, 2.0):
        _pressed(presses, "space", began, JUMP)
        _pressed(presses, "left", began + 1.0, [0] * 16)
    assert presses.lifts(3, 10.0) == ["space"]
    path = presses.path("space", 3)
    assert path(0.25)[1] == pytest.approx(-30.0) and abs(path(1.0)[1]) < 1.0
    assert 0.5 < presses.lasts("space", 3) < 1.0
    # The thing going by on its own was not lifted by anything: its way is taken out of what the press did.
    assert presses.lifts(4, 10.0) == []


def test_curves_that_disagree_are_not_enough_and_two_that_agree_are():
    presses = WhatAPressDoes()
    _pressed(presses, "space", 0.0, JUMP)
    _pressed(presses, "space", 2.0, [-10, -40, -80, -120, -160, -200, -240, -260, -260, -260, -260, -260, -260])
    assert presses.path("space", 3) is None
    _pressed(presses, "space", 4.0, JUMP)
    assert presses.lifts(3, 10.0) == ["space"]


def test_a_press_made_while_falling_is_not_read_as_a_lift():
    """A thing gaining speed is not taken to go on at the speed it had: its fall is not subtracted as its way."""
    presses = WhatAPressDoes()
    falling = [(0.0, 50.0, 60.0), (0.05, 50.0, 62.0), (0.10, 50.0, 66.0), (0.15, 50.0, 72.0)]
    for began in (0.0, 2.0):
        presses.pressed("left", began, [_thing(1, 3, 50.0, 72.0, vy=120.0, path=falling)], 1)
        for index in range(1, 12):
            presses.saw({1: _thing(1, 3, 50.0, min(100.0, 72.0 + 8.0 * index))}, began + index * 0.05)
    assert presses.lifts(3, 10.0) == []


def test_a_sudden_move_is_followed_as_the_same_thing():
    from core.perception.what_moves_in_the_picture import WhatMoves

    moves = WhatMoves()
    picture = np.full((240, 320, 3), 40, np.uint8)
    followed = []
    for frame in range(30):
        shown = picture.copy()
        y = 180 if frame < 20 else max(8, 180 - 30 * (frame - 19))
        shown[y:y + 16, 60:76] = (230, 210, 40)
        moves.see(shown, frame * 0.05)
        scale = moves.shape[1] / 320
        at = (68 * scale, (y + 8) * scale)
        nearest = min(moves.things.values(), key=lambda t: (t.x - at[0]) ** 2 + (t.y - at[1]) ** 2, default=None)
        if frame >= 18 and nearest is not None:
            followed.append(nearest.number)
    assert len(set(followed)) == 1, followed


@pytest.mark.asyncio
async def test_a_runner_learns_her_jump_and_clears_what_runs_at_her():
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
            await page.goto(f"file://{WORLDS / 'jump.html'}?start=1&seed=3")
            box = await page.locator("canvas").bounding_box()
            eyes = _Page(page, (box["x"], box["y"], box["width"], box["height"]))
            said: list[str] = []
            rules = " ".join(await page.evaluate("__world.game.rules"))
            await play_as_it_happens(eyes.look, eyes, keys=["space", "left", "right", "up"], seconds=40.0,
                                     say=said.append, told=rules, read_words=recognize_text)
            cleared, hits = await page.evaluate("[__world.cleared || 0, __world.hits || 0]")
        finally:
            await browser.close()
    assert any("lifts me clear" in line for line in said), said
    assert cleared >= 3, (cleared, hits, said)
