"""A strip of the ground a place's code says is to be gone over is that ground, not a bar of what she has left.

Offline 2026-10-13, played whole from its title screen (tools/measure_whole_play_on_worlds.py), the paint world's
floor was read as bars as she painted it: rows of paint growing, rows of bare floor shortening, and one of those named
for the PAINTED written above it. Each shortening was counted a loss, so she came to think holding down cost her, and
with losses outrunning gains a stretch of play was ended as going against her while she was winning it. Here the
paint world is drawn by a browser while its floor is painted row by row, as her planner paints it, and her bar reader
reads its canvas: the floor shows as bars, and none of them is taken for what she has left.
"""
from __future__ import annotations

import asyncio
import base64
import io
import re
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.unit]

WORLDS = Path(__file__).parent / "fixtures" / "worlds_that_move"
_A_DRIVEN_CLOCK = """
(() => {
  let now = 1000; const pending = [];
  Object.defineProperty(performance, "now", { value: () => now, configurable: true });
  window.requestAnimationFrame = (fn) => { pending.push(fn); return pending.length; };
  window.__drive = (dt, n) => {
    for (let i = 0; i < n; i++) { now += dt * 1000; const due = pending.splice(0, pending.length); for (const fn of due) fn(now); }
  };
})();
"""


def _paint():
    from core.perception.reading_a_world_s_code import read_a_world_s_code

    page = (WORLDS / "paint.html").read_text()
    return read_a_world_s_code([(WORLDS / "world.js").read_text(), *re.findall(r"<script>([\s\S]*?)</script>", page)])


def _pictures() -> list[np.ndarray]:
    """The canvas while a corridor of bare floor between painted rows is painted along, two cells at a time with
    pauses between, as a sweep that stops to keep clear of something paints it; no bug is there."""
    playwright_api = pytest.importorskip("playwright.async_api")
    from PIL import Image

    async def seen():
        async with playwright_api.async_playwright() as playwright:
            try:
                browser = await playwright.chromium.launch(headless=True)
            except Exception as why:  # noqa: BLE001 - no engine installed here
                pytest.skip(f"no browser engine: {why}")
            page = await browser.new_page(viewport={"width": 480, "height": 400})
            await page.add_init_script(_A_DRIVEN_CLOCK)
            await page.goto(f"file://{WORLDS / 'paint.html'}?seed=3&start=1")
            # Rows 2, 4, 8 and 10 painted whole, and rows 3 and 9 at their ends: two corridors of bare floor, columns 6
            # to 17, with paint above, below and at either end. She stands painting a cell of her own far from them.
            await page.evaluate("""(() => { __drive(1 / 60, 1); const g = __world.game; __world.lives = 1e9; g.bugs = [];
                for (const r of [2, 4, 8, 10]) for (let c = 0; c < 24; c++) g.painted[r * 24 + c] = 1;
                for (const r of [3, 9]) for (const c of [0, 1, 2, 3, 4, 5, 18, 19, 20, 21, 22, 23]) g.painted[r * 24 + c] = 1;
                g.me.x = 2; g.me.y = 300; return 0; })()""")
            out = []
            for step in range(36):
                # Each corridor painted from its left end, two cells every third picture.
                await page.evaluate(f"""(() => {{ const g = __world.game, n = Math.min(12, 2 * Math.floor({step} / 3));
                    for (const r of [3, 9]) for (let c = 6; c < 6 + n; c++) g.painted[r * 24 + c] = 1;
                    __drive(1 / 60, 18); return 0; }})()""")
                url = await page.evaluate("document.querySelector('canvas').toDataURL('image/png')")
                out.append(np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")))
            await browser.close()
            return out

    return asyncio.run(seen())


def _verdicts(pictures) -> tuple[list[dict], object]:
    from core.agency.what_she_has_left import bars_read
    from core.perception.how_full_a_bar_is import BarsOnTheScreen

    bars, verdicts = BarsOnTheScreen(), []
    for i, picture in enumerate(pictures):
        verdicts += list(bars_read(bars, picture, 100.0 + 0.3 * i, [], lambda x, y: "middle"))
    return verdicts, bars


def test_the_floor_she_paints_is_read_as_bars_and_none_is_what_she_has_left(monkeypatch):
    import core.agency.what_its_code_says_she_sees as says
    import core.agency.what_she_has_left as resources
    from core.agency.what_she_has_left import what_is_left
    from core.cognition.a_guide_to_a_place import THE_GUIDE, Guide

    pictures = _pictures()
    guide = Guide(place="paint")
    guide.world_its_code_makes = _paint()
    token = THE_GUIDE.set(guide)
    try:
        # Even a false resource label cannot overrule source-grounded scenery.
        monkeypatch.setattr(resources, "what_a_bar_measures", lambda *args: ("HEALTH", "down is bad"))
        with monkeypatch.context() as unread:
            # Without its ground role, the false label turns shortening rows into losses.
            unread.setattr(says, "a_strip_of_ground", lambda bar, shape, guide=None: False)
            blind, _bars = _verdicts(pictures)
        assert any(v["what"] == "loss" for v in blind), blind
        seen, bars = _verdicts(pictures)
        assert seen == [] and what_is_left(bars) == 1.0
    finally:
        THE_GUIDE.reset(token)


def test_a_strip_off_the_ground_or_with_no_ground_named_is_left_to_be_a_bar():
    from types import SimpleNamespace

    from core.agency.what_its_code_says_she_sees import a_strip_of_ground

    guide = SimpleNamespace(world_its_code_makes=_paint(), stage_in_her_picture=None)
    shape = (213, 320)                                   # the paint stage, 480 by 320, read 320 wide
    floor = SimpleNamespace(row=120, start=10, end=200, colour=(201, 180, 137))
    above = SimpleNamespace(row=12, start=10, end=200, colour=(201, 180, 137))    # where its words are, above the floor
    painted = SimpleNamespace(row=12, start=10, end=200, colour=(47, 127, 209))  # the colour its code paints with
    assert a_strip_of_ground(floor, shape, guide) and a_strip_of_ground(painted, shape, guide)
    assert not a_strip_of_ground(above, shape, guide)
    assert not a_strip_of_ground(floor, shape, SimpleNamespace(world_its_code_makes=None))
