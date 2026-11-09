"""Her hue and her pointer stay on screen for as long as her browser is hers, and her looks do not make them blink.

8 November 2026, Bryan: "can we have that border be a bit more aesthetically pleasing? more like a hue than a line.
and can her icon be what her mouse would be without that rectangle around it? just would help me identify where her
mouse or clicker is at any given time." The mark had been a one-pixel frame and a rectangle around what she clicked,
and it lasted five seconds after an action began, so through a whole video it was not on screen at all.
"""
from __future__ import annotations

import asyncio
import math
import pytest

from core.perception import browser_presence as presence
from core.perception.browser_presence import BOOTSTRAP

PAGE = "<!doctype html><html><body style='margin:0;background:#fff'><p>A page</p></body></html>"
CLICK = {"x": 200, "y": 150, "width": 120, "height": 40, "clickX": 260, "clickY": 170}
SHOW = "data => window.__auraBrowserPresence.show(data)"


def _violet(picture: bytes) -> int:
    from io import BytesIO

    from PIL import Image

    image = Image.open(BytesIO(picture)).convert("RGB")
    return sum(1 for r, g, b in image.get_flattened_data() if b > 200 and b - g > 20)


async def _page(playwright):
    browser = await playwright.chromium.launch(headless=True)
    context = await browser.new_context(viewport={"width": 640, "height": 400})
    await context.add_init_script(BOOTSTRAP)
    page = await context.new_page()

    async def serve(route):
        await route.fulfill(status=200, body=PAGE, content_type="text/html")

    await page.route("https://plain.test/", serve)
    await page.goto("https://plain.test/")
    return browser, page


EDGE = {"x": 0, "y": 180, "width": 8, "height": 40}
POINTER = {"x": 245, "y": 155, "width": 30, "height": 30}
FAR_CORNER_OF_THE_BOX = {"x": 302, "y": 182, "width": 16, "height": 6}


@pytest.mark.asyncio
async def test_the_hue_and_her_pointer_stay_after_the_action_ends_and_no_rectangle_is_drawn():
    playwright_api = pytest.importorskip("playwright.async_api")
    async with playwright_api.async_playwright() as playwright:
        browser, page = await _page(playwright)
        try:
            await page.evaluate(SHOW, {"action": "clicking", "stage": "completed", "box": CLICK})
            await page.wait_for_timeout(1500)
            assert _violet(await page.screenshot(clip=EDGE)) > 40, "no hue at the window's edge"
            assert _violet(await page.screenshot(clip=POINTER)) > 40, "her pointer is not where she clicked"
            assert _violet(await page.screenshot(clip=FAR_CORNER_OF_THE_BOX)) == 0, "a rectangle was drawn"
        finally:
            await browser.close()


@pytest.mark.asyncio
async def test_her_pointer_is_where_it_was_on_the_next_page_she_opens():
    playwright_api = pytest.importorskip("playwright.async_api")
    async with playwright_api.async_playwright() as playwright:
        browser, page = await _page(playwright)
        try:
            where = await page.evaluate(SHOW, {"action": "clicking", "stage": "started", "box": CLICK})
            assert where == {"x": 260, "y": 170}
            await page.goto("https://plain.test/")
            await page.evaluate(SHOW, {"action": "navigating", "stage": "completed", "box": None, "point": where})
            assert _violet(await page.screenshot(clip=POINTER)) > 40
        finally:
            await browser.close()


@pytest.mark.asyncio
async def test_a_picture_of_part_of_the_window_hides_only_what_of_the_mark_is_in_it():
    playwright_api = pytest.importorskip("playwright.async_api")
    async with playwright_api.async_playwright() as playwright:
        browser, page = await _page(playwright)
        try:
            await page.evaluate(SHOW, {"action": "clicking", "stage": "completed", "box": CLICK})
            video = {"x": 120, "y": 100, "width": 400, "height": 220}
            await page.evaluate("clip => window.__auraBrowserPresence.suspend(clip)", video)
            assert _violet(await page.screenshot(clip=POINTER)) == 0, "her pointer is in her own picture"
            assert _violet(await page.screenshot(clip=EDGE)) > 40, "the hue blinked for a picture it is not in"
            await page.evaluate("() => window.__auraBrowserPresence.resume()")
            await page.evaluate("() => window.__auraBrowserPresence.suspend()")
            assert _violet(await page.screenshot(clip=EDGE)) == 0, "a picture of the whole window has the hue in it"
            await page.evaluate("() => window.__auraBrowserPresence.resume()")
            assert _violet(await page.screenshot(clip=POINTER)) > 40
        finally:
            await browser.close()


class _Browser:
    visible = True
    _presence_available = True

    def __init__(self, page):
        self.page = page


class _Page:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []
        self.url = "https://example.org/watch"

    async def evaluate(self, script, *args):
        self.calls.append((script, args))
        return self.answer


def test_the_mark_is_held_until_her_browser_closes_and_her_pointer_goes_with_her(monkeypatch):
    async def quietly(*_args, **_kwargs):
        return None

    from core.event_bus import get_event_bus

    monkeypatch.setattr(get_event_bus(), "publish", quietly)
    page = _Page({"x": 260, "y": 170})
    browser = _Browser(page)
    asyncio.run(presence.show_activity(browser, "clicking", box=CLICK))
    assert presence._VISIBLE_UNTIL[page] == math.inf
    asyncio.run(presence.show_activity(browser, "navigating", stage="completed"))
    assert page.calls[-1][1][0]["point"] == {"x": 260.0, "y": 170.0}
    asyncio.run(presence.show_activity(browser, "closed", stage="completed"))
    assert presence._VISIBLE_UNTIL[page] == 0.0


def test_her_looks_say_what_part_of_the_window_they_take():
    import inspect

    from core.perception import a_picture_of_her_page
    from core.skills import sovereign_browser_drawing, watching_along

    assert "without_presence(page, clip)" in inspect.getsource(watching_along)
    assert "without_presence(page, clip)" in inspect.getsource(sovereign_browser_drawing)
    assert "without_presence(page, dict(clip) if clip else None)" in inspect.getsource(a_picture_of_her_page)


def test_on_the_desktop_her_actions_show_her_hue_and_pointer_kept_out_of_captures():
    """The same on the desktop: an input action through her bridge shows the hue at the screen's edges and her neuron
    where the pointer is, in windows no capture sees and no click lands on."""
    from pathlib import Path

    launcher = Path("scripts/AuraLauncher.swift").read_text(encoding="utf-8")
    assert 'desktopInputCommands: Set<String> = ["move", "click", "write", "press", "hotkey", "scroll"]' in launcher
    assert "self?.noteDesktopAction(at: pointer, actionMayTake: allowed)" in launcher
    overlay = launcher.split("private func desktopOverlayWindow(frame: NSRect) -> NSWindow {", 1)[1].split("\n    }\n", 1)[0]
    assert "window.sharingType = .none" in overlay and "window.ignoresMouseEvents = true" in overlay
    assert "desktopHueReach: CGFloat = 22" in launcher and "const HUE_REACH = 22;" in Path(
        "core/perception/browser_presence.js").read_text(encoding="utf-8")
    assert "max(desktopStretchLongestGap, allowed)" in launcher


def test_what_a_desktop_action_may_take_travels_with_it():
    import inspect

    from core.security import native_desktop_bridge

    assert '"timeout": float(timeout), **payload' in inspect.getsource(native_desktop_bridge._invoke_resident_bridge)
