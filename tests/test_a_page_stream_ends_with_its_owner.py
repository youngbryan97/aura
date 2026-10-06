"""Stopping a page watcher releases its browser stream, including a frame arriving during startup."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.perception.frames_as_they_are_drawn import PageFrames
from core.skills.screen_pursuit_on_a_page import OnAPage, _PagePictures

pytestmark = pytest.mark.asyncio


class _Session:
    def __init__(self):
        self.callback = None
        self.sent = []
        self.detached = False
        self.fail_ack = False

    def on(self, event, callback):
        self.callback = callback

    async def send(self, method, params=None):
        self.sent.append(method)
        if method == "Page.startScreencast":
            self.callback({"data": "", "metadata": {}, "sessionId": 1})
        if method == "Page.screencastFrameAck" and self.fail_ack:
            raise RuntimeError("the page closed")

    async def detach(self):
        self.detached = True


def _page(session):
    async def cdp(page):
        return session

    async def evaluate(script):
        return [800, 600]

    return SimpleNamespace(context=SimpleNamespace(new_cdp_session=cdp), evaluate=evaluate)


async def test_a_frame_arriving_during_startup_is_acknowledged():
    session = _Session()
    frames = PageFrames(_page(session))
    assert await frames._start()
    await asyncio.sleep(0)
    assert "Page.screencastFrameAck" in session.sent
    await frames.close()
    assert session.detached and "Page.stopScreencast" in session.sent
    assert not frames._acks


async def test_stopping_the_watcher_closes_the_stream_even_across_retries():
    owner = OnAPage(page=None, name="a page")
    for _ in range(5):
        session = _Session()
        frames = PageFrames(_page(session))
        await frames._start()
        owner._pictures = _PagePictures(frames.page, {}, frames)
        await owner.stop_watching()
        assert session.detached and not frames._acks
        assert owner._pictures is None


async def test_a_page_closing_during_ack_does_not_leave_an_unhandled_task():
    session = _Session()
    session.fail_ack = True
    frames = PageFrames(_page(session))
    await frames._start()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not frames._acks
    await frames.close()


async def test_failure_to_stop_a_stream_still_detaches_the_session():
    class FailedStop(_Session):
        async def send(self, method, params=None):
            if method == "Page.stopScreencast":
                raise RuntimeError("the page disappeared")
            await super().send(method, params)

    session = FailedStop()
    frames = PageFrames(_page(session))
    await frames._start()
    await frames.close()
    assert session.detached and frames._session is None and not frames._acks


async def test_screenshot_fallback_remains_available_without_a_stream_or_canvas():
    import io

    from PIL import Image

    encoded = io.BytesIO()
    Image.new("RGB", (40, 30), "red").save(encoded, format="PNG")

    async def screenshot(**kwargs):
        return encoded.getvalue()

    source = _PagePictures(SimpleNamespace(screenshot=screenshot), {}, SimpleNamespace(unavailable=True))
    picture = await source()
    assert picture.shape[:2] == (30, 40)
    assert picture[10, 10, 2] == 255


async def test_direct_canvas_frames_show_the_requested_surface_in_a_page_with_two_canvases():
    from core.perception.frames_as_they_are_drawn import CanvasFrames

    api = pytest.importorskip("playwright.async_api")
    async with api.async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport={"width": 900, "height": 600})
            await page.set_content('''<body style="margin:0"><canvas id="large" width="500" height="300"></canvas>
              <canvas id="small" width="180" height="150"></canvas></body>''')
            await page.evaluate('''for (const [id, colour] of [['large','red'], ['small','blue']]) {
                const ctx = document.getElementById(id).getContext('2d');
                ctx.fillStyle=colour; ctx.fillRect(0,0,500,300);
            }''')
            clip = await page.locator("#small").bounding_box()
            frames = CanvasFrames(page)
            picture, _ = await frames.look(clip)
            assert picture.shape[:2] == (150, 180)
            assert picture[60, 60, 2] > 200 and picture[60, 60, 0] < 30
            # A partial crop must use the page renderer, which preserves its coordinates.
            partial = {**clip, "width": clip["width"] / 2}
            assert await CanvasFrames(page).look(partial) is None
        finally:
            await browser.close()
