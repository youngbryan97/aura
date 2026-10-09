"""A picture of part of her page is cut from a picture of the whole window, so the page is never laid out again for it.

LIVE 2026-10-09 the window she played in flickered, a frame at a time, to the
game alone on grey, each time she photographed the game: a picture of part of
a page makes Chromium lay the page out at that part's size while it is taken.
"""
from __future__ import annotations

import asyncio
import base64
import io

import numpy as np
import pytest
from PIL import Image

from core.perception import a_picture_of_her_page as pictures


class _Window:
    """A window two screen pixels to each of its own: red where the part is, white elsewhere."""

    def __init__(self) -> None:
        image = Image.new("RGB", (2000, 1400), "white")
        image.paste((200, 30, 30), (600, 600, 1400, 1200))
        out = io.BytesIO()
        image.save(out, format="PNG")
        self.data, self.asked, self.parts = base64.b64encode(out.getvalue()).decode(), [], 0

    async def send(self, method, params):
        self.asked.append((method, params))
        return {"data": self.data}


class _Page:
    viewport_size = {"width": 1000, "height": 700}

    def __init__(self, window):
        self.window = window

        class _Context:
            async def new_cdp_session(_self, page):
                return window

        self.context = _Context()

    async def screenshot(self, **_options):
        self.window.parts += 1
        raise AssertionError("a picture of part of the page was asked for")


@pytest.mark.unit
def test_the_part_is_cut_from_the_whole_window_at_the_pages_own_size():
    window = _Window()
    picture = asyncio.run(pictures.picture_of(_Page(window), {"x": 300, "y": 300, "width": 400, "height": 300}, css=True))
    assert picture.shape == (300, 400, 3) and tuple(picture[150, 200]) == (200, 30, 30)
    assert window.asked == [("Page.captureScreenshot", {"format": "jpeg", "quality": 85})] and window.parts == 0


@pytest.mark.unit
def test_at_the_screens_own_pixels_it_is_twice_the_size():
    picture = asyncio.run(pictures.picture_of(_Page(_Window()), {"x": 300, "y": 300, "width": 400, "height": 300}, css=False,
                                              kind="png"))
    assert picture.shape == (600, 800, 3) and np.all(picture[300, 400] == (200, 30, 30))
