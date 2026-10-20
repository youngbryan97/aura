"""Watching past an advert: the video's length is the one its page states, and a skip control on screen is pressed.

LIVE 20 October 2026, the watching rehearsal: the video element held a six-second advert when its length was read,
so an eleven-minute video was taken for six seconds, every moment was stamped 0:06, and the watching ended two
minutes in. Bryan, watching an advert play in his browser: "one thing.... skipping ads".
"""
from __future__ import annotations

import asyncio
import json

import pytest

from core.skills.watching_along import WHAT_THE_PAGE_SAYS
from core.skills.watching_the_screen import _skip_on_screen

WINDOW = {"X": 864.0, "Y": 33.0, "Width": 864.0, "Height": 1022.0}


def test_a_skip_control_in_the_window_is_the_one_pressed():
    elements = [
        ("AXLink", "Skip navigation", (880.0, -877.0), (128.0, 40.0)),  # above the page, for keyboards
        ("AXButton", "Learn more", (900.0, 400.0), (120.0, 40.0)),
        ("AXButton", "Skip", (1600.0, 420.0), (100.0, 40.0)),
    ]
    assert _skip_on_screen(elements, WINDOW) == 2


def test_a_skip_control_off_screen_or_in_words_only_is_not():
    assert _skip_on_screen([("AXLink", "Skip navigation", (880.0, -877.0), (128.0, 40.0))], WINDOW) is None
    assert _skip_on_screen([("AXStaticText", "Skip", (1600.0, 420.0), (100.0, 40.0))], WINDOW) is None
    assert _skip_on_screen([("AXButton", "Don't skip this one", (1600.0, 420.0), (100.0, 40.0))], WINDOW) is None
    assert _skip_on_screen([("AXButton", "Skip", (100.0, 420.0), (100.0, 40.0))], WINDOW) is None  # another window


def _page(advert_s: float) -> str:
    stated = {"@context": "https://schema.org", "@type": "VideoObject",
              "name": "I built a glove that Explodes whatever it Punches", "duration": "PT11M13S"}
    return (f'<script type="application/ld+json">{json.dumps(stated)}</script>'
            f'<main><video width="640" height="360" data-advert="{advert_s}"></video><p>an advert plays first</p></main>')


def test_the_length_is_the_one_the_page_states_not_the_adverts():
    playwright = pytest.importorskip("playwright.async_api")

    async def look() -> dict:
        async with playwright.async_playwright() as running:
            try:
                browser = await running.chromium.launch(headless=True)
            except Exception as why:  # noqa: BLE001 - no browser here is a skip, not a failure
                pytest.skip(f"no headless browser here: {why}")
            try:
                page = await browser.new_page()
                await page.set_content(_page(6.0))
                return await page.evaluate(WHAT_THE_PAGE_SAYS)
            finally:
                await browser.close()

    said = asyncio.run(look())
    assert said["length"] == 673
    assert said["name"] == "I built a glove that Explodes whatever it Punches"
    assert "an advert plays first" in said["words"]
