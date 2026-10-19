"""A conversation on a page is held to its end: each new line from the other side answered, written and sent, on a
page Aura has never seen. The words here come from a stand-in; in her they are her own."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PAGE = Path(__file__).parent / "fixtures/conversations/a_stranger_to_talk_with.html"


@pytest.mark.asyncio
async def test_each_new_line_is_answered_until_the_place_to_write_is_gone():
    playwright_api = pytest.importorskip("playwright.async_api")
    from core.skills.talking_on_a_page import talk_on_the_page

    asked: list[list[str]] = []

    async def reply(lines, page_text):
        asked.append([line.text for line in lines])
        assert page_text  # the page's own words travel with the conversation
        return f"line {len(asked)} from me"

    async with playwright_api.async_playwright() as playwright:
        try:
            browser = await playwright.chromium.launch(headless=True)
        except Exception as why:  # noqa: BLE001 - no engine installed here
            pytest.skip(f"no browser engine: {why}")
        page = await browser.new_page()
        await page.goto(f"file://{PAGE}?seconds=8")
        held = await talk_on_the_page(page, reply, until=time.monotonic() + 20)
        theirs = await page.evaluate("window.__chat.theirs")
        ours = await page.evaluate("window.__chat.ours")
        await browser.close()
    assert held["over"] and "Was that a person or a machine?" in " ".join(held["page_text"])
    assert ours == held["sent"] and len(ours) >= 3                    # each line written was sent and landed
    assert asked[0] == ["hey, how's it going"]                         # answered from the first line they wrote
    assert held["lines"][:3] == [("them", theirs[0]), ("me", ours[0]), ("them", theirs[1])]


@pytest.mark.asyncio
async def test_a_line_her_mind_gave_no_words_for_is_asked_again_once():
    """A reply that came back empty (a busy lane, a decode out of time) is asked for again, not left until they
    write again; and only once more."""
    playwright_api = pytest.importorskip("playwright.async_api")
    from core.skills.talking_on_a_page import ASKED_AT_MOST, talk_on_the_page

    asked: list[int] = []

    async def reply(lines, page_text):
        asked.append(sum(1 for line in lines if line.who == "them"))
        return "" if len(asked) == 1 else ("" if asked[-1] == 2 else f"line {len(asked)} from me")

    async with playwright_api.async_playwright() as playwright:
        try:
            browser = await playwright.chromium.launch(headless=True)
        except Exception as why:  # noqa: BLE001 - no engine installed here
            pytest.skip(f"no browser engine: {why}")
        page = await browser.new_page()
        await page.goto(f"file://{PAGE}?seconds=8")
        held = await talk_on_the_page(page, reply, until=time.monotonic() + 20)
        await browser.close()
    assert asked[:2] == [1, 1]                       # their first line, asked for again when the first try gave nothing
    assert asked.count(2) == ASKED_AT_MOST           # their second line given nothing twice, then let go
    assert held["sent"][0] == "line 2 from me"
