"""A conversation on a page is read for who said what, what is new from the other side, and when it ends, on a page
Aura has never seen and that names nothing of any site: a stranger who greets, answers each line after a pause,
and asks for a guess when the time runs out."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from core.perception.a_conversation_on_a_page import LOOK, read_conversation

pytestmark = pytest.mark.unit

PAGE = Path(__file__).parent / "fixtures/conversations/a_stranger_to_talk_with.html"


@pytest.mark.asyncio
async def test_who_said_what_is_read_and_the_end_is_the_place_to_write_gone():
    playwright_api = pytest.importorskip("playwright.async_api")
    async with playwright_api.async_playwright() as playwright:
        try:
            browser = await playwright.chromium.launch(headless=True)
        except Exception as why:  # noqa: BLE001 - no engine installed here
            pytest.skip(f"no browser engine: {why}")
        page = await browser.new_page()
        await page.goto(f"file://{PAGE}?seconds=6")
        first = await page.evaluate(LOOK)
        await asyncio.sleep(1.2)
        second = await page.evaluate(LOOK)
        greeted = read_conversation(second, [], before=first)
        assert greeted.theirs() == ["hey, how's it going"] and not greeted.over
        assert "When the time runs out" in " ".join(greeted.page_text)       # the page's words are not a line of the talk

        await page.fill("input", "not bad, just got home")
        await page.keyboard.press("Enter")
        await asyncio.sleep(1.6)
        talk = read_conversation(await page.evaluate(LOOK), ["not bad, just got home"], stream=greeted.stream)
        assert [(line.who, line.text) for line in talk.lines] == [
            ("them", "hey, how's it going"), ("me", "not bad, just got home"), ("them", "ha, fair enough")]

        await asyncio.sleep(5.0)
        ended = read_conversation(await page.evaluate(LOOK), ["not bad, just got home"], stream=greeted.stream)
        assert ended.over and "Was that a person or a machine?" in " ".join(ended.page_text)
        await browser.close()


def test_her_line_finds_the_stream_without_a_look_before():
    look = {"writing": {"x": 0, "y": 500, "w": 400, "h": 30}, "blocks": [
        {"text": "Chat", "path": "header:1/span:1", "control": False},
        {"text": "hello?", "path": "main:1/div:1/div:1/p:1", "control": False},
        {"text": "hi, who's this", "path": "main:1/div:1/div:2/p:1", "control": False},
        {"text": "Send", "path": "form:1/button:1", "control": True}]}
    talk = read_conversation(look, ["Hi, who's this"])
    assert talk.stream == "main:1/div:1"
    assert [(line.who, line.text) for line in talk.lines] == [("them", "hello?"), ("me", "hi, who's this")]
    assert talk.page_text == ["Chat"]
