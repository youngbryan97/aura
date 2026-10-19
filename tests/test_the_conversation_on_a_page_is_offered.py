"""The conversation on a page is offered to her as one thing on the page, and choosing it holds the conversation.

A page with a place to write a line offers it; a search box, or a form of several fields, does not. The words here
come from a stand-in for her model; in her they are her own."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

PAGE = Path(__file__).parent / "fixtures/conversations/a_stranger_to_talk_with.html"

_NOT_A_CONVERSATION = {
    "a search box": '<form role="search"><input type="text" name="q"></form><p>results</p>',
    "a form of several fields": '<form><input type="text" name="name"><input type="email"><textarea></textarea></form>',
}


async def _a_page(playwright_api):
    playwright = await playwright_api.async_playwright().start()
    try:
        browser = await playwright.chromium.launch(headless=True)
    except Exception as why:  # noqa: BLE001 - no engine installed here
        await playwright.stop()
        pytest.skip(f"no browser engine: {why}")
    return playwright, browser, await browser.new_page()


@pytest.mark.asyncio
async def test_a_place_to_write_a_line_is_offered_and_a_form_is_not():
    playwright_api = pytest.importorskip("playwright.async_api")
    from core.skills.sovereign_browser_conversation import CONVERSATION, with_the_conversation

    playwright, browser, page = await _a_page(playwright_api)
    try:
        await page.goto(f"file://{PAGE}?seconds=30")
        offered = await with_the_conversation(SimpleNamespace(page=page), {"elements": [{"role": "button"}]})
        found = {}
        for what, html in _NOT_A_CONVERSATION.items():
            await page.set_content(html)
            found[what] = await with_the_conversation(SimpleNamespace(page=page), {"elements": [{"role": "button"}]})
    finally:
        await browser.close()
        await playwright.stop()
    assert offered["elements"][-1]["selector"] == CONVERSATION
    assert offered["elements"][-1]["role"] == "conversation"
    for what, observation in found.items():
        assert [e["role"] for e in observation["elements"]] == ["button"], what


@pytest.mark.asyncio
async def test_choosing_it_holds_the_conversation_and_says_what_came_of_it(monkeypatch):
    playwright_api = pytest.importorskip("playwright.async_api")
    import core.skills.her_reply_in_a_conversation as replying
    from core.skills.sovereign_browser_conversation import (
        CONVERSATION,
        chose_the_conversation,
        talked_on_the_page,
    )

    asked: list[str] = []

    async def stand_in(lines, *, request, page_text, **_):
        asked.append(request)
        return f"line {len(asked)} from me"

    monkeypatch.setattr(replying, "her_reply", stand_in)
    playwright, browser, page = await _a_page(playwright_api)
    try:
        await page.goto(f"file://{PAGE}?seconds=8")
        step = await talked_on_the_page(SimpleNamespace(page=page), "talk with whoever is there", {"url": page.url})
        ours = await page.evaluate("window.__chat.ours")
    finally:
        await browser.close()
        await playwright.stop()
    assert chose_the_conversation([(SimpleNamespace(selector=CONVERSATION), "")])
    assert not chose_the_conversation([(SimpleNamespace(selector="#send"), "")])
    assert step["over"] and step["moved"] and len(ours) >= 3
    assert set(asked) == {"talk with whoever is there"}  # what she was asked to do there travels with each line
    assert step["did"].startswith(f"Talked on the page: {len(ours)} lines of mine")
