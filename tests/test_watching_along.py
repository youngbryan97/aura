"""What a page plays is offered to her as one thing on it, and choosing it watches it with whoever asked: before it
plays she says what she expects, while it plays she looks and reads its captions and says what she thinks when there
is something new, and when it ends she says what it was. A real film plays here; her eyes and her turns are stand-ins.
In her they are her vision model and her own cognitive engine."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

PAGE = Path(__file__).parent / "fixtures/watching/a_page_that_plays.html"
ASKED = "Watch this with me. Before it starts, what do you think it'll be about? Then think out loud as it plays."


async def _a_page(playwright_api):
    playwright = await playwright_api.async_playwright().start()
    try:
        browser = await playwright.chromium.launch(headless=True, args=["--autoplay-policy=no-user-gesture-required"])
    except Exception as why:  # noqa: BLE001 - no engine installed here
        await playwright.stop()
        pytest.skip(f"no browser engine: {why}")
    page = await browser.new_page(viewport={"width": 800, "height": 600})
    await page.goto(f"file://{PAGE}")
    await page.wait_for_function("document.getElementById('film').readyState >= 1")
    return playwright, browser, page


@pytest.mark.asyncio
async def test_what_a_page_plays_is_offered_and_handed_over_to_watching():
    playwright_api = pytest.importorskip("playwright.async_api")
    from core.skills.sovereign_browser_on_the_page import WATCHING, handed_over, watched_on_the_page, with_what_plays

    playwright, browser, page = await _a_page(playwright_api)
    try:
        offered = await with_what_plays(SimpleNamespace(page=page), {"elements": [{"role": "button"}]})
        await page.set_content("<p>nothing plays here</p>")
        nothing = await with_what_plays(SimpleNamespace(page=page), {"elements": [{"role": "button"}]})
    finally:
        await browser.close()
        await playwright.stop()
    assert offered["elements"][-1]["selector"] == WATCHING and "0:06 long" in offered["elements"][-1]["name"]
    assert [e["role"] for e in nothing["elements"]] == ["button"]
    assert handed_over([(SimpleNamespace(selector=WATCHING), "")]) is watched_on_the_page
    assert handed_over([(SimpleNamespace(selector="#cc"), "")]) is None


@pytest.mark.asyncio
async def test_she_watches_a_film_through_saying_what_she_thinks_before_while_and_after():
    playwright_api = pytest.importorskip("playwright.async_api")
    from core.skills.watching_along import watch_along

    turns: list[tuple[str, list[str]]] = []
    sensed: list[tuple[str, str]] = []
    kept: list[dict] = []

    async def look(when: str) -> str:
        return "a test pattern of coloured bars" if when else "a page with a dark player and a title"

    async def turn(request: str, observation: str, said: list[str]) -> str:
        assert request == ASKED  # the person's request is the turn, every time
        turns.append((observation, list(said)))
        return f"thought {len(turns)}"

    async def say(words: str) -> bool:
        return True

    playwright, browser, page = await _a_page(playwright_api)
    try:
        watching = await watch_along(page, ASKED, look=look, turn=turn, say=say,
                                     perceive=lambda seen, read: sensed.append((seen, read)), keep=kept.append,
                                     look_every_s=0.5, quiet_at_least_s=1.5, at_most_s=25.0)
        pressed = await page.evaluate("document.getElementById('cc').getAttribute('aria-pressed')")
    finally:
        await browser.close()
        await playwright.stop()

    assert watching.ended and watching.title == "A short film about colour"
    assert turns[0][0].startswith("before it plays") and "A short film about colour" in turns[0][0]
    assert turns[-1][0].startswith("it has ended")
    during = [observation for observation, _said in turns[1:-1]]
    assert during and all(o.startswith("playing, at ") for o in during)  # she spoke while it played, more than once
    assert pressed == "true"  # its own captions control was pressed
    assert any("words shown:" in o for o in during)  # and she read them as it played
    assert turns[-1][1] == [f"thought {n}" for n in range(1, len(turns))]  # what she said so far was hers, in order
    assert sensed and all(seen == "a test pattern of coloured bars" for seen, _read in sensed)
    assert {row["kind"] for row in kept} >= {"before", "moment", "said", "after"}


def test_asked_to_watch_she_watches_first_once_a_page(monkeypatch):
    """A video starts when its page opens: asked to watch it, she watches before reading the page, once a page."""
    import core.skills.sovereign_browser_on_the_page as on_the_page

    monkeypatch.setattr(on_the_page, "_WATCHED_HERE", set())
    page = {"url": "https://videos.example/watch?v=1", "elements": [{"role": "button"}, {"role": "watching", "selector": on_the_page.WATCHING}]}
    assert on_the_page.watched_first(page, ASKED, take=False)["resolved_actions"][0]["selector"] == on_the_page.WATCHING
    assert on_the_page.watched_first(page, "Summarise what this page says.") is None  # not asked to watch
    assert on_the_page.watched_first({**page, "elements": [{"role": "button"}]}, ASKED) is None  # nothing plays
    assert on_the_page.watched_first(page, ASKED) is not None
    assert on_the_page.watched_first(page, ASKED) is None  # once a page: after it, her model decides
