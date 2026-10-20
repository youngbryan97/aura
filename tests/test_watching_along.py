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
    from core.skills.sovereign_browser_on_the_page import (
        WATCHING,
        handed_over,
        watched_on_the_page,
        with_what_plays,
    )

    playwright, browser, page = await _a_page(playwright_api)
    try:
        offered = await with_what_plays(SimpleNamespace(page=page), {"elements": [{"role": "button"}]})
        await page.set_content("<p>nothing plays here</p>")
        nothing = await with_what_plays(SimpleNamespace(page=page), {"elements": [{"role": "button"}]})
    finally:
        await browser.close()
        await playwright.stop()
    assert offered["elements"][-1]["selector"] == WATCHING and "0:06 long" in offered["elements"][-1]["name"]
    assert offered["elements"][-1]["share_of_view"] == 0.48  # 640 by 360 in an 800 by 600 view
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
        remembered: list = []

        async def remember(watched):
            remembered.append(watched)

        watching = await watch_along(page, ASKED, look=look, turn=turn, say=say, remember=remember,
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
    assert any("heard:" in o for o in during)  # and she heard what was said as it played
    assert turns[-1][1] == [f"thought {n}" for n in range(1, len(turns))]  # what she said so far was hers, in order
    assert sensed and all(seen == "a test pattern of coloured bars" for seen, _read in sensed)
    assert {row["kind"] for row in kept} >= {"before", "moment", "said", "after"}
    assert remembered == [watching]  # kept as an episode of hers, to recall what she thought of it


def test_asked_to_watch_she_watches_first_once_a_request(monkeypatch):
    """A video starts when its page opens: asked to watch it, she watches before reading the page, once a request,
    so the next video a site plays by itself at a new address is not watched on its own."""
    import core.skills.sovereign_browser_on_the_page as on_the_page

    monkeypatch.setattr(on_the_page, "_WATCHED_FOR", {})
    page = {"url": "https://videos.example/watch?v=1",
            "elements": [{"role": "button"}, {"role": "watching", "selector": on_the_page.WATCHING, "share_of_view": 0.293}]}
    trailer = {**page, "elements": [{"role": "watching", "selector": on_the_page.WATCHING, "share_of_view": 0.128}]}
    assert on_the_page.watched_first(trailer, ASKED) is None  # a channel's trailer is not what the page is for
    assert on_the_page.watched_first(page, ASKED, take=False)["resolved_actions"][0]["selector"] == on_the_page.WATCHING
    assert on_the_page.watched_first(page, "Summarise what this page says.") is None  # not asked to watch
    assert on_the_page.watched_first({**page, "elements": [{"role": "button"}]}, ASKED) is None  # nothing plays
    assert on_the_page.watched_first(page, ASKED) is not None
    assert on_the_page.watched_first(page, ASKED) is None  # once a request: after it, her model decides
    assert on_the_page.watched_first({**page, "url": "https://videos.example/watch?v=2"}, ASKED) is None  # the next one


def test_a_page_handed_to_her_to_watch_is_watched_first_however_small_its_video(monkeypatch):
    """LIVE 20 October 2026: asked to watch a page whose 640-pixel film was under the share of her window, she
    described the page instead. A page the request names is what the page is for."""
    import core.skills.sovereign_browser_on_the_page as on_the_page

    monkeypatch.setattr(on_the_page, "_WATCHED_FOR", {})
    small = {"url": "http://127.0.0.1:8765/watching/a_page_that_plays.html",
             "elements": [{"role": "watching", "selector": on_the_page.WATCHING, "share_of_view": 0.13}]}
    elsewhere = {**small, "url": "http://127.0.0.1:8765/watching/another_page.html"}
    asked = "Watch this with me: http://127.0.0.1:8765/watching/a_page_that_plays.html. Tell me what you think."
    assert on_the_page.watched_first(elsewhere, asked) is None  # a page it does not name: the share decides
    assert on_the_page.watched_first(small, asked)["resolved_actions"][0]["selector"] == on_the_page.WATCHING


@pytest.mark.asyncio
async def test_an_advert_ending_first_does_not_end_the_watching(monkeypatch):
    """A site that plays an advert first plays it in the same video, which reports itself ended when the advert
    does and then plays the video asked for."""
    import core.skills.watching_along as watching_along

    monkeypatch.setattr(watching_along, "AN_END_HOLDS_S", 0.0)
    states = iter([{"at": 5.0, "duration": 15.0}, {"ended": True},  # the advert ends...
                   {"at": 1.0, "duration": 60.0}, {"at": 2.0, "duration": 60.0},  # ...and the video asked for plays
                   {"at": 30.0, "duration": 60.0}, {"ended": True}, {"ended": True}])

    class Page:
        async def evaluate(self, script, *_):
            if script == watching_along.WHAT_PLAYS:
                return {"x": 0, "y": 0, "w": 640, "h": 360, "vw": 800, "vh": 600, "words": "", "paused": False,
                        "ended": False, **next(states)}
            return {"title": "A video"} if script == watching_along.WHAT_THE_PAGE_SAYS else ""

    async def look(_when):
        return "a picture"

    said = []

    async def turn(_request, observation, _said):
        said.append(observation.split(".")[0])
        return "a thought"

    async def say(_words):
        return True

    watching = await watching_along.watch_along(Page(), ASKED, look=look, turn=turn, say=say,
                                                perceive=lambda *_: None, look_every_s=0.0, quiet_at_least_s=0.0)
    assert watching.ended
    assert [m.at for m in watching.moments] == [5.0, 2.0, 30.0]  # it watched on past the advert's end


@pytest.mark.asyncio
async def test_a_watching_becomes_an_episode_of_hers_in_her_own_words(monkeypatch):
    from core.container import ServiceContainer
    from core.skills.watching_along import Watching, into_her_memory

    episodes: list[dict] = []

    class Episodic:
        async def record_episode_async(self, **episode):
            episodes.append(episode)
            return "e1"

    monkeypatch.setattr(ServiceContainer, "get", classmethod(lambda _cls, name, default=None: Episodic() if name == "episodic_memory" else default))
    watched = Watching(request=ASKED, title="A short film about colour", said=[(0.0, "Colour bars, I bet."), (3.0, "There it goes.")], ended=True)
    await into_her_memory(watched, url="https://videos.example/watch?v=1")
    assert episodes[0]["action"] == "watching_along" and "A short film about colour" in episodes[0]["context"]
    assert episodes[0]["outcome"] == "0:00 Colour bars, I bet.\n0:03 There it goes."  # her words, as said
    assert episodes[0]["metadata"]["provenance"] == "her own words while watching"


@pytest.mark.asyncio
async def test_everything_said_between_two_looks_is_heard():
    """LIVE 19 October 2026: looking every five seconds caught one caption line in two. She listens four times a
    second between looks, so a look three seconds apart still hears all three lines of the film."""
    playwright_api = pytest.importorskip("playwright.async_api")
    from core.skills.watching_along import watch_along

    async def look(_when):
        return "bars of colour"

    async def turn(_request, _observation, _said):
        return ""

    async def say(_words):
        return True

    playwright, browser, page = await _a_page(playwright_api)
    try:
        watching = await watch_along(page, ASKED, look=look, turn=turn, say=say, remember=lambda _w: _none(),
                                     perceive=lambda *_: None, look_every_s=3.0, quiet_at_least_s=100.0, at_most_s=20.0)
    finally:
        await browser.close()
        await playwright.stop()
    heard = " ".join(m.read for m in watching.moments)
    assert all(line in heard for line in ("the colours begin", "a moving pattern", "and now the end comes")), heard


async def _none():
    return None


def test_she_speaks_when_more_is_new_than_when_she_last_spoke_and_at_a_lull():
    from core.skills.watching_along import Moment, _WhatIsNew

    new = _WhatIsNew()
    new.took_in(Moment(at=1.0, seen="a planet turning", read="this is the planet earth"))
    assert new.worth_a_remark()  # the first remark in play needs only something new
    new.remarked()
    new.took_in(Moment(at=6.0, seen="a planet turning", read="the planet earth"))
    assert not new.worth_a_remark(), "nothing new since she spoke"
    new.took_in(Moment(at=11.0, seen="a volcano erupting into the sky", read="volcanoes reshape continents over aeons"))
    assert new.worth_a_remark()
    busy = _WhatIsNew()
    for n in range(4):
        busy.took_in(Moment(at=n, read="one two"))
    busy.took_in(Moment(at=5.0, seen="glaciers", read="glaciers carve valleys through mountains and leave lakes behind them"))
    assert not busy.worth_a_remark(), "not while it is saying more than usual: she waits for a lull"


@pytest.mark.asyncio
async def test_watched_with_them_it_goes_on_beside_and_the_browsing_is_done(monkeypatch):
    """LIVE 20 October 2026: once the watching had gone on beside the conversation, her browsing went another round
    and asked her model what to do on a page she had handed over. The step says it goes on beside."""
    import core.skills.sovereign_browser_on_the_page as on_the_page
    import core.skills.watching_the_screen as screen
    import core.skills.watching_together as together

    started, evaluated = [], []

    class _Page:
        async def evaluate(self, script):
            evaluated.append(script)
            return {"duration": 6.0, "w": 640, "h": 360} if script is on_the_page.WHAT_PLAYS else {"words": "a short film"}

        async def title(self):
            return "A short film about colour"

    monkeypatch.setattr(screen, "_their_browser", lambda _url: ("com.example.Browser", "Browser"))
    monkeypatch.setattr(together, "start_watching_in_a_with", lambda *a, **k: started.append(k))
    step = await on_the_page.watched_on_the_page(SimpleNamespace(page=_Page()), ASKED,
                                                 {"url": "http://127.0.0.1:8765/watching/a_page_that_plays.html"})
    assert step["beside"] and step["over"] and started and started[0]["title"] == "A short film about colour"
    assert on_the_page.QUIET_HERE in evaluated  # her own page quiet: it plays where they watch
