"""Whether a page serves what a task needs, from its developer tools, its code and its words, and settled by looking.

LIVE 2026-10-06: game pages that opened and could not be played were taken as the game. Arriving is not working.
Every case here is a page served from a made-up site whose every answer the test decides: a status, a missing file.
"""
from __future__ import annotations

import asyncio

import pytest

from core.skills.whether_a_page_serves import watching, whether_it_serves

pytestmark = pytest.mark.slow

_ANIMATES = """<canvas id=c width=480 height=320></canvas><script>
const g = document.getElementById('c').getContext('2d'); let t = 0;
setInterval(() => { g.fillStyle = '#123'; g.fillRect(0, 0, 480, 320); g.fillStyle = '#fc0'; g.fillRect(40 + (t++ * 9) % 380, 140, 40, 40); }, 30);
</script>"""
_BLANK_AND_THROWS = """<canvas width=480 height=320 style="background:#000"></canvas><script>window.setTimeout(() => { null.start(); }, 10)</script>"""
_FLASH_WITHOUT_PLAYER = """<h1>Tunnel Rush</h1><object type="application/x-shockwave-flash" data="/games/tunnel.swf" width=550 height=400>
<param name=movie value="/games/tunnel.swf"></object>"""
_GAME_FILE_GONE = """<h1>A game</h1><canvas width=480 height=320></canvas><script>fetch('/build/game.wasm')</script>"""
_VIDEO_GONE = """<h1>A film</h1><video src="/films/gone.mp4" width=480 height=270 autoplay muted></video>"""
_PLAYER_SAYS_IT_FAILED = """<h1>A game</h1><her-player style="display:block;width:480px;height:320px"></her-player><script>
customElements.define('her-player', class extends HTMLElement { constructor() { super(); this.attachShadow({mode: 'open'}).innerHTML =
'<div style="width:480px;height:320px;background:#222;color:#fff">Something went wrong: this content could not be loaded.</div>'; } });
</script><canvas width=10 height=10></canvas>"""
_AN_ARTICLE = "<h1>Bees</h1>" + "<p>" + "Bees gather nectar and make honey in the hive through the summer. " * 30 + "</p>"
_NOT_FOUND = "<h1>Page not found</h1><p>The page you asked for has been removed.</p>"


def _judge(body: str, task: str, *, status: int = 200, missing: tuple[str, ...] = ()) -> object:
    from playwright.async_api import async_playwright

    async def go():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            try:
                page = await browser.new_page(viewport={"width": 900, "height": 700})
                watching(page)

                async def answer(route):
                    path = route.request.url.split("test.example", 1)[-1]
                    if path in missing:
                        await route.fulfill(status=404, body="not here")
                    elif path in ("/", ""):
                        await route.fulfill(status=status, content_type="text/html", body=f"<!doctype html><title>t</title><body>{body}</body>")
                    else:
                        await route.fulfill(status=404, body="not here")

                await page.route("http://test.example/**", answer)
                await page.goto("http://test.example/", wait_until="load")
                await page.wait_for_timeout(400)
                return await whether_it_serves(page, task, look_for_s=0.6)
            finally:
                await browser.close()

    return asyncio.run(go())


@pytest.mark.parametrize(("body", "task", "missing", "because"), [
    (_FLASH_WITHOUT_PLAYER, "play this game", ("/games/tunnel.swf",), "plugin content"),
    (_GAME_FILE_GONE, "play this game", ("/build/game.wasm",), "could not be had"),
    (_BLANK_AND_THROWS, "play this game", (), "draws nothing"),
    (_VIDEO_GONE, "watch this film", ("/films/gone.mp4",), "could not be had"),
    (_PLAYER_SAYS_IT_FAILED, "play this game", (), "says: something went wrong"),
    (_NOT_FOUND, "read this page and tell me what it says", (), "it says: page not found"),
])
def test_a_page_that_does_not_do_what_it_is_for_is_known_and_said_why(body, task, missing, because):
    verdict = _judge(body, task, missing=missing)
    assert verdict.ok is False, verdict
    assert because in verdict.says().lower(), verdict.says()


def test_a_page_whose_document_is_not_there_does_not_serve():
    verdict = _judge(_AN_ARTICLE, "read this page", status=404)
    assert verdict.ok is False and "answered 404" in verdict.says()


@pytest.mark.parametrize(("body", "task", "seen"), [
    (_ANIMATES, "play this game", "draws, and moves"),
    (_AN_ARTICLE, "read this page and tell me what it says", "words to read"),
])
def test_a_page_seen_doing_what_it_is_for_serves(body, task, seen):
    verdict = _judge(body, task)
    assert verdict.ok is True and seen in verdict.says(), verdict


@pytest.mark.unit
def test_a_page_seen_running_is_played_first_and_only_once_unasked():
    """LIVE 2026-10-06 her model read each game's page for a minute or more before choosing to play what it draws."""
    from core.skills import whether_a_page_serves as serving
    from core.skills.sovereign_browser_drawing import DRAWING, the_way_is_the_drawing

    serving._SEEN_RUNNING["https://games.example.net/tunnel-rush"] = "its canvas draws, and moves"
    here = {"url": "https://games.example.net/tunnel-rush/"}
    assert the_way_is_the_drawing(here, "Read about this game.") is None  # only a task that runs it
    assert the_way_is_the_drawing(here, "Play this game and win it.", take=False) is not None  # looked at, not taken
    decision = the_way_is_the_drawing(here, "Play this game and win it.")
    assert decision["resolved_actions"][0]["selector"] == DRAWING and "its canvas draws, and moves" in decision["why"]
    assert the_way_is_the_drawing(here, "Play this game and win it.") is None  # once: what follows the play is hers to decide
    assert the_way_is_the_drawing({"url": "https://elsewhere.example/"}, "Play this game.") is None


def test_a_page_judged_running_is_remembered_as_seen_running():
    from core.skills.whether_a_page_serves import seen_running

    verdict = _judge(_ANIMATES, "play this game")
    assert verdict.ok and seen_running("http://test.example/") == "its canvas draws, and moves"
