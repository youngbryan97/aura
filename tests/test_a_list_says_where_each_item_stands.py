"""Each item of a list of like controls carries its place in the list and the list's length.

LIVE 2026-10-03 19:58, asked to take the minute modulo the number of games on
the Cartoon Network list and play that game, she counted 58 games where there
are 56 and opened the wrong one. The count is a fact of the page; the observer
measures it and the decision is shown it.

Served at https addresses through route interception, as in
test_a_search_box_is_not_the_page.py.
"""
from __future__ import annotations

import pytest

from core.skills.sovereign_browser import SovereignBrowserSkill

TILES = "".join(
    f"<div class='tile'><a class='card' href='/game-{n}'><h3>Game number {n}</h3><span>{2000 + n % 15}</span></a></div>"
    for n in range(56)
)
PAGE = f"""<html><body>
  <nav><a class='nav' href='/'>Home</a><a class='nav' href='/about'>About</a><a class='nav' href='/games'>Games</a></nav>
  <main><h1>Flash games</h1><div class='grid'>{TILES}</div></main>
  <footer><a href='/privacy'>Privacy</a></footer></body></html>"""


async def _serve(route) -> None:
    await route.fulfill(status=200, content_type="text/html", body=PAGE)


@pytest.fixture
async def browser():
    from core.capabilities.phantom_browser import PhantomBrowser

    instance = PhantomBrowser(visible=False, browser_type="chromium", principal="owner")
    if not await instance.ensure_ready():
        await instance.close()
        pytest.skip("no browser engine available in this environment")
    try:
        await instance.page.route("**/aura-list", _serve)
        yield instance
    finally:
        await instance.close()


@pytest.mark.asyncio
async def test_each_tile_knows_its_place_among_fifty_six(browser):
    await browser.page.goto("https://example.com/aura-list", wait_until="load")
    observation = await browser.observe(principal="owner")
    tiles = [e for e in observation["elements"] if str(e.get("name", "")).startswith("Game number")]
    assert tiles, observation["elements"][:5]
    assert tiles[0]["alike"] == [1, 56]
    assert tiles[20]["alike"] == [21, 56]
    nav = [e for e in observation["elements"] if e.get("name") == "About"]
    assert nav and "alike" not in nav[0]


@pytest.mark.asyncio
async def test_her_decision_is_shown_the_place(browser):
    await browser.page.goto("https://example.com/aura-list", wait_until="load")
    observation = await browser.observe(principal="owner")
    rendered = SovereignBrowserSkill._render_observation(observation, "play game number 20")
    assert "Game number 20 2005 (21 of 56 alike)" in " ".join(rendered.split()), rendered[:3000]
