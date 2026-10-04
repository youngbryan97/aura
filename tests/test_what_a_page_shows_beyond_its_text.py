"""What a page shows that its plain text misses: a player's words in its shadow root, where a link goes, and that a frame is not a button.

LIVE 2026-10-03 20:04 she pressed a frame titled "ad" six times as a game's
start. Offline 2026-10-04 a game player's "failed to load" sat over the page
and her reading of it never saw the words, and the link to the game's other
copy was named only after the game.

Served at https addresses through route interception, as in
test_a_search_box_is_not_the_page.py.
"""
from __future__ import annotations

import pytest

from core.skills.sovereign_browser import SovereignBrowserSkill

PAGE = """<html><body><main>
  <h1>A game</h1>
  <game-player id="p"></game-player>
  <h2>The game elsewhere</h2>
  <a href="https://archive.example.org/details/the-game">The game in 2003</a>
  <a href="/about">About</a>
  <iframe title="ad" tabindex="0" src="about:blank" style="width:300px;height:250px"></iframe>
</main>
<script>
  const host = document.getElementById('p');
  const root = host.attachShadow({mode: 'open'});
  root.innerHTML = '<div style="width:400px;height:300px"><p>Something went wrong. The game file failed to load.</p></div>';
</script></body></html>"""


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
        await instance.page.route("**/aura-player", _serve)
        yield instance
    finally:
        await instance.close()


@pytest.mark.asyncio
async def test_a_players_own_words_are_part_of_what_the_page_says(browser):
    await browser.page.goto("https://example.com/aura-player", wait_until="load")
    observation = await browser.observe(principal="owner")
    assert "game-player shows: Something went wrong. The game file failed to load." in observation["text"]


@pytest.mark.asyncio
async def test_a_link_off_the_site_says_where_it_goes_and_a_frame_is_no_control(browser):
    await browser.page.goto("https://example.com/aura-player", wait_until="load")
    observation = await browser.observe(principal="owner")
    elsewhere = next(e for e in observation["elements"] if e.get("name") == "The game in 2003")
    assert elsewhere.get("goes_to") == "archive.example.org"
    about = next(e for e in observation["elements"] if e.get("name") == "About")
    assert "goes_to" not in about
    assert not any(e.get("role") == "iframe" or e.get("name") == "ad" for e in observation["elements"])
    rendered = SovereignBrowserSkill._render_observation(observation, "play the game")
    assert "The game in 2003 (to archive.example.org)" in rendered
