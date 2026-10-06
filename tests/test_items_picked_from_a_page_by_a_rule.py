"""Items picked from a page's list by the person's rule: the list read off the page, the task said of each one."""
from __future__ import annotations

import asyncio

import pytest

from core.skills.sovereign_browser_picking import _THE_LIST, the_task_for_each


@pytest.mark.unit
@pytest.mark.parametrize(("asked", "each"), [
    ("Go to https://www.webdesignmuseum.org/flash-game-exhibitions/cartoon-network-flash-games and play three of the games, one after another, "
     "and win each one. To pick them: number the games on the list from 0.", "Play this game and win it."),
    ("Go to https://example.org/recipes and cook two of the recipes and tell me how each one tastes. Number the recipes from 1.",
     "Cook this recipe and tell me how it tastes."),
    ("Open the playlist and play 3 songs. Take the current minute of the hour.", "Open the playlist and play this song."),
])
def test_the_task_is_said_of_one_item(asked, each):
    assert the_task_for_each(asked) == each


_PAGE = """<!doctype html><html><body><header><nav><a href="/">Home</a><a href="/about">About</a><a href="/shop">Shop</a></nav></header>
<main><h1>Games</h1><div class="grid">""" + "".join(
    f'<div class="cell"><article class="card"><a href="/games/{n}"><img alt=""><h3>Game {n}</h3><span>200{n % 9}</span></a></article></div>'
    for n in range(12)) + """</div><p>See also <a href="/more">more</a> and <a href="/less">less</a>.</p></main>
<footer><a href="/a">a</a><a href="/b">b</a><a href="/c">c</a><a href="/d">d</a></footer></body></html>"""


@pytest.mark.slow
def test_the_list_is_the_pages_repeated_links_in_reading_order(tmp_path):
    from playwright.async_api import async_playwright

    page_file = tmp_path / "games.html"
    page_file.write_text(_PAGE, "utf-8")

    async def read() -> list[dict]:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            try:
                page = await browser.new_page()
                await page.goto(page_file.as_uri())
                return await page.evaluate(_THE_LIST)
            finally:
                await browser.close()

    items = asyncio.run(read())
    assert [i["text"] for i in items] == [f"Game {n}" for n in range(12)]
    assert items[3]["href"].endswith("/games/3")
