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


class _Skill:
    def __init__(self) -> None:
        self.pursued: list[tuple[str, str]] = []
        self.said: list[str] = []

    async def _handle_pursue(self, browser, url, goal, max_steps, *, action_context=None, said_before=""):
        self.pursued.append((url, goal))
        return {"ok": True, "completed": True, "concluded": f"done with {url}"}

    async def _safe_browse(self, browser, url):
        return True

    def _say_out_loud(self, line, parts=None):
        self.said.append(line)


@pytest.mark.unit
def test_a_goal_with_no_rule_is_one_pursuit_as_before():
    from core.skills.sovereign_browser_picking import pursued

    skill = _Skill()
    done = asyncio.run(pursued(skill, object(), "https://example.org", "Play the first game on the list.", 30))
    assert skill.pursued == [("https://example.org", "Play the first game on the list.")] and done["ok"]


@pytest.mark.unit
def test_each_item_the_rule_picks_is_pursued_in_turn(monkeypatch):
    import datetime

    import core.language.picking_by_a_rule as rule
    import core.skills.sovereign_browser_picking as picking

    async def the_list(browser):
        return [{"text": f"Game {n}", "href": f"https://example.org/games/{n}"} for n in range(56)]

    class _At(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.datetime(2026, 10, 6, 7, 23)

    monkeypatch.setattr(picking, "the_list_on_the_page", the_list)
    monkeypatch.setattr(rule.datetime, "datetime", _At)
    skill = _Skill()
    asked = ("Go to https://example.org/games and play three of the games, one after another, and win each one. To pick them: number the games "
             "on the list from 0. Take the current minute of the hour, divide it by how many games there are, and play the game whose number is "
             "the remainder. When that game is over, go back to the list, add 19 to the number, take the remainder again, and play that game. "
             "Then add 19 once more for the third game.")
    done = asyncio.run(picking.pursued(skill, object(), "https://example.org/games", asked, 30))
    assert [url for url, _goal in skill.pursued] == [f"https://example.org/games/{n}" for n in (23, 42, 5)]
    assert all(goal.startswith("Play this game and win it.") for _url, goal in skill.pursued)
    assert "the minute is 23; 23 divided by 56 leaves 23: number 23, “Game 23”" in skill.said[0]
    assert done["completed"] and [p["item"] for p in done["picked"]] == ["Game 23", "Game 42", "Game 5"]
