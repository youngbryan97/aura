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
    def __init__(self, refusing: tuple[str, ...] = ()) -> None:
        self.pursued: list[tuple[str, str]] = []
        self.said: list[str] = []
        self.opened: list[str] = []
        self.refusing = refusing

    async def _handle_pursue(self, browser, url, goal, max_steps, *, action_context=None, said_before=""):
        self.pursued.append((url, goal))
        return {"ok": True, "completed": True, "concluded": f"done with {url}"}

    async def _safe_browse(self, browser, url):
        if any(r in url for r in self.refusing):
            return False
        self.opened.append(url)
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
    assert skill.opened[1:] == [f"https://example.org/games/{n}" for n in (23, 42, 5)]
    assert [url for url, _goal in skill.pursued] == [None, None, None]  # each pursued where it was opened
    assert all(goal.startswith("Play this game and win it.") for _url, goal in skill.pursued)
    assert "the minute is 23; 23 divided by 56 leaves 23: number 23, “Game 23”" in skill.said[0]
    assert done["completed"] and [p["item"] for p in done["picked"]] == ["Game 23", "Game 42", "Game 5"]


@pytest.mark.unit
def test_an_item_whose_page_cannot_be_had_is_found_wherever_else_it_is(monkeypatch):
    import core.skills.sovereign_browser_going as going
    import core.skills.sovereign_browser_picking as picking

    async def the_list(browser):
        return [{"text": f"Game {n}", "href": f"https://example.org/games/{n}"} for n in range(3)]

    async def no_copy(skill, browser, url):
        return ""

    async def kept(skill, browser, name, task=""):
        return f"https://games.example.net/{name.lower().replace(' ', '-')}" if name == "Game 1" else ""

    monkeypatch.setattr(picking, "the_list_on_the_page", the_list)
    monkeypatch.setattr(going, "the_archived_copy", no_copy)
    monkeypatch.setattr(going, "the_same_thing_elsewhere", kept)
    skill = _Skill(refusing=("example.org/games/",))
    asked = "Go to https://example.org/list and play two of the games. Number the games from 0. Start at 1, take the remainder. Then add 1 and take the remainder again."
    done = asyncio.run(picking.pursued(skill, object(), "https://example.org/list", asked, 30))
    assert any("I found it at games.example.net, where it runs in the page" in line for line in skill.said)
    # Game 2 is kept nowhere, and is said to be so; the rule goes on to Game 0, kept nowhere either, and then only comes back round.
    assert [(p["item"], p["ok"]) for p in done["picked"]] == [("Game 1", True), ("Game 2", False), ("Game 0", False)]
    assert any("nor anywhere I can find it kept. To keep to two, the rule goes on" in line for line in skill.said)
    assert not done["completed"]


@pytest.mark.unit
@pytest.mark.parametrize(("asked", "kept_nowhere", "played"), [
    # The demo's rule at 1:53: 53, 16, 35 of 56; the 53rd is kept nowhere, so 35 + 19 = 54 is played after the others.
    ("Go to https://example.org/games and play three of the games, one after another, and win each one. To pick them: number the games on the "
     "list from 0. Take the current minute of the hour, divide it by how many games there are, and play the game whose number is the remainder. "
     "When that game is over, go back to the list, add 19 to the number, take the remainder again, and play that game. Then add 19 once more "
     "for the third game.", {"Game 53"}, ["Game 16", "Game 35", "Game 54"]),
    # Another family: recipes counted from 1, a different start and step; the second is kept nowhere.
    ("Go to https://example.org/recipes and cook two of the recipes. Number the recipes from 1. Start at 4, take the remainder by how many "
     "recipes there are, and cook that recipe. Then add 10 and take the remainder again.", {"Game 13"}, ["Game 3", "Game 23"]),
])
def test_a_pick_not_to_be_had_anywhere_gives_its_place_to_the_next_the_rule_makes(monkeypatch, asked, kept_nowhere, played):
    import datetime

    import core.language.picking_by_a_rule as rule
    import core.skills.sovereign_browser_picking as picking

    async def the_list(browser):
        return [{"text": f"Game {n}", "href": f"https://example.org/games/{n}"} for n in range(56)]

    async def nowhere(skill, browser, url, name, task=""):
        return "" if name in kept_nowhere else url

    class _At(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.datetime(2026, 10, 6, 13, 53)

    monkeypatch.setattr(picking, "the_list_on_the_page", the_list)
    monkeypatch.setattr(picking, "_the_item_itself", nowhere)
    monkeypatch.setattr(rule.datetime, "datetime", _At)
    skill = _Skill()
    done = asyncio.run(picking.pursued(skill, object(), "https://example.org/games", asked, 30))
    assert [p["item"] for p in done["picked"] if p["had"]] == played
    assert done["completed"]
    assert any("the rule goes on" in line and f"“{played[-1]}”" in line for line in skill.said)
    assert f"“{sorted(kept_nowhere)[0]}” could not be had anywhere." in done["concluded"]


@pytest.mark.unit
def test_a_window_closed_under_her_ends_the_picks_and_is_said(monkeypatch):
    import core.skills.sovereign_browser_picking as picking

    class _Page:
        closed = False

        def is_closed(self):
            return self.closed

    class _Browser:
        page = _Page()

    async def the_list(browser):
        return [{"text": f"Game {n}", "href": f"https://example.org/games/{n}"} for n in range(5)]

    browser = _Browser()
    skill = _Skill()
    pursue = skill._handle_pursue

    async def closed_after(browser_, url, goal, max_steps, **kw):
        browser.page.closed = True  # the person closes the window during the first game
        return await pursue(browser_, url, goal, max_steps, **kw)

    monkeypatch.setattr(picking, "the_list_on_the_page", the_list)
    skill._handle_pursue = closed_after
    asked = "Go to https://example.org/list and play three of the games. Number the games from 0. Start at 1, take the remainder. Then add 1 and take the remainder again."
    done = asyncio.run(picking.pursued(skill, browser, "https://example.org/list", asked, 30))
    assert [p["item"] for p in done["picked"]] == ["Game 1"]
    assert any("has been closed, so I stop here, with two of the picks not done" in line for line in skill.said)


@pytest.mark.unit
def test_a_page_that_arrives_and_does_not_work_is_left_for_where_the_same_thing_does(monkeypatch):
    """Sent to a page to play something, she finds it does not work there, says why, and plays it where it does."""
    import core.skills.sovereign_browser_going as going
    import core.skills.sovereign_browser_picking as picking
    import core.skills.whether_a_page_serves as serving

    class _Page:
        url = "https://example.org/broken-game"

        async def evaluate(self, script, *args):
            return "Tunnel Rush"

    class _Browser:
        page = _Page()

    async def judged(page, task, **kw):
        return serving.Serves(page.url != "https://games.example.net/tunnel-rush" and False, ["its game file could not be had (answered 404: game.swf)"])

    async def elsewhere(skill, browser, name, task="", not_at=""):
        found = going.Where("https://games.example.net/tunnel-rush")
        found.serves = serving.Serves(True, seen="its canvas draws, and moves")
        return found

    monkeypatch.setattr(serving, "whether_it_serves", judged)
    monkeypatch.setattr(serving, "watching", lambda page: None)
    monkeypatch.setattr(going, "the_same_thing_elsewhere", elsewhere)
    skill = _Skill()
    asyncio.run(picking.pursued(skill, _Browser(), "https://example.org/broken-game", "Play Tunnel Rush and win it.", 30))
    assert skill.pursued == [(None, "Play Tunnel Rush and win it.")]  # pursued where it was found, not where it was sent
    assert any("does not work: its game file could not be had" in line for line in skill.said)
    assert any("I found it at games.example.net, and there its canvas draws, and moves." in line for line in skill.said)


def test_each_picks_account_is_how_it_ended_not_done():
    # LIVE 2026-10-07 three games each ended "Played, and not won", and the account of the three said each was "done".
    from core.skills.sovereign_browser_picking import _how_it_ended

    steps = [{"landed": 3}, {"why": "Played, and not won; I could not get further with it.", "done": True}]
    assert _how_it_ended({"steps": steps}) == "played, and not won; I could not get further with it"
    assert _how_it_ended({"concluded": "It was won.", "steps": steps}) == "it was won"
    assert _how_it_ended({"steps": [{"landed": 1}]}) == ""
