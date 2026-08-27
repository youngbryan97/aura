"""Getting to the right server is not getting to the thing.

Arrival was judged on the host alone, so a page answering "The page could not
be found" counted as arrived — and a whole run anchored itself to it. She read
a not-found page, found no part of it that answered to her, and reported
truthfully that nothing on screen offered a move.

LIVE 2026-08-27: she searched for a sliding puzzle, ranked the results, chose
one correctly, opened it, recorded "reach:sliding puzzle -> Success", and spent
the next minute acting on a 404.

A result that turns out not to be the place is also a reason to try the next
one rather than to stop. She already did the work of deciding which results
could be it; throwing that away over one dead link wastes the decision.
"""

from __future__ import annotations

import pytest

from core.agency.reach_place import MOST_TRIES, _says_it_is_not_there, reach


class Browser:
    """A browser that serves whatever each URL was set up to serve."""

    _preferred_browser = "Google Chrome"

    def __init__(self, serves: dict[str, str], results: list[dict[str, str]] | None = None):
        self.serves = serves
        self.results = results or []
        self.opened: list[str] = []
        self._at = ""

    async def current_page(self) -> dict[str, str]:
        return {"url": self._at, "title": self.serves.get(self._at, "")}

    async def open_url(self, url: str, new_tab: bool = False) -> None:
        self.opened.append(url)
        self._at = url

    async def search_results(self, wanted: str, count: int = 5):
        return list(self.results)


async def picks_the_first(wanted, candidates, **_kw):
    return candidates[0], "it looked like the place", [row["url"] for row in candidates]


@pytest.fixture
def choosing(monkeypatch):
    monkeypatch.setattr("core.agency.reach_place._choose_destination", picks_the_first)
    monkeypatch.setattr("core.agency.reach_place._remember_arrival", lambda *a, **k: None)


# ── how a page says it is not there ──────────────────────────────────────

@pytest.mark.parametrize(
    "title",
    ["The page could not be found", "404 Not Found", "Error 404", "Page unavailable",
     "This page does not exist", "No longer available"],
)
def test_a_page_that_announces_it_is_missing_is_recognised(title):
    assert _says_it_is_not_there(title) is True


@pytest.mark.parametrize(
    "title",
    ["Sliding Puzzle — Play Online", "Lost and Found Office", "404 Cafe & Bar",
     "Notfound Records", "", "   "],
)
def test_and_a_page_that_is_merely_about_missing_things_is_not(title):
    assert _says_it_is_not_there(title) is False


# ── what reach does with one ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_not_found_page_is_not_arrival(choosing):
    browser = Browser(
        serves={"https://games.example/sliding": "The page could not be found"},
        results=[{"url": "https://games.example/sliding", "title": "Sliding Puzzle"}],
    )
    got = await reach("sliding puzzle", browser=browser)
    assert got.arrived is False
    assert "could not be found" in got.reason


@pytest.mark.asyncio
async def test_and_she_tries_the_next_result_she_ranked(choosing):
    browser = Browser(
        serves={
            "https://games.example/sliding": "404 Not Found",
            "https://other.example/puzzle": "Sliding Puzzle — Play Online",
        },
        results=[
            {"url": "https://games.example/sliding", "title": "Sliding Puzzle"},
            {"url": "https://other.example/puzzle", "title": "Puzzle"},
        ],
    )
    got = await reach("sliding puzzle", browser=browser)
    assert got.arrived is True
    assert got.url == "https://other.example/puzzle"
    assert browser.opened == ["https://games.example/sliding", "https://other.example/puzzle"]


@pytest.mark.asyncio
async def test_and_says_that_the_first_one_did_not_exist(choosing):
    browser = Browser(
        serves={"https://a.example/x": "404", "https://b.example/y": "Sliding Puzzle"},
        results=[{"url": "https://a.example/x", "title": "X"},
                 {"url": "https://b.example/y", "title": "Y"}],
    )
    got = await reach("sliding puzzle", browser=browser)
    assert "did not exist" in got.because


@pytest.mark.asyncio
async def test_she_does_not_walk_a_whole_search_end_to_end(choosing):
    browser = Browser(
        serves={f"https://s{n}.example/p": "Not Found" for n in range(8)},
        results=[{"url": f"https://s{n}.example/p", "title": f"P{n}"} for n in range(8)],
    )
    got = await reach("sliding puzzle", browser=browser)
    assert got.arrived is False
    assert len(browser.opened) == MOST_TRIES


@pytest.mark.asyncio
async def test_a_page_that_is_really_there_is_reached_first_time(choosing):
    browser = Browser(
        serves={"https://games.example/sliding": "Sliding Puzzle — Play Online"},
        results=[{"url": "https://games.example/sliding", "title": "Sliding Puzzle"}],
    )
    got = await reach("sliding puzzle", browser=browser)
    assert got.arrived is True
    assert browser.opened == ["https://games.example/sliding"]
    assert "did not exist" not in (got.because or "")


@pytest.mark.asyncio
async def test_a_url_given_outright_is_still_checked(choosing):
    browser = Browser(serves={"https://games.example/sliding": "The page could not be found"})
    got = await reach("https://games.example/sliding", browser=browser)
    assert got.arrived is False


def test_a_title_that_is_only_a_status_number_is_still_a_missing_page():
    assert _says_it_is_not_there("404") is True
    assert _says_it_is_not_there(" 404 ") is True
