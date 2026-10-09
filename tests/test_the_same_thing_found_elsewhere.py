"""A thing that cannot be had where a list points is found elsewhere, as a person finds it: searched for, and each page looked at.

No site is known: a lookalike with another name, a refusal, and a page about the
thing where nothing runs are each passed over; the page where it is and runs is kept.
"""
from __future__ import annotations

import pytest

PAGES = {
    "lookalike": ("Tunnel Rush - play free online", "<h1>Tunnel Rush</h1><canvas width=600 height=400></canvas>"),
    "refusal": ("Attention Required! | Cloudflare", "<h1>Sorry, you have been blocked</h1>"),
    "about-it": ("Toonami Tunnel Rush - a review", "<h1>Toonami Tunnel Rush</h1><p>We remember it fondly.</p>"),
    "the-thing": ("Toonami Tunnel Rush : Toonami", "<h1>Toonami Tunnel Rush</h1><canvas width=560 height=384></canvas>"),
}


def _serves(title: str, body: str):
    async def _fulfil(route) -> None:
        await route.fulfill(status=200, content_type="text/html", body=f"<html><head><title>{title}</title></head><body>{body}</body></html>")

    return _fulfil


@pytest.fixture
async def browser():
    from core.capabilities.phantom_browser import PhantomBrowser

    instance = PhantomBrowser(visible=False, browser_type="chromium", principal="owner")
    if not await instance.ensure_ready():
        await instance.close()
        pytest.skip("no browser engine available in this environment")
    try:
        for name, (title, body) in PAGES.items():
            await instance.context.route(f"**/aura-{name}", _serves(title, body))
        yield instance
    finally:
        await instance.close()


class _Skill:
    def __init__(self, browser) -> None:
        self.browser = browser

    async def _safe_browse(self, browser, url):
        return await browser.browse(url, principal="owner")


@pytest.mark.asyncio
async def test_the_page_where_it_is_and_runs_is_found_past_lookalikes_refusals_and_pages_about_it(browser, monkeypatch):
    from core.skills import where_things_are_kept
    from core.skills.sovereign_browser_going import the_same_thing_elsewhere

    hits = [where_things_are_kept.Candidate(PAGES[n][0], f"https://example.com/aura-{n}", "a search") for n in ("lookalike", "refusal", "about-it", "the-thing")]

    async def looked_for(skill, browser, name, *, task=""):
        assert where_things_are_kept.kind_of_thing(task) == "game"  # what kind of thing it is is looked for too
        return hits

    monkeypatch.setattr(where_things_are_kept, "where_it_might_be", looked_for)
    found = await the_same_thing_elsewhere(_Skill(browser), browser, "Toonami: Tunnel Rush", task="Play this game and win it.")
    assert found == "https://example.com/aura-the-thing"


@pytest.mark.asyncio
async def test_nothing_found_is_said_as_nothing(browser, monkeypatch):
    from core.skills import where_things_are_kept
    from core.skills.sovereign_browser_going import the_same_thing_elsewhere

    async def looked_for(skill, browser, name, *, task=""):
        return [where_things_are_kept.Candidate(PAGES["lookalike"][0], "https://example.com/aura-lookalike", "a search")]

    monkeypatch.setattr(where_things_are_kept, "where_it_might_be", looked_for)
    assert await the_same_thing_elsewhere(_Skill(browser), browser, "Toonami: Tunnel Rush", task="Play it.") == ""


@pytest.mark.unit
def test_the_kind_of_thing_is_read_from_the_task_and_chooses_the_catalogues():
    from core.skills.where_things_are_kept import CATALOGUES, kind_of_thing

    assert kind_of_thing("Play this game and win it.") == "game"
    assert kind_of_thing("Read this book and sum it up.") == "book"
    asked_of = lambda kind: [c.name for c in CATALOGUES if kind in c.keeps]  # noqa: E731
    assert asked_of("game") == ["the Internet Archive"] and asked_of("book") == ["the Internet Archive", "Open Library"]
    assert asked_of("photo") == ["Wikimedia Commons"]


@pytest.mark.unit
def test_its_own_name_with_a_word_of_its_series_names_it():
    # LIVE 2026-10-07 "Scooby-Doo: Scooby Snapshot" is kept as "Scooby Snapshot", and was taken for another thing.
    from core.skills.sovereign_browser_going import _names_it

    assert _names_it("Scooby-Doo: Scooby Snapshot", "Scooby Snapshot")
    assert not _names_it("Toonami: Tunnel Rush", PAGES["lookalike"][0])
    assert not _names_it("Scooby-Doo: Scooby Snapshot", "Scooby-Doo Mystery")


def test_a_page_titled_with_its_own_name_whose_words_name_the_series_is_it():
    # LIVE 2026-10-09 "Scooby-Doo: Ask Swami Shaggy" is kept as "Ask Swami Shaggy", topic Scooby-Doo, and was passed over.
    from core.skills.sovereign_browser_going import _named_on_its_page, _own_name_in

    assert _own_name_in("Scooby-Doo: Ask Swami Shaggy", "Ask Swami Shaggy")
    assert _named_on_its_page("Scooby-Doo: Ask Swami Shaggy", "Ask Swami Shaggy",
                              "Ask Swami Shaggy by Cartoon Network Topics Flash, Scooby-Doo, Cartoon Network, game")
    assert not _named_on_its_page("Scooby-Doo: Ask Swami Shaggy", "Ask Swami Shaggy", "Ask Swami Shaggy, a fortune teller game")
    title, body = PAGES["lookalike"]
    assert not _named_on_its_page("Toonami: Tunnel Rush", title, body)
