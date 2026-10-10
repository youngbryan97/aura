"""Where a thing was found doing what it was wanted for is kept, and gone to first another day when the place that
pointed to it refuses her.

LIVE 2026-10-10 a museum's pages were refused to her all evening, and each game it listed was looked for afresh, one
never found. Nothing here is any one site.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _kept_apart(tmp_path, monkeypatch):
    import core.runtime.what_she_learned as learned

    monkeypatch.setattr(learned, "_KEPT_IN", tmp_path)


def test_where_it_was_found_is_where_she_goes_first_and_an_archived_copy_is_not_kept(monkeypatch):
    import core.skills.sovereign_browser_going as going

    opened: list[str] = []

    async def first_that_serves(skill, browser, name, task, candidates, not_at):
        opened.extend(c.url for c in candidates)
        return candidates[0].url

    monkeypatch.setattr(going, "_the_first_that_serves", first_that_serves)
    browser = SimpleNamespace(page=object())
    assert asyncio.run(going.where_i_found_it_before(None, browser, "A Game: Its Name", task="play it")) == ""
    going.found_it_at("A Game: Its Name", "https://web.archive.org/web/2020/https://example.org/a-game")
    assert asyncio.run(going.where_i_found_it_before(None, browser, "A Game: Its Name", task="play it")) == ""
    going.found_it_at("A Game: Its Name", "https://example.org/details/a-game")
    found = asyncio.run(going.where_i_found_it_before(None, browser, "a game its name", task="play it"))
    assert found == "https://example.org/details/a-game" and opened == [found]
    assert asyncio.run(going.where_i_found_it_before(None, browser, "A Game: Its Name", task="play it",
                                                     not_at="https://example.org/details/a-game")) == ""
