"""A page about a thing that does not run it is asked where the thing is kept, and its words are read for what it shows.

LIVE 2026-10-08: a game's name with a possessive ("Eddy’s") found nothing in a
catalogue that keeps it as "Eddy"; a museum page's own "in Internet Archive"
link was to a 2020 copy of the game's page, and was unwrapped to the live page
that refuses browsers; and a player whose panel said "Something went wrong" was
judged to draw its screen, its words lost behind its stylesheet.
"""
from __future__ import annotations

import pytest

from core.skills.sovereign_browser_going import _unwrapped
from core.skills.where_things_are_kept import _words
from core.skills.whether_a_page_serves import _EMBEDS


@pytest.mark.unit
def test_a_possessive_is_its_word():
    assert _words("Ed, Edd n Eddy’s Candy Machine Deluxe") == ["ed", "edd", "eddy", "candy", "machine", "deluxe"]


@pytest.mark.unit
def test_only_the_archives_own_rewriting_of_a_link_is_undone():
    here = "https://web.archive.org/web/20260716134550/https://museum.example/games/x"
    rewritten = "https://web.archive.org/web/20260716134550/https://archive.org/details/x-game"
    pointed = "https://web.archive.org/web/20200812212251/http://cartoons.example/games/x/index.html"
    assert _unwrapped(rewritten, here) == "https://archive.org/details/x-game"
    assert _unwrapped(pointed, here) == pointed
    assert _unwrapped("https://archive.org/details/x-game", here) == "https://archive.org/details/x-game"


@pytest.mark.unit
def test_a_players_words_are_read_without_its_stylesheet():
    assert "shown(el.shadowRoot || el)" in _EMBEDS and "STYLE" in _EMBEDS
