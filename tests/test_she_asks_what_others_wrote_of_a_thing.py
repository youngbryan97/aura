"""Where a thing's instructions cannot be found, she reads what others wrote of it: the record where it is kept, its
fans' wiki, and players' questions.

LIVE 2026-10-10 her searches for a game found a talking-cat app and Windows 11; its fans' wiki said how it is won, and
that waking the dog loses it.
"""
from __future__ import annotations

import asyncio

import pytest

from core.cognition.taking_stock import BEFORE, Situation, from_what_others_wrote, take_stock
from core.skills.sovereign_browser_going import _named_on_its_page, _names_it
from core.skills.what_others_wrote import FANS, KEPT, worlds_it_belongs_to

pytestmark = pytest.mark.unit

THING = "Tom's Trap-o-Matic"
WIKI = ("Tom's Trap-O-Matic is an online flash game based on the Tom and Jerry series. The goal is to let Tom catch "
        "Jerry by selecting various traps and placing on the corresponding flooring. Setting the traps at the wrong "
        "location or waking Spike up from his sleep results in a loss. Setting the traps at the right location without "
        "waking Spike up will cause the cage to trap Jerry, resulting in a win.")


def test_where_its_fans_keep_a_wiki_comes_from_what_it_is_part_of():
    assert worlds_it_belongs_to(THING, said_of_it=["Tom and Jerry", "Flash", "Cartoon Network Studios"])[:3] == \
        ["tom", "tomandjerry", "cartoonnetworkstudios"]
    assert worlds_it_belongs_to("Scooby-Doo: Scooby Snapshot")[0] == "scoobydoo"


def test_a_page_titled_by_the_things_own_name_and_naming_whose_it_is_is_of_it():
    assert _named_on_its_page(THING, "Trap-o-Matic Game at Gameshot.org", "Tom and Jerry are still not able to help it")
    assert not _named_on_its_page(THING, "Trap-o-Matic Game at Gameshot.org", "A mousetrap kit for your kitchen")
    assert not _names_it(THING, "My Talking Tom - Apps on Google Play")


def test_what_its_fans_wrote_is_kept_as_how_it_is_won_and_lost():
    async def read(_seconds):
        return [(KEPT, "a review of Tom's Trap-o-Matic", "https://archive.org/details/x", "Thanks. It's finally here."),
                (FANS, "Tom's Trap-O-Matic", "https://tomandjerry.fandom.com/wiki/Tom%27s_Trap-O-Matic", WIKI)]

    counsel = asyncio.run(take_stock(Situation(THING, f"play {THING}", (), "", BEFORE),
                                     {"what others wrote": from_what_others_wrote(read)}, seconds=5))
    kept = [h.text for h in counsel.kept]
    assert any("waking Spike up from his sleep results in a loss" in t for t in kept)
    assert any("resulting in a win" in t for t in kept)
    assert all(h.source == FANS and "tomandjerry.fandom.com" in h.where for h in counsel.kept)
    assert "what its fans wrote" in counsel.gathered()
