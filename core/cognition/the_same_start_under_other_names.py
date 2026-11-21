"""A world she has not met whose beginning is one she has met, with its things called by other names.

Two applications can draw the same board game, each with pieces of its own. She names a kind of piece by its look, so
every name at the second board is new, and nothing she worked out at the first is any use there. A game's beginning
still says which kind stands where. Where every place holds a piece of the same side in both, and each name in one
stands on exactly the places one name in the other does, the two beginnings are one beginning under two sets of names.
The names then map one to one, and what she learned of a kind at the first board is her first guess about its fellow
at the second.

It is a guess. Two games can begin alike and go on differently, so what she borrows is held as anything she learns
there is held, and the board she is at corrects it.

Places are compared as they are given. A caller whose places can be seen another way (a board seen from its other
side) turns one beginning before it asks, and says which way it turned it: a beginning can line up two ways under two
different sets of names, and only something other than the beginning says which way is true.
"""
from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any

__all__ = ["the_same_start"]

#: What stands on a place at the beginning: whose it is, and its name, None where its name was not read.
Standing = tuple[Any, Hashable | None]


def the_same_start(new: Mapping[Hashable, Standing], kept: Mapping[Hashable, Standing]) -> dict | None:
    """Each name in ``new`` as ``kept`` calls it, where the two beginnings are one beginning under two sets of names;
    None where they are not. A place whose name was not read lines up with any piece of the same side, and says
    nothing of names."""
    if not new or len(new) != len(kept):
        return None
    names: dict = {}
    taken: dict = {}
    for place, (side, name) in new.items():
        there = kept.get(place)
        if there is None or there[0] != side:
            return None
        if name is None or there[1] is None:
            continue
        if names.setdefault(name, there[1]) != there[1] or taken.setdefault(there[1], name) != name:
            return None
    return names or None
