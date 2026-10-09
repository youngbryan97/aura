"""Moves made on two places: one thing used on another, two alike turned together, a thing carried to a place.

Each is named by its act and its two places ('use "A" on "B"', 'match "A" with
"B"', 'drag "A" to "B"'), and each is done on the two places the screen reading
showed: a use or a match as two clicks, a carry as a press carried from the
one to the other. What offers, says and does such a move reads its name here,
so a new act on two places is added once.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["CARRY", "MATCH", "USE", "TwoPlaces", "two_places_of"]

USE, MATCH, CARRY = "use", "match", "carry"


@dataclass(frozen=True)
class TwoPlaces:
    """A move on two places: what it does, and the two places by what she calls them."""

    act: str
    one: str
    other: str

    @property
    def by_clicks(self) -> bool:
        """Whether it is made as two clicks, the one and then the other; else as one press carried between them."""
        return self.act != CARRY

    def said(self) -> str:
        """The move as she says it while making it."""
        if self.act == USE:
            return f'Using "{self.one}" on "{self.other}"'
        if self.act == MATCH:
            return f'Matching "{self.one}" with "{self.other}"'
        return f'Carrying "{self.one}" to "{self.other}"'


def two_places_of(move: str) -> TwoPlaces | None:
    """The act and the two places of a move on two places, or None when it is not one."""
    from core.agency.putting_things_in_place import what_is_carried
    from core.agency.taking_and_using import what_is_used
    from core.agency.things_that_go_together import what_is_matched

    for act, read in ((USE, what_is_used), (MATCH, what_is_matched), (CARRY, what_is_carried)):
        found = read(move)
        if found is not None:
            return TwoPlaces(act, found[0], found[1])
    return None
