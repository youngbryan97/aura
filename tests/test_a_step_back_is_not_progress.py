"""Three defects in her judgement, each invisible in the world it was written for.

`looking_ahead` and `how_good_is_this` were built for a world where every
move is irreversible and the same place cannot be reached twice. Every one of
these is fine there and severe anywhere else, which is what makes them worth
a file: a mind that only works in the world it was tuned in is not general,
however well it works there.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from core.agency.how_good_is_this import terms
from core.agency.looking_ahead import look_ahead


@dataclass(frozen=True)
class _APlace:
    """A state in a world where a move can be undone."""

    at: int
    seen: frozenset = frozenset()

    def as_text(self) -> str:
        return str(self.at)

    def numbers(self) -> tuple[float, ...]:
        return (2.0 ** self.at,)

    def newness(self) -> float:
        return 0.0 if self.at in self.seen else 1.0

    def empty(self) -> int:
        return 2

    def places(self) -> int:
        return 4


@dataclass
class _AWorldThatCanBeUndone:
    """Steps along a line. Every move has an opposite."""

    held: int = 0
    predicted: int = 0

    def expect(self, state: _APlace, act: str) -> _APlace | None:
        if act not in {"on", "back"}:
            return None
        there = state.at + (1 if act == "on" else -1)
        if not 0 <= there <= 8:
            return None
        return _APlace(at=there, seen=state.seen)

    def confidence(self) -> float:
        return 1.0


def test_a_line_that_steps_back_is_not_a_line_that_got_somewhere():
    """The search folded back on itself and collected the same rise twice.

    From a state whose reading rises along the line, stepping back and then
    forward again reached a higher reading than stepping forward once — so
    the best plan the search could find was to pace. Measured on a sealed
    world: eighty moves between two squares, a perfectly correct model of
    every act, and nought arrivals.
    """

    knows = _AWorldThatCanBeUndone()
    here = _APlace(at=4, seen=frozenset({0, 1, 2, 3, 4}))
    ranked = look_ahead(knows, here, ["on", "back"], toward="256", budget_s=0.05)
    assert ranked
    assert ranked["on"][0] > ranked["back"][0], (
        "going back scores at least as well as going on"
    )


def test_the_search_still_works_where_nothing_can_be_undone():
    """The fix must cost nothing in the world it was written for."""

    @dataclass
    class _OneWay:
        def expect(self, state: _APlace, act: str) -> _APlace | None:
            return _APlace(at=state.at + 1) if act == "on" else None

        def confidence(self) -> float:
            return 1.0

    ranked = look_ahead(_OneWay(), _APlace(at=1), ["on"], toward="256", budget_s=0.05)
    assert ranked and ranked["on"][0] > 0


def test_newness_is_nought_for_anything_that_cannot_be_asked():
    """A situation with no notion of having been visited is not thereby
    always new, and scoring it as new would make every board a discovery."""

    class _ABoard:
        def numbers(self):
            return (4.0, 8.0)

    assert terms(_ABoard())["newness"] == 0.0


def test_newness_separates_somewhere_new_from_somewhere_seen():
    seen = _APlace(at=3, seen=frozenset({3}))
    new = _APlace(at=3, seen=frozenset())
    assert terms(new)["newness"] > terms(seen)["newness"]


def test_the_room_term_reads_what_the_state_calls_it():
    """Named `free` it was never read, and room was nought everywhere."""

    assert terms(_APlace(at=1))["room"] == pytest.approx(0.5)
