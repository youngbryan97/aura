"""The record is keyed on the KIND of position, not on the reading itself.

Keyed on a truncated string of everything on the screen, no two situations
were ever alike — so forty runs of experience amounted to nothing on the
forty-first. Chase and Simon's result is that skill is stored patterns:
recognition proposes the candidate moves that search then refines.
"""

from __future__ import annotations

import pytest

from core.agency.deliberate_action import ActionOption, deliberate
from core.perception.what_is_there import arranged


def board(rows: list[list[str]]):
    return arranged([
        (0.2 + r * 0.15, 0.2 + c * 0.15, said)
        for r, row in enumerate(rows)
        for c, said in enumerate(row)
        if said
    ])


MOVES = [ActionOption(name=name) for name in ("up", "down", "left", "right")]

HERE = board([["2", "4", "", "8"], ["", "2", "4", ""], ["4", "", "", "2"], ["64", "2", "", "4"]])
#: The same kind of position — a 64 held bottom-left, the same fill, the same
#: leaders — with every small thing different.
LIKE_IT = board([["4", "2", "", "2"], ["", "8", "2", ""], ["2", "", "", "4"], ["64", "4", "", "2"]])
#: A different kind of position: the big one has left the corner.
NOT_LIKE_IT = board([["64", "4", "", "2"], ["", "8", "2", ""], ["2", "", "", "4"], ["4", "2", "", "8"]])


class _Graph:
    def __init__(self) -> None:
        self.written: list[tuple[str, str, str, bool]] = []

    def record_outcome(self, action, context, outcome, success):
        self.written.append((action, context, outcome, success))

    def query_consequences(self, _action, *_a, **_k):
        return []


async def _decide(seeing, graph):
    async def names_one(_objective, _evidence):
        return "I'll press left."

    return await deliberate(
        "get to 256", seeing.as_text(), MOVES, think=names_one,
        seeing=seeing, graph=graph, announce=False, lived=False,
    )


# ── the shape itself ─────────────────────────────────────────────────────

def test_two_positions_of_the_same_kind_look_the_same():
    assert HERE.as_shape() == LIKE_IT.as_shape()


def test_the_same_contents_in_a_different_place_is_a_different_kind():
    assert HERE.as_shape() != NOT_LIKE_IT.as_shape()


def test_the_reading_itself_is_never_the_same_twice():
    """Which is the whole reason keying on it made every position unique."""
    assert HERE.as_text() != LIKE_IT.as_text()


# ── what reaches the record ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_record_is_written_under_the_kind_of_position():
    graph = _Graph()
    made = await _decide(HERE, graph)
    assert made.shape == HERE.as_shape()


@pytest.mark.asyncio
async def test_a_reading_that_kept_no_arrangement_still_records_something():
    graph = _Graph()

    async def names_one(_objective, _evidence):
        return "I'll press left."

    made = await deliberate(
        "get to 256", "2 4 8 64", MOVES, think=names_one,
        graph=graph, announce=False, lived=False,
    )
    assert made.shape == ""


@pytest.mark.asyncio
async def test_the_kind_of_position_is_short_enough_to_be_a_key():
    graph = _Graph()
    made = await _decide(HERE, graph)
    assert 0 < len(made.shape) < 120
