"""The whole loop, against a world that really does move, end to end.

Not the reader and not the rule: the two of them driven by the loop that uses
them, on a laid-out thing that slides and combines when it is pushed, read
with the drift a real reading has and the gaps a real reading leaves.

This is the measurement that was missing. Every part of the chain had tests
and the chain did not: LIVE 2026-08-31, fifty-four moves of the real game with
one of them watched, because the frame was worked out afresh from each glance
and two glances of one board were never in the same frame. Every piece passed
its own tests throughout.

What it measures today, and what it does not. Moves now reach the learner at
all, and the frame she holds stops changing under her — both were nought
before, and both are what this guards. What it cannot yet assert is that MOST
of her moves teach her something: driven through her own reasoning, three of
twenty-nine reached the learner, and the frame she settles on still has the
score and the New Game button in it, so the rule that is right about the board
scores nothing against them. That is the next thing, and it is written here
rather than asserted, because a test that claims a bar this does not clear
would be the fourth way this chain has reported itself working.

A run driven by a `policy` rather than by her own reasoning learns nothing at
all: what reaches the learner is gated on the choice object that only the
deliberating path produces. Recorded here because it is the same defect class
— a mechanism that cannot fire — found by the same harness.
"""

from __future__ import annotations

import asyncio
import itertools
import random

import pytest

import core.skills.screen_pursuit as sp

SIDE = 4


def _slide(row: list[int]) -> list[int]:
    """Everything moves as far as it can and combines on the way."""
    kept = [value for value in row if value]
    out: list[int] = []
    index = 0
    while index < len(kept):
        if index + 1 < len(kept) and kept[index] == kept[index + 1]:
            out.append(kept[index] * 2)
            index += 2
        else:
            out.append(kept[index])
            index += 1
    return out + [0] * (SIDE - len(out))


def _pushed(grid: list[list[int]], way: str) -> list[list[int]]:
    rows = [list(row) for row in grid]
    if way in ("up", "down"):
        rows = [list(column) for column in zip(*rows, strict=False)]
    if way in ("right", "down"):
        rows = [row[::-1] for row in rows]
    rows = [_slide(row) for row in rows]
    if way in ("right", "down"):
        rows = [row[::-1] for row in rows]
    if way in ("up", "down"):
        rows = [list(column) for column in zip(*rows, strict=False)]
    return rows


class _World:
    """A thing that moves when it is pushed, drawn where it is drawn."""

    def __init__(self, seed: int = 5) -> None:
        self.roll = random.Random(seed)
        self.grid = [[0] * SIDE for _ in range(SIDE)]
        self._deal()
        self._deal()
        self.pressed: list[str] = []

    def _deal(self) -> None:
        empty = [(r, c) for r in range(SIDE) for c in range(SIDE) if not self.grid[r][c]]
        if empty:
            row, column = self.roll.choice(empty)
            self.grid[row][column] = 2

    def push(self, way: str) -> None:
        self.pressed.append(way)
        after = _pushed(self.grid, way)
        if after != self.grid:
            self.grid = after
            self._deal()

    def drawn(self) -> dict:
        """One reading of it, with the drift and the furniture a real one has."""
        drift = 0.004
        layout = [
            {
                "text": str(self.grid[row][column]),
                "center_x": 0.20 + column * 0.15 + self.roll.uniform(-drift, drift),
                "center_y": 0.35 + row * 0.12 + self.roll.uniform(-drift, drift),
            }
            for row in range(SIDE)
            for column in range(SIDE)
            if self.grid[row][column]
        ]
        for text, x, y in (
            ("Score", 0.62, 0.10),
            (str(sum(sum(row) for row in self.grid)), 0.72, 0.10),
            ("New Game", 0.50, 0.24),
        ):
            layout.append({"text": text, "center_x": x, "center_y": y})
        return {
            "ok": True,
            "text": " ".join(one["text"] for one in layout),
            "layout": layout,
            "bounds": [],
            "scoped_to": "TheThing",
            "read_within": "the window",
        }


@pytest.fixture
def world(monkeypatch):
    it = _World()

    async def read(app_name="", over=None):
        return it.drawn()

    async def press(key, *, expect_app=""):
        it.push(key)
        return True

    async def press_many(keys, *, expect_app=""):
        for key in keys:
            it.push(key)
        return len(list(keys))

    async def frontmost(_app=""):
        return True

    async def identity():
        return {"url": "", "title": "", "error": ""}

    monkeypatch.setattr(sp, "read_screen", read)
    monkeypatch.setattr(sp, "press", press)
    monkeypatch.setattr(sp, "press_many", press_many)
    async def to_the_front(_app=""):
        return True

    monkeypatch.setattr(sp, "_ensure_frontmost", frontmost)
    monkeypatch.setattr(sp, "current_page_identity", identity)
    monkeypatch.setattr(sp, "_bring_the_thing_back_to_the_front", to_the_front)
    return it


def _a_mind_that_pushes_each_way():
    """A mind with nothing in it, so what is measured is the machinery.

    Through her own reasoning rather than through a `policy`, because that is
    the path the runtime takes and the two do not learn alike.
    """
    ways = itertools.cycle(("left", "up", "right", "down"))

    async def think(_prompt, **_kw):
        return f"I will press {next(ways)} to push everything to one side."

    return think


def _play(**kw):
    return asyncio.run(
        sp.pursue_on_screen(
            goal="make the numbers larger",
            success_when=r"\b4096\b",
            think=_a_mind_that_pushes_each_way(),
            narrate=False,
            lived=False,
            research=False,
            target_app="TheThing",
            **kw,
        )
    )


def test_she_works_out_how_it_moves_by_playing_it(world):
    """The end of the chain: a rule, from her own moves, in a real number of them."""
    from core.perception.how_it_moves import HowItMoves

    seen: dict[str, object] = {}
    original = HowItMoves.watched

    def watching(self, before, action, after):
        seen["it"] = self
        return original(self, before, action, after)

    HowItMoves.watched = watching
    try:
        _play(max_cycles=60, max_seconds=90.0)
    finally:
        HowItMoves.watched = original

    knows = seen.get("it")
    assert knows is not None, "not one move was ever handed to the rule learner"
    assert knows.seen > 0, f"the learner was reached and given nothing: {knows.says()}"
    assert world.pressed, "she never moved"


def test_a_move_she_makes_can_teach_her_something(world):
    """The measurement that was missing: fifty-four moves, one of them watched."""
    from core.perception.how_it_moves import HowItMoves

    seen: dict[str, object] = {}
    original = HowItMoves.watched

    def watching(self, before, action, after):
        seen["it"] = self
        return original(self, before, action, after)

    HowItMoves.watched = watching
    try:
        _play(max_cycles=40, max_seconds=90.0)
    finally:
        HowItMoves.watched = original

    knows = seen.get("it")
    assert knows is not None
    assert knows.seen > 0, (
        f"she pressed {len(world.pressed)} times and learned from none of them"
    )


def test_the_thing_she_reads_keeps_its_shape(world):
    """A frame that changes between glances makes every pair incomparable."""
    from core.perception.the_lattice_she_holds import TheLatticeSheHolds

    shapes: set[tuple[int, int]] = set()
    original = TheLatticeSheHolds.fit

    def fitting(self, said):
        out = original(self, said)
        if out is not None:
            shapes.add((out.rows, out.columns))
        return out

    TheLatticeSheHolds.fit = fitting
    try:
        _play(max_cycles=40, max_seconds=90.0)
    finally:
        TheLatticeSheHolds.fit = original

    assert len(shapes) <= 2, f"the thing she was acting in changed shape: {shapes}"
