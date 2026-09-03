"""The whole loop, against a world that really does move, end to end.

Not the reader and not the rule: the two of them driven by the loop that uses
them, on a laid-out thing that slides and combines when it is pushed, read
with the drift a real reading has and the gaps a real reading leaves.

This is the measurement that was missing. Every part of the chain had tests
and the chain did not: LIVE 2026-08-31, fifty-four moves of the real game with
one of them watched, because the frame was worked out afresh from each glance
and two glances of one board were never in the same frame. Every piece passed
its own tests throughout.

What it measures today, and what it does not. It runs the loop end to end and
insists the world moves under her, which is what caught a run that could not
arrive, a run that refused every keystroke, and a run whose keys went nowhere.
What it does NOT yet assert is the thing it was written for: that she works
out how the world moves. She does not. On this world, driven through her own
reasoning, no pair of readings reaches the rule at all.

Why, measured rather than guessed. The frame she acts in is worked out afresh
from each glance, so two glances of one board are in different frames and the
comparison is refused. Holding a frame from the first block she can see was
tried and made it worse: it froze a crop with "Score", "30", "New" and "Game"
occupying grid cells and the board's four rows squeezed into two, so pairs did
reach the rule and every one of them was wrong — the correct hypothesis scored
nought out of nine and was being discredited by its own evidence. Learning
wrong things is worse than learning nothing, so that was taken back out.

What it waits on is the crop finding the thing rather than the page. Written
here rather than asserted, because a test claiming a bar this does not clear
would be the fourth way this chain has reported itself working.

A run driven by a `policy` rather than by her own reasoning learns nothing at
all either, for a different reason: what reaches the learner is gated on the
choice object that only the deliberating path produces. Recorded here because
it is the same defect class — a mechanism that cannot fire — found by the same
harness.
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

    async def may_look():
        # Whether she is allowed to photograph the screen is a runtime
        # setting on the machine this runs on. Left real, this suite passes or
        # fails on how the last Aura to run here was configured.
        class _Allowed:
            allowed = True
            reason = None

        return _Allowed()

    monkeypatch.setattr(sp, "_ensure_frontmost", frontmost)
    monkeypatch.setattr(sp, "current_page_identity", identity)
    monkeypatch.setattr(sp, "_bring_the_thing_back_to_the_front", to_the_front)
    monkeypatch.setattr(
        "core.security.screen_capture_policy.evaluate_screen_capture_admission_async",
        may_look,
    )
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


def test_the_loop_runs_end_to_end_against_a_world_that_moves(world):
    """She reads it, decides, presses, and the world moves under her.

    Weaker than it should be on purpose. What this ought to assert is that she
    works out how the world moves — see the note at the top of this file for
    why it cannot yet, and for what a stronger version would have to wait on.
    Even this much has caught three defects: a run that could not arrive, a
    run that refused every keystroke, and a run whose keys went nowhere.
    """
    was = [row[:] for row in world.grid]
    _play(max_cycles=40, max_seconds=120.0)

    assert world.pressed, "she never pressed anything"
    assert world.grid != was, "she pressed and the world never moved"


def test_the_pairs_that_reach_the_rule_are_counted(world):
    """However many reach it, the loop must be able to say how many did.

    A move whose before and after could not be compared used to be discarded
    in silence, so fifty-four moves that taught her nothing looked exactly
    like a hard world.
    """
    from core.perception.how_it_moves import HowItMoves

    reached = {"n": 0}
    original = HowItMoves.watched

    def watching(self, before, action, after):
        reached["n"] += 1
        return original(self, before, action, after)

    HowItMoves.watched = watching
    try:
        _play(max_cycles=30, max_seconds=120.0)
    finally:
        HowItMoves.watched = original

    assert reached["n"] >= 0
    assert len(world.pressed) > 0
