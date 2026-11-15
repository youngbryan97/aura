"""A world as its code makes it: how her body is moved and kept in, the ground to be gone over, what each kind of
thing does to her, the clock and the lives, in the stage's own pixels and seconds.

A person handed a game's source and asked how it plays reads it for its physics before anything else: the hero runs
150 pixels a second while an arrow is held and is kept between the top bar and the floor; the floor is cut into
20-pixel squares from y = 40, and a square is painted when the middle of the hero is over it; 85% of them painted wins.
Played without that, every one of those numbers has to be found by trying and watching, and some of them (where the
squares begin, which part of her paints) are not shown by anything she can see until she has gone wrong over them.

This is what such a reading comes to, as data: what a reader of the code finds
(core/perception/reading_a_world_s_code.py for a page's scripts; core/perception/how_a_program_plays.py for a Flash
program) and what her play takes from it (core/agency/ground_her_code_names.py). Nothing here is run from the program,
and nothing names a game.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from core.verify import invariant

__all__ = ["AKindOfThing", "AWorldAsItsCodeMakesIt", "GroundToCover", "HerBody", "from_how_it_plays", "kept", "unkept"]


@dataclass(frozen=True)
class GroundToCover:
    """Ground cut into a grid, a cell of it marked once a point of her body is over it, and won by how many are.

    ``cell`` is a cell's width and height and ``origin`` where cell (0, 0) begins, both in stage pixels; ``cells`` is
    how many columns and rows there are; ``marks_at`` is the point of her body whose cell is marked, from her body's
    top-left corner; ``wins_at`` is the share of cells marked that wins, where the code says."""

    cell: tuple[float, float]
    origin: tuple[float, float] = (0.0, 0.0)
    cells: tuple[int, int] | None = None
    marks_at: tuple[float, float] | None = None
    wins_at: float | None = None
    #: What the code keeps the marked cells in, and counts them by: where a check against the running world looks.
    kept_in: str = ""
    counted_by: str = ""
    #: The colour a marked cell is drawn in: what is gone over, to her eyes.
    drawn_in: str = ""

    def cell_of(self, x: float, y: float) -> tuple[int, int] | None:
        """The cell a stage point is in; None outside the grid."""
        column = math.floor((x - self.origin[0]) / self.cell[0])
        row = math.floor((y - self.origin[1]) / self.cell[1])
        if column < 0 or row < 0:
            return None
        if self.cells is not None and (column >= self.cells[0] or row >= self.cells[1]):
            return None
        return column, row

    def middle_of(self, column: int, row: int) -> tuple[float, float]:
        return (self.origin[0] + (column + 0.5) * self.cell[0], self.origin[1] + (row + 0.5) * self.cell[1])

    def every_cell(self) -> list[tuple[int, int]]:
        if self.cells is None:
            return []
        return [(c, r) for r in range(self.cells[1]) for c in range(self.cells[0])]

    @property
    def count(self) -> int | None:
        return self.cells[0] * self.cells[1] if self.cells is not None else None


@dataclass(frozen=True)
class HerBody:
    """Her body as the code moves it.

    ``paces`` is how fast each key held moves it, in stage pixels a second (a key moving it along both axes, or two
    held together, add); ``kept_inside`` the rectangle its top-left corner is held to, (left, top, right, bottom);
    ``size`` its width and height; ``start`` where its top-left corner begins. ``pushes`` is, for a body carried on by
    its own going, how much each key adds to its going each second, with "" for the pull on it when no key is held;
    ``lifts`` how fast a key sends it up while it stands, with ``pull`` the fall it is under once off the ground."""

    name: str = ""
    size: tuple[float, float] | None = None
    start: tuple[float, float] | None = None
    paces: dict[str, tuple[float, float]] = field(default_factory=dict)
    kept_inside: tuple[float, float, float, float] | None = None
    pushes: dict[str, tuple[float, float]] = field(default_factory=dict)
    lifts: dict[str, float] = field(default_factory=dict)
    pull: float | None = None
    #: Where it goes the way it faces: how fast each key turns it (radians a second), how fast each gathers its speed
    #: (pixels a second, each second), and the speeds it is held between (backward, forward).
    turns: dict[str, float] = field(default_factory=dict)
    drives: dict[str, float] = field(default_factory=dict)
    speeds: tuple[float, float] | None = None
    #: Whether it is put where the pointer is, across and down.
    follows_pointer: tuple[bool, bool] = (False, False)
    #: The colour it is drawn in ("#f2f2f2"), where its code fills it with one.
    drawn_in: str = ""
    #: Whether anything inside where she is kept stops her (a wall, a box she walks into): where not, nothing she meets
    #: there is in her way, whatever play has made of it.
    stopped_inside: bool = False

    def kept_middle(self) -> tuple[float, float, float, float] | None:
        """Where the middle of her body can be: the rectangle its top-left corner is held to, moved by half her size."""
        if self.kept_inside is None or self.size is None:
            return None
        left, top, right, bottom = self.kept_inside
        w, h = self.size
        return left + w / 2, top + h / 2, right + w / 2, bottom + h / 2


@dataclass(frozen=True)
class AKindOfThing:
    """A kind of thing the code makes, by the name it is kept under: what touching her does ("costs a life", "scores",
    "nothing"), how it moves ("falls", "wanders", "chases her", "bounces", "stands"), how fast at most in stage pixels
    a second, and its size."""

    name: str
    touching: str = ""
    moves: str = ""
    speed: float | None = None
    size: tuple[float, float] | None = None
    #: Each look it is drawn with, and what touching her does when it looks so: ("#f2c230", "scores"); the second is ""
    #: where the code does not tell the looks apart in what touching does.
    looks: tuple[tuple[str, str], ...] = ()
    #: What its getting past an edge of the stage does, where the code does it in the same breath as touching her ("an
    #: invader that lands or touches her costs a life"): then touching her costs nothing that letting it by would not.
    past_an_edge: str = ""


@dataclass
class AWorldAsItsCodeMakesIt:
    """What a world's code says of how it plays, in its stage's pixels and in seconds."""

    stage: tuple[float, float] = (0.0, 0.0)
    hers: HerBody | None = None
    ground: GroundToCover | None = None
    things: list[AKindOfThing] = field(default_factory=list)
    #: How long a game lasts where a clock ends it, in seconds; and how many lives it gives.
    clock_s: float | None = None
    lives: int | None = None
    #: Each key it listens for, and what it does: "moves", "jumps", "acts", "pauses", ...
    keys: dict[str, str] = field(default_factory=dict)
    #: Each key under which it makes a thing of a kind it keeps, and that kind's name: space sends "shots".
    sends: dict[str, str] = field(default_factory=dict)
    #: Where in the code each finding was read: what a check against the running world names when it fails.
    read_at: dict[str, str] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.hers or self.ground or self.things or self.keys or self.clock_s)

    def thing(self, name: str) -> AKindOfThing | None:
        return next((t for t in self.things if t.name == name), None)


#: The ways the arrows move a body, where a program's code says only that they move it.
_ARROWS = {"left": (-1.0, 0.0), "right": (1.0, 0.0), "up": (0.0, -1.0), "down": (0.0, 1.0)}


def from_how_it_plays(plays: Any, frames_a_second: float | None = None) -> AWorldAsItsCodeMakesIt | None:
    """What a Flash program's reading (core/perception/how_a_program_plays.py) says of its world, as a world: its stage,
    her body kept where its class keeps her, the arrows moving her at its greatest speed (a number given per frame,
    at the file's frames a second), and ground cut into the cells it marks. A Flash program places a body by its
    registration point, which the code does not show against what is drawn: where the cells are marked is left to be
    her middle (``marks_at`` None), and said so. None where it reads nothing of these."""
    if not plays:
        return None
    stage = tuple(getattr(plays, "stage", None) or (0.0, 0.0))
    covers = dict(getattr(plays, "covers", None) or {})
    hers_read = dict(getattr(plays, "hers", None) or {})
    numbers = dict(hers_read.get("numbers") or {})
    field_ = covers.get("field")
    most = next((v for k, v in numbers.items() if "max" in k.lower() and "speed" in k.lower()), None)
    paces = {}
    if most and frames_a_second:
        for key, what in getattr(plays, "keys", {}).items():
            if getattr(what, "does", "") == "moves" and key in _ARROWS:
                dx, dy = _ARROWS[key]
                paces[key] = (dx * most * frames_a_second, dy * most * frames_a_second)
    hers = None
    if hers_read or paces:
        hers = HerBody(name=str(hers_read.get("symbol") or hers_read.get("class") or ""), paces=paces,
                       kept_inside=tuple(field_) if field_ else None)
    ground = None
    cell = covers.get("cell")
    if cell:
        if field_:
            cells = (math.ceil(field_[2] / cell[0]), math.ceil(field_[3] / cell[1]))
        elif stage[0] and stage[1]:
            cells = (math.ceil(stage[0] / cell[0]), math.ceil(stage[1] / cell[1]))
        else:
            cells = None
        ground = GroundToCover(cell=tuple(cell), cells=cells, counted_by=str(covers.get("counted_by") or ""))
    world = AWorldAsItsCodeMakesIt(stage=stage, hers=hers, ground=ground,
                                   keys={k: getattr(a, "does", "") for k, a in getattr(plays, "keys", {}).items()})
    if covers.get("marked_in"):
        world.read_at["ground"] = str(covers["marked_in"])
    return world or None


def _tuple(value: Any) -> tuple | None:
    return tuple(value) if isinstance(value, list | tuple) else None


def kept(world: AWorldAsItsCodeMakesIt | None) -> dict[str, Any]:
    """A world's reading as plain data, to be kept with the program's read."""
    if not world:
        return {}
    hers = world.hers
    ground = world.ground
    return {
        "stage": list(world.stage),
        "hers": None if hers is None else {
            "name": hers.name, "size": hers.size and list(hers.size), "start": hers.start and list(hers.start),
            "paces": {k: list(v) for k, v in hers.paces.items()}, "kept_inside": hers.kept_inside and list(hers.kept_inside),
            "pushes": {k: list(v) for k, v in hers.pushes.items()}, "lifts": dict(hers.lifts), "pull": hers.pull,
            "turns": dict(hers.turns), "drives": dict(hers.drives), "speeds": hers.speeds and list(hers.speeds),
            "follows_pointer": list(hers.follows_pointer), "drawn_in": hers.drawn_in,
            "stopped_inside": hers.stopped_inside},
        "ground": None if ground is None else {
            "cell": list(ground.cell), "origin": list(ground.origin), "cells": ground.cells and list(ground.cells),
            "marks_at": ground.marks_at and list(ground.marks_at), "wins_at": ground.wins_at,
            "kept_in": ground.kept_in, "counted_by": ground.counted_by, "drawn_in": ground.drawn_in},
        "things": [{"name": t.name, "touching": t.touching, "moves": t.moves, "speed": t.speed,
                    "size": t.size and list(t.size), "looks": [list(look) for look in t.looks],
                    "past_an_edge": t.past_an_edge} for t in world.things],
        "clock_s": world.clock_s, "lives": world.lives, "keys": dict(world.keys),
        "sends": dict(world.sends), "read_at": dict(world.read_at)}


def unkept(held: Any) -> AWorldAsItsCodeMakesIt | None:
    """A kept reading, as it was read; None where nothing was kept."""
    if not isinstance(held, dict) or not held:
        return None
    try:
        h, g = held.get("hers"), held.get("ground")
        hers = None if not h else HerBody(
            name=str(h.get("name") or ""), size=_tuple(h.get("size")), start=_tuple(h.get("start")),
            paces={k: tuple(v) for k, v in (h.get("paces") or {}).items()}, kept_inside=_tuple(h.get("kept_inside")),
            pushes={k: tuple(v) for k, v in (h.get("pushes") or {}).items()}, lifts=dict(h.get("lifts") or {}),
            pull=h.get("pull"), turns=dict(h.get("turns") or {}), drives=dict(h.get("drives") or {}),
            speeds=_tuple(h.get("speeds")), follows_pointer=tuple(h.get("follows_pointer") or (False, False)),
            drawn_in=str(h.get("drawn_in") or ""), stopped_inside=bool(h.get("stopped_inside")))
        ground = None if not g else GroundToCover(
            cell=tuple(g["cell"]), origin=tuple(g.get("origin") or (0.0, 0.0)), cells=_tuple(g.get("cells")),
            marks_at=_tuple(g.get("marks_at")), wins_at=g.get("wins_at"), kept_in=str(g.get("kept_in") or ""),
            counted_by=str(g.get("counted_by") or ""), drawn_in=str(g.get("drawn_in") or ""))
        things = [AKindOfThing(str(t.get("name") or ""), str(t.get("touching") or ""), str(t.get("moves") or ""),
                               t.get("speed"), _tuple(t.get("size")),
                               tuple(tuple(look) for look in t.get("looks") or []),
                               str(t.get("past_an_edge") or "")) for t in held.get("things") or []]
        world = AWorldAsItsCodeMakesIt(stage=tuple(held.get("stage") or (0.0, 0.0)), hers=hers, ground=ground,
                                       things=things, clock_s=held.get("clock_s"), lives=held.get("lives"),
                                       keys=dict(held.get("keys") or {}), sends=dict(held.get("sends") or {}),
                                       read_at=dict(held.get("read_at") or {}))
    # not a failure: a held record that does not read as a world is none
    except (KeyError, TypeError, ValueError):
        return None
    return world or None


@invariant('perception.saved_world_readings_preserve_action_effects', scope='perception',
           owner='core/perception/a_world_as_its_code_makes_it.py', observational=False)
def _saved_world_effects_invariant() -> tuple:
    import json

    world = AWorldAsItsCodeMakesIt(stage=(120., 80.), hers=HerBody(name='body', paces={'a': (30., 0.)}),
                                   things=[AKindOfThing('markers', moves='rises', speed=100.),
                                           AKindOfThing('obstacles', past_an_edge='costs a life')],
                                   sends={'j': 'markers'}, keys={'a': 'moves', 'j': 'acts'},
                                   read_at={'body': 'update', 'sends': 'emit'})
    assert unkept(json.loads(json.dumps(kept(world)))) == world
    return ()
