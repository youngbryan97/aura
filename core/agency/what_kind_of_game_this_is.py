"""What kind of game this is, from what she has measured of it, as a shape other games can be held against.

A player who has played one paddle game knows the next one inside a minute,
and not because it is called the same thing: the bar answers to them, a ball
comes at them and turns back off them, another bar across the way is the other
side. That is a shape of relations between roles, and her analogy machinery
(core/agency/the_world_it_is_most_like.py, core/cognition/structure_mapping.py)
already matches worlds by exactly that, for boards. This gives a world that
moves the same treatment: the roles are her thing and each kind of thing; the
relations are what her fast way of playing has measured about them.

The same facts make one plain sentence for a watcher: what she is, what each
thing does, and what she is doing about it. Composed from the measurements;
nothing in it is guessed.
"""
from __future__ import annotations

import math
from typing import Any

from core.agency.what_meeting_things_does import AVOID, CLICK, MEET, SHOOT
from core.cognition.structure_mapping import Graph, Relation

__all__ = ["in_a_sentence", "shape_of_a_moving_world"]

ME = "me"


def _kind(number: int) -> str:
    return f"kind {number}"


def _the_other_side(moves: Any, mine: Any) -> list[Any]:
    tall, wide = moves.shape
    found = []
    for thing in moves.things.values():
        if thing.number == mine.number or not thing.moved:
            continue
        alike = abs(math.log(max(1.0, thing.w) / max(1.0, mine.w))) < 0.4 and abs(math.log(max(1.0, thing.h) / max(1.0, mine.h))) < 0.4
        across = (thing.x < wide / 2) != (mine.x < wide / 2) or (thing.y < tall / 2) != (mine.y < tall / 2)
        if alike and across:
            found.append(thing)
    return found


def shape_of_a_moving_world(moves: Any, hers: Any, meeting: Any, physics: Any, keys: list[str]) -> Graph:
    """The relations her measurements show between her thing and each kind of thing."""
    relations: list[Relation] = []
    mine = hers.thing(moves)
    if mine is None:
        return Graph(name="a moving world", relations=())
    if hers.follows_pointer:
        relations.append(Relation("follows the pointer", (ME,)))
    else:
        across, updown = hers.axes(keys)
        if across:
            relations.append(Relation("moves across", (ME,)))
        if updown:
            relations.append(Relation("moves up and down", (ME,)))
    other_side = {thing.kind for thing in _the_other_side(moves, mine)}
    for kind in sorted({thing.kind for thing in moves.things.values() if thing.moved and thing.number != mine.number}):
        name = _kind(kind)
        if kind in other_side:
            relations.append(Relation("across from", (name, ME)))
            continue
        _ax, ay = physics.gravity(kind)
        if ay > 0:
            relations.append(Relation("falls", (name,)))
        edges = physics.kinds[kind].edges if kind in physics.kinds else {}
        if any(edge.bounces >= 2 for edge in edges.values()):
            relations.append(Relation("bounces", (name,)))
        if any(edge.wraps >= 1 for edge in edges.values()):
            relations.append(Relation("comes round", (name,)))
        if kind in physics.kinds and len(physics.kinds[kind].meetings) >= 2:
            relations.append(Relation("turns back off", (name, ME)))
        stance = meeting.stance(kind)
        if meeting.known(kind) or kind in meeting.told:
            relations.append(Relation({MEET: "is got by", AVOID: "costs", SHOOT: "is shot by", CLICK: "is clicked by"}.get(stance, "is left by"), (name, ME)))
    for made in hers.makes.values():
        relations.append(Relation("fired by", (_kind(made.kind), ME)))
    return Graph(name="a moving world", relations=tuple(relations))


def in_a_sentence(moves: Any, hers: Any, meeting: Any, physics: Any, keys: list[str]) -> str:
    """What kind of game this is, said plainly, from the same measurements."""
    from core.agency.playing_as_it_happens import describe, where_on_screen

    mine = hers.thing(moves)
    if mine is None:
        return ""
    parts = [f"I'm the {describe(moves, mine.kind, mine)} at the {where_on_screen(moves, mine.x, mine.y)}"]
    if hers.follows_pointer:
        parts[0] += ", and I go where the mouse goes"
    else:
        moving = sorted(hers.keys_that_move_her(keys))
        if moving:
            parts[0] += f", moved by {' and '.join(moving)}"
    other_side = _the_other_side(moves, mine)
    if other_side:
        parts.append(f"the {describe(moves, other_side[0].kind, other_side[0])} across from me is the other side")
    for kind in sorted({thing.kind for thing in moves.things.values() if thing.moved and thing.number != mine.number}):
        if any(thing.kind == kind for thing in other_side):
            continue
        name = describe(moves, kind)
        bits = []
        if kind in physics.kinds and physics.gravity(kind)[1] > 0:
            bits.append("falls")
        if kind in physics.kinds and any(e.bounces >= 2 for e in physics.kinds[kind].edges.values()):
            bits.append("bounces off the walls")
        if kind in physics.kinds and len(physics.kinds[kind].meetings) >= 2:
            bits.append("turns back off me")
        stance = meeting.stance(kind)
        if meeting.known(kind) or kind in meeting.told:
            bits.append({MEET: "is worth getting", AVOID: "costs me", SHOOT: "is worth shooting", CLICK: "is worth clicking"}.get(stance, ""))
        bits = [bit for bit in bits if bit]
        if bits:
            parts.append(f"the {name} " + ", ".join(bits[:-1]) + (" and " if len(bits) > 1 else "") + bits[-1])
    return "; ".join(parts) + "."
