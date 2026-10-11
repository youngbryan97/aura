"""What her fast way of playing keeps of one game, for the next time she plays it.

Lose a game and play it again, and the second run should not start from
nothing: which kinds of thing are to be met and which kept clear of, how
things move here and what the walls do, which part of her paddle wins points.
Which thing is hers and what her keys do are not taken on trust from last
time: a few seconds of trying them settles it again. All of that is kept
under the game's own name and given back only to the same game. Another game gets none of it unless
it is the same game, because what is true of one world's bombs and walls is
not true of another's.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

__all__ = ["kept_from", "to_keep"]


# Indexed visual kinds and every record that refers to their positions.
INDEXED_TABLES = ({
    "table": ["kinds"],
    "keyed_by": [["evidence"], ["told"], ["physics"]],
    "references": [["hers", "kind"]],
    "histories": [
        ["physics", "*", "accelerations"], ["physics", "*", "speeds"],
        ["physics", "*", "meetings"], ["physics", "*", "edges", "*", "3"],
        ["physics", "*", "edges", "*", "4"],
    ],
    "structures": [
        {"path": ["kinds", "*"], "type": "mapping", "required_fields": ["colour", "size"]},
        {"path": ["kinds", "*", "colour"], "length": 3, "required": True},
        {"path": ["evidence", "*"], "length": 9},
        {"path": ["physics", "*"], "type": "mapping"},
        {"path": ["physics", "*", "accelerations"], "type": "list"},
        {"path": ["physics", "*", "accelerations", "*"], "length": 2},
        {"path": ["physics", "*", "speeds"], "type": "list"},
        {"path": ["physics", "*", "edges"], "type": "mapping"},
        {"path": ["physics", "*", "edges", "*"], "length": 5},
        {"path": ["physics", "*", "edges", "*", "3"], "type": "list", "required": True},
        {"path": ["physics", "*", "edges", "*", "4"], "type": "list", "required": True},
        {"path": ["physics", "*", "meetings"], "type": "list"},
        {"path": ["physics", "*", "meetings", "*"], "length": 3},
    ],
},)


def to_keep(keep: dict[str, Any]) -> dict[str, Any]:
    """The plain-data form of what one stretch of play left in ``keep``."""
    held: dict[str, Any] = {}
    held["_indexed_tables"] = list(INDEXED_TABLES)
    kinds = keep.get("kinds") or []
    # Colour and size, which is what a kind is matched by. Its colour mix is
    # a fixed-length vector that a long record is cut down by halving its
    # longest lists, and halved it stopped play (offline 2026-10-04); seen
    # again, it is rebuilt in a second.
    held["kinds"] = [{"colour": list(kind.colour), "size": round(float(kind.size), 2)} for kind in kinds]
    hers = keep.get("hers")
    if hers is not None and hers.kind is not None:
        held["hers"] = {
            "kind": hers.kind,
            "ways": {key: list(values[-30:]) for key, values in hers._hers.by_key.items()},
            "follows_pointer": hers.follows_pointer,
            "along": list(hers.follows_along),
        }
    meeting = keep.get("meeting")
    if meeting is not None:
        held["evidence"] = {
            str(kind): [
                round(e.touch_sum, 3), round(e.shoot, 3), e.touched, e.passed, e.shot,
                round(e.pass_sum, 3), e.touches_settled, e.passes_settled, e.immediate,
            ]
            for kind, e in meeting.evidence.items()
        }
        held["told"] = {str(kind): stance for kind, stance in meeting.told.items()}
    physics = keep.get("physics")
    if physics is not None:
        held["physics"] = {
            str(number): {
                "accelerations": [list(a) for a in list(kind.accelerations)[-80:]],
                "speeds": list(kind.speeds)[-80:],
                "edges": {
                    name: [edge.bounces, edge.leaves, edge.wraps, edge.where[-20:], edge.kept[-20:]]
                    for name, edge in kind.edges.items()
                },
                "meetings": [list(m) for m in kind.meetings[-40:]],
            }
            for number, kind in physics.kinds.items()
        }
    if keep.get("meeting_with"):
        held["meeting_with"] = {str(part): counts for part, counts in keep["meeting_with"].items()}
    shots = keep.get("by_shots") or {}
    if shots.get("sends_from"):
        # Where a press sends something from, and the setting that last got there, by each way of letting go.
        measures = {way: [found.across, found.down, found.held_s]
                    for way, tried in (shots.get("shots") or {}).items() if (found := tried.the_measure) is not None}
        held["by_shots"] = {"sends_from": list(shots["sends_from"]), "measures": measures}
    return held


def _shots_kept_from(held: dict[str, Any]) -> dict[str, Any]:
    """Where a press sent something from, and the measure each way of letting go had, as shots she can start from:
    the measure is tried first, and kept while it still gets there (core/agency/how_hard_and_which_way.py)."""
    from core.agency.how_hard_and_which_way import Setting, Shot, Shots

    shots = {}
    for way, (across, down, held_s) in (held.get("measures") or {}).items():
        tried = Shots(way=way)
        tried.took(Shot(Setting(float(across), float(down), float(held_s)), ended_at=None, gained=1))
        shots[way] = tried
    return {"sends_from": tuple(held["sends_from"]), "shots": shots}


def kept_from(held: dict[str, Any]) -> dict[str, Any]:
    """``keep`` rebuilt from what was kept, for the first stretch of a game played before."""
    from core.agency.what_meeting_things_does import WhatMeetingDoes, _Evidence
    from core.perception.how_things_move_here import HowThingsMoveHere, _Edge
    from core.perception.what_moves_in_the_picture import LOOK_BINS, Kind
    from core.runtime.what_she_learned import validate_indexed_state

    keep: dict[str, Any] = {}
    if not isinstance(held, dict):
        return keep
    held = validate_indexed_state(held, indexed_tables=INDEXED_TABLES)
    if isinstance(held, dict) and isinstance(held.get("by_shots"), dict) and held["by_shots"].get("sends_from"):
        keep["by_shots"] = _shots_kept_from(held["by_shots"])
    if not isinstance(held, dict) or not held.get("kinds"):
        return keep
    keep["kinds"] = [
        Kind(number, np.full(LOOK_BINS, 1.0 / LOOK_BINS), float(kind["size"]), tuple(int(c) for c in kind["colour"]))
        for number, kind in enumerate(held["kinds"])
    ]
    # Which thing is hers, what her keys do, and whether the mouse moves her
    # are found again by trying them, each session. Kept, they were never
    # tried again: LIVE 2026-10-04 a game played while its window was frozen
    # left "the up key does nothing" behind, and in the next session,
    # believing she knew her keys, she never pressed up and lost every game;
    # and a kept kind sent her first look to the wrong thing.
    meeting = WhatMeetingDoes()
    for kind, values in (held.get("evidence") or {}).items():
        meeting.evidence[int(kind)] = _Evidence(*values)
    meeting.told.update({int(kind): stance for kind, stance in (held.get("told") or {}).items()})
    keep["meeting"] = meeting
    physics = HowThingsMoveHere()
    for number, kind_held in (held.get("physics") or {}).items():
        kind = physics.kinds[int(number)]
        kind.accelerations.extend(tuple(a) for a in kind_held.get("accelerations") or [])
        kind.speeds.extend(kind_held.get("speeds") or [])
        kind.edges = defaultdict(_Edge, {
            name: _Edge(values[0], values[1], values[2], list(values[3]), list(values[4]))
            for name, values in (kind_held.get("edges") or {}).items()
        })
        kind.meetings = [tuple(m) for m in kind_held.get("meetings") or []]
    keep["physics"] = physics
    if held.get("meeting_with"):
        keep["meeting_with"] = {float(part): list(counts) for part, counts in held["meeting_with"].items()}
    return keep
