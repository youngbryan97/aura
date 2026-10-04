"""What her fast way of playing keeps of one game, for the next time she plays it.

Lose a game and play it again, and the second run should not start from
nothing: which kind of thing is hers, which kinds of thing are to be met and
which kept clear of, how things move here and what the walls do, which part
of her paddle wins points. What her keys do is not taken on trust from last
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


def to_keep(keep: dict[str, Any]) -> dict[str, Any]:
    """The plain-data form of what one stretch of play left in ``keep``."""
    held: dict[str, Any] = {}
    kinds = keep.get("kinds") or []
    held["kinds"] = [
        {"colour": list(kind.colour), "size": round(float(kind.size), 2), "look": [round(float(v), 4) for v in kind.look]}
        for kind in kinds
    ]
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
                round(e.pass_sum, 3), e.touches_settled, e.passes_settled,
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
    return held


def kept_from(held: dict[str, Any]) -> dict[str, Any]:
    """``keep`` rebuilt from what was kept, for the first stretch of a game played before."""
    from core.agency.what_meeting_things_does import WhatMeetingDoes, _Evidence
    from core.agency.which_one_answers_to_her import WhichIsHers
    from core.perception.how_things_move_here import HowThingsMoveHere, _Edge
    from core.perception.what_moves_in_the_picture import Kind

    keep: dict[str, Any] = {}
    if not isinstance(held, dict) or not held.get("kinds"):
        return keep
    keep["kinds"] = [
        Kind(number, np.asarray(kind["look"], dtype=float), float(kind["size"]), tuple(int(c) for c in kind["colour"]))
        for number, kind in enumerate(held["kinds"])
    ]
    hers_held = held.get("hers") or {}
    if hers_held:
        # Which kind of thing was hers is a prior; what her keys do, and
        # whether the mouse moves her, are found again by trying them. Kept,
        # they were never tried again: LIVE 2026-10-04 a game played while
        # its window was frozen left "the up key does nothing" behind, and in
        # the next session, believing she knew her keys, she never pressed up
        # to find out and lost every game.
        hers = WhichIsHers()
        hers.kind = int(hers_held["kind"])
        keep["hers"] = hers
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
