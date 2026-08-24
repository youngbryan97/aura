"""Working out where the parts go, so nobody has to type coordinates.

A design brief says what the parts are and what they do. It must not be
asked where they sit, because a language model handed a coordinate system
will produce numbers that look like coordinates and put the battery inside
the propeller. So the arrangement is derived here, from three things the
model already knows: which parts enclose others, which parts are joined to
which, and how big each one is.

The rules are the ones an assembly actually follows. A part tagged as an
enclosure holds the parts in its own subsystem, packed along its longest
inside dimension in the order they are connected. Parts that have to be
outside — thrusters, aerials, wheels, sensors that look out — are placed on
the enclosure's surface, spread evenly around whichever axis suits them.
Everything left over is stacked clear of the rest.

Explode directions follow from the result: a part travels away from the
assembly's centre along whichever axis it is nearest the edge of, which is
what makes an exploded view read as a sequence of removals.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np

from core.engineering.geometry import Placement
from core.engineering.model import Design, Part

__all__ = ["arrange", "interference", "auto_explode"]

#: Tags that say a part holds other parts.
_ENCLOSURE_TAGS = frozenset({"enclosure", "housing", "case", "chassis", "frame", "vessel", "tank"})

#: Tags that say a part belongs on the outside where it can reach the world.
_EXTERNAL_TAGS = frozenset(
    {"motor", "thruster", "propeller", "wheel", "aerial", "antenna", "camera",
     "light", "sensor", "port", "connector", "handle", "mount", "fin", "arm"}
)

#: How much clear space to leave between packed parts, as a fraction of the
#: enclosure's inside length. Zero packing looks wrong and reads as a solid.
_PACK_GAP = 0.06


def _extent(part: Part) -> np.ndarray:
    if part.solid is None:
        return np.zeros(3)
    low, high = part.solid.mesh().bounds()
    return np.asarray(high) - np.asarray(low)


def _inside_extent(part: Part) -> np.ndarray:
    """The enclosure's usable space after subtracting its wall."""
    outer = _extent(part)
    if part.solid is None:
        return outer
    params = part.solid.parameters()
    wall = params.get("wall")
    thickness = float(wall.value) if wall is not None else 0.0
    if thickness <= 0:
        return outer * 0.82
    return np.maximum(outer - 2.0 * thickness, outer * 0.05)


def _is_enclosure(part: Part) -> bool:
    return bool(_ENCLOSURE_TAGS & set(part.tags))


def _is_external(part: Part) -> bool:
    return bool(_EXTERNAL_TAGS & set(part.tags))


def _has_placement(part: Part) -> bool:
    return part.placement.position != (0.0, 0.0, 0.0) or part.placement.rotation != (0.0, 0.0, 0.0)


def _connection_order(design: Design, parts: list[Part]) -> list[Part]:
    """Order parts by how far they are from a source in the link graph.

    A battery, then what it feeds, then what that feeds. Packing in that
    order puts things that talk to each other next to each other, which is
    what shortens the wiring in a real build.
    """
    ids = [p.id for p in parts]
    if len(ids) < 3:
        return parts
    neighbours: dict[str, set[str]] = {pid: set() for pid in ids}
    for link in design.connections:
        a = link.source.split(".")[0]
        b = link.target.split(".")[0]
        if a in neighbours and b in neighbours:
            neighbours[a].add(b)
            neighbours[b].add(a)
    degree = {pid: len(neighbours[pid]) for pid in ids}
    if not any(degree.values()):
        return parts
    start = max(ids, key=lambda pid: degree[pid])
    seen = {start}
    order = [start]
    frontier = [start]
    while frontier:
        current = frontier.pop(0)
        for nxt in sorted(neighbours[current]):
            if nxt not in seen:
                seen.add(nxt)
                order.append(nxt)
                frontier.append(nxt)
    for pid in ids:
        if pid not in seen:
            order.append(pid)
    by_id = {p.id: p for p in parts}
    return [by_id[pid] for pid in order if pid in by_id]


def _pack_inside(host: Part, contents: list[Part]) -> dict[str, Placement]:
    """Lay parts along the host's longest inside axis, centred on the others."""
    if not contents:
        return {}
    inside = _inside_extent(host)
    axis = int(np.argmax(inside))
    run = float(inside[axis])
    sizes = [float(_extent(p)[axis]) for p in contents]
    total = sum(sizes)
    gap = run * _PACK_GAP
    needed = total + gap * (len(contents) - 1)
    # Squeeze the gaps before overflowing the enclosure.
    if needed > run and len(contents) > 1:
        gap = max((run - total) / (len(contents) - 1), 0.0)
        needed = total + gap * (len(contents) - 1)
    cursor = -needed / 2.0
    host_centre = np.asarray(host.placement.position, dtype=float)
    placements: dict[str, Placement] = {}
    for part, size in zip(contents, sizes, strict=True):
        position = np.array(host_centre, dtype=float)
        position[axis] = host_centre[axis] + cursor + size / 2.0
        placements[part.id] = Placement(
            position=(float(position[0]), float(position[1]), float(position[2])),
            rotation=part.placement.rotation,
        )
        cursor += size + gap
    return placements


def _ring_outside(
    host: Part, parts: list[Part], *, axis: int = 2, radial: float = 1.0
) -> dict[str, Placement]:
    """Space parts evenly around the host, on its outside surface."""
    if not parts:
        return {}
    extent = _extent(host)
    others = [i for i in range(3) if i != axis]
    reach = (float(max(extent[others[0]], extent[others[1]])) / 2.0) * radial
    centre = np.asarray(host.placement.position, dtype=float)
    placements: dict[str, Placement] = {}
    count = len(parts)
    for index, part in enumerate(parts):
        angle = 2.0 * math.pi * index / count
        own = _extent(part)
        offset = reach + float(max(own[others[0]], own[others[1]])) / 2.0
        position = np.array(centre, dtype=float)
        position[others[0]] += offset * math.cos(angle)
        position[others[1]] += offset * math.sin(angle)
        # Sit them toward the back of the host, where a thruster lives.
        position[axis] -= float(extent[axis]) * 0.28
        placements[part.id] = Placement(
            position=(float(position[0]), float(position[1]), float(position[2])),
            rotation=part.placement.rotation,
        )
    return placements


def _stack_clear(anchor: np.ndarray, parts: list[Part], axis: int = 2) -> dict[str, Placement]:
    """Put whatever is left in a row, clear of everything already placed."""
    placements: dict[str, Placement] = {}
    cursor = float(anchor[axis])
    for part in parts:
        size = float(_extent(part)[axis])
        position = np.array(anchor, dtype=float)
        position[axis] = cursor + size / 2.0
        placements[part.id] = Placement(
            position=(float(position[0]), float(position[1]), float(position[2])),
            rotation=part.placement.rotation,
        )
        cursor += size * 1.25
    return placements


def arrange(design: Design, *, force: bool = False) -> Design:
    """Fill in every part's position, leaving any the brief already set.

    ``force`` re-places everything, which is what a caller wants after
    changing a part's size and before drawing it again.
    """
    parts = list(design.parts)
    if not parts:
        return design
    fixed = {p.id for p in parts if _has_placement(p) and not force}
    placements: dict[str, Placement] = {p.id: p.placement for p in parts if p.id in fixed}

    enclosures = [p for p in parts if _is_enclosure(p) and p.solid is not None]
    # The biggest enclosure is the body; the rest are lids and covers that
    # sit on its ends rather than holding anything themselves.
    enclosures.sort(key=lambda p: -float(np.prod(np.maximum(_extent(p), 1e-9))))
    body = enclosures[0] if enclosures else None

    if body is not None and body.id not in placements:
        placements[body.id] = Placement()

    remaining = [p for p in parts if p.id not in placements and p.solid is not None]

    if body is not None:
        axis = int(np.argmax(_extent(body)))
        # Covers go on the ends of the body.
        covers = [p for p in remaining if _is_enclosure(p)]
        span = float(_extent(body)[axis]) / 2.0
        for index, cover in enumerate(covers):
            position = np.zeros(3)
            direction = 1.0 if index % 2 == 0 else -1.0
            position[axis] = direction * (span + float(_extent(cover)[axis]) * 0.25)
            placements[cover.id] = Placement(
                position=(float(position[0]), float(position[1]), float(position[2])),
                rotation=cover.placement.rotation,
            )
        remaining = [p for p in remaining if p.id not in placements]

        external = [p for p in remaining if _is_external(p)]
        internal = [p for p in remaining if p not in external]
        placements.update(_pack_inside(body, _connection_order(design, internal)))
        placements.update(_ring_outside(body, external, axis=axis))
        remaining = [p for p in remaining if p.id not in placements]

    if remaining:
        anchor = np.zeros(3)
        if placements:
            anchor = np.array(
                [max(p.position[i] for p in placements.values()) for i in range(3)],
                dtype=float,
            )
        placements.update(_stack_clear(anchor, _connection_order(design, remaining)))

    updated = tuple(
        replace(part, placement=placements.get(part.id, part.placement)) for part in parts
    )
    return auto_explode(design.with_parts(updated))


def auto_explode(design: Design) -> Design:
    """Give every part a direction to travel in an exploded view.

    A part moves away from the assembly's centre along the axis it is
    furthest out on, so the pieces separate the way they would come off in
    a workshop rather than flying apart at random.
    """
    if not design.parts:
        return design
    positions = np.array(
        [p.placement.position for p in design.parts if p.solid is not None], dtype=float
    )
    if len(positions) == 0:
        return design
    centre = positions.mean(axis=0)
    updated = []
    for part in design.parts:
        if part.solid is None or part.explode not in ((0.0, 0.0, 1.0), (0, 0, 1)):
            updated.append(part)
            continue
        offset = np.asarray(part.placement.position, dtype=float) - centre
        if float(np.linalg.norm(offset)) < 1e-9:
            # Dead centre: send it up, which is where the deepest part of an
            # assembly goes when the rest is lifted off it.
            direction = np.array([0.0, 0.0, 1.0])
        else:
            axis = int(np.argmax(np.abs(offset)))
            direction = np.zeros(3)
            direction[axis] = math.copysign(1.0, offset[axis])
            # A gentle sideways component keeps parts on the same axis from
            # sliding along each other's trails.
            secondary = np.argsort(-np.abs(offset))[1]
            direction[secondary] = math.copysign(0.35, offset[secondary] or 1.0)
        scale = 0.55 + 0.45 * float(np.linalg.norm(offset)) / (
            float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0))) or 1.0
        )
        vector = direction * scale
        updated.append(
            replace(part, explode=(float(vector[0]), float(vector[1]), float(vector[2])))
        )
    return design.with_parts(tuple(updated))


def interference(design: Design, *, tolerance: float = 1e-4) -> tuple[dict[str, Any], ...]:
    """Which parts occupy the same space, once they have been placed.

    Bounding boxes only, which catches the gross clashes and never reports
    one that is not there beyond the box approximation. A part declared as
    an enclosure is allowed to overlap the parts it holds, since that is
    what holding them means.
    """
    boxes: list[tuple[Part, np.ndarray, np.ndarray]] = []
    for part in design.parts:
        if part.solid is None:
            continue
        mesh = part.solid.mesh().transformed(part.placement)
        low, high = mesh.bounds()
        boxes.append((part, np.asarray(low), np.asarray(high)))
    clashes: list[dict[str, Any]] = []
    for index, (part_a, low_a, high_a) in enumerate(boxes):
        for part_b, low_b, high_b in boxes[index + 1 :]:
            if _is_enclosure(part_a) or _is_enclosure(part_b):
                continue
            overlap = np.minimum(high_a, high_b) - np.maximum(low_a, low_b)
            if float(overlap.min()) <= tolerance:
                continue
            volume = float(np.prod(overlap))
            clashes.append(
                {
                    "parts": (part_a.id, part_b.id),
                    "overlap_m3": volume,
                    "overlap_mm": [float(v * 1000.0) for v in overlap],
                    "plain": (
                        f"The {part_a.lay_name or part_a.name.lower()} and the "
                        f"{part_b.lay_name or part_b.name.lower()} occupy the same space, "
                        f"overlapping by {overlap[0] * 1000:.0f} by {overlap[1] * 1000:.0f} "
                        f"by {overlap[2] * 1000:.0f} millimetres. One of them has to move."
                    ),
                }
            )
    return tuple(clashes)
