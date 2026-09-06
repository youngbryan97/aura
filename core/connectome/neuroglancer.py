"""core/connectome/neuroglancer.py — exporting the map so it can be looked at.

Neuroglancer is the viewer the fly and the human reconstructions are published
in: a WebGL renderer with a JSON state, cross-sections and a 3-D pane, layers
for images, segmentations, meshes and annotations. It reads a small number of
well-specified formats, and one of them needs no image volume at all.

That is the one used here. Aura has no electron micrographs and never will, so
the export is a segment-property map and an annotation layer: a point per cell,
a line per strong connection, and properties that carry the class, the region
and the laminar band. The coordinates are not decoration — depth is the cell's
own trophic level, so a reader is looking at the hierarchy rather than at a
spring layout that would have come out differently on the next run.

The state is a plain dictionary. Written to a file it can be dropped into any
Neuroglancer instance; appended to a viewer URL after ``#!`` it opens directly.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .microcircuit import LAYER_ORDER, LaminarAssignment
from .types import CellClass, ConnectomeSnapshot, EdgeKind

logger = logging.getLogger("Aura.Connectome.Neuroglancer")

__all__ = [
    "CELL_CLASS_COLOURS",
    "layout_cells",
    "segment_properties",
    "viewer_state",
    "write_export",
]

#: One colour per class, chosen so excitatory and inhibitory separate for a
#: red-green colour-blind reader as well: the pair differs in lightness, not
#: only in hue.
CELL_CLASS_COLOURS: dict[str, str] = {
    str(CellClass.EXCITATORY): "#f2b134",
    str(CellClass.INHIBITORY): "#3b6ea5",
    str(CellClass.MODULATORY): "#8e6fb5",
    str(CellClass.GLIAL): "#7a8b8b",
}

#: Nanometres per unit of trophic height and per layout step. Neuroglancer wants
#: physical units, and a scale where one hierarchy level is a micron puts the
#: whole system in a volume a reader can navigate at default zoom.
NM_PER_LEVEL: float = 1_000.0
NM_PER_STEP: float = 120.0


def layout_cells(
    snapshot: ConnectomeSnapshot,
    assignment: LaminarAssignment | None = None,
    *,
    limit: int = 20_000,
) -> dict[str, tuple[float, float, float]]:
    """Place every cell in space: depth from hierarchy, position from region.

    Regions are laid out around a circle in name order and cells within a
    region on a spiral inside it, so the same snapshot always produces the same
    picture and two exports can be compared by eye without one of them having
    been shaken into a different shape.
    """
    import math

    heights = assignment.heights if assignment else {}
    regions = sorted({unit.region for unit in snapshot.units.values()})
    region_angle = {
        region: (2.0 * math.pi * index / max(1, len(regions)))
        for index, region in enumerate(regions)
    }
    per_region: dict[str, int] = {}
    coordinates: dict[str, tuple[float, float, float]] = {}
    for uid in sorted(snapshot.units)[:limit]:
        unit = snapshot.units[uid]
        rank = per_region.get(unit.region, 0)
        per_region[unit.region] = rank + 1
        angle = region_angle.get(unit.region, 0.0) + 0.11 * math.sqrt(rank)
        radius = NM_PER_STEP * (12.0 + 3.0 * math.sqrt(rank))
        depth = heights.get(uid, 0.0) * NM_PER_LEVEL
        coordinates[uid] = (
            radius * math.cos(angle),
            radius * math.sin(angle),
            depth,
        )
    return coordinates


def segment_properties(
    snapshot: ConnectomeSnapshot,
    assignment: LaminarAssignment | None = None,
    *,
    labels: Mapping[str, str] | None = None,
    limit: int = 20_000,
) -> dict[str, Any]:
    """A ``neuroglancer_segment_properties`` document, inline form.

    Segment ids have to be integers, so each cell gets one from its position in
    sorted order and the mapping is written out beside the properties. A hash
    would be shorter and would change every time a cell was added, which is the
    opposite of what an id is for.
    """
    ordered = sorted(snapshot.units)[:limit]
    ids = [str(index + 1) for index in range(len(ordered))]
    names = [snapshot.units[uid].name for uid in ordered]
    classes = [str(snapshot.units[uid].cell_class) for uid in ordered]
    regions = [snapshot.units[uid].region for uid in ordered]
    bands = [
        (assignment.layer.get(uid, "") if assignment else "") or "unassigned" for uid in ordered
    ]
    types = [(labels or {}).get(uid, "") for uid in ordered]
    return {
        "@type": "neuroglancer_segment_properties",
        "inline": {
            "ids": ids,
            "properties": [
                {"id": "label", "type": "label", "values": names},
                {"id": "cell_class", "type": "string", "values": classes},
                {"id": "region", "type": "string", "values": regions},
                {"id": "layer", "type": "string", "values": bands},
                {"id": "cell_type", "type": "string", "values": types},
            ],
        },
        "aura": {"uid_by_segment": dict(zip(ids, ordered, strict=True))},
    }


def viewer_state(
    snapshot: ConnectomeSnapshot,
    assignment: LaminarAssignment | None = None,
    *,
    max_cells: int = 4_000,
    max_edges: int = 4_000,
    min_contacts: int = 4,
) -> dict[str, Any]:
    """A Neuroglancer state with a point per cell and a line per strong pair.

    The edge layer is restricted to connections carrying at least ``min_contacts``
    call sites, which is H01's threshold for a strong connection. Drawing all
    eighty thousand would render as a solid block and hide the thing worth
    seeing.
    """
    coordinates = layout_cells(snapshot, assignment)
    # Pick what to draw by traffic, not by name. Taking the first few thousand
    # cells in sorted order gives a subgraph with almost no strong connections
    # in it, which draws a picture of nothing.
    degree: dict[str, int] = {}
    for conn in snapshot.connections.values():
        if conn.kind is not EdgeKind.DRIVE:
            continue
        degree[conn.pre] = degree.get(conn.pre, 0) + conn.contacts
        degree[conn.post] = degree.get(conn.post, 0) + conn.contacts
    strong_endpoints = {
        uid
        for conn in snapshot.connections.values()
        if conn.kind is EdgeKind.DRIVE and conn.contacts >= min_contacts
        for uid in (conn.pre, conn.post)
        if uid in coordinates
    }
    ranked = sorted(
        (uid for uid in coordinates if uid in degree),
        key=lambda uid: (-degree.get(uid, 0), uid),
    )
    chosen_set: set[str] = set()
    for uid in sorted(strong_endpoints, key=lambda u: (-degree.get(u, 0), u)):
        if len(chosen_set) >= max_cells:
            break
        chosen_set.add(uid)
    for uid in ranked:
        if len(chosen_set) >= max_cells:
            break
        chosen_set.add(uid)
    chosen = sorted(chosen_set)

    points: list[dict[str, Any]] = []
    for index, uid in enumerate(chosen):
        unit = snapshot.units[uid]
        x, y, z = coordinates[uid]
        points.append(
            {
                "type": "point",
                "id": f"cell{index + 1}",
                "point": [round(x, 1), round(y, 1), round(z, 1)],
                "description": f"{unit.name} [{unit.cell_class}] {unit.region}",
            }
        )

    lines: list[dict[str, Any]] = []
    strong = sorted(
        (
            conn
            for conn in snapshot.connections.values()
            if conn.kind is EdgeKind.DRIVE
            and conn.contacts >= min_contacts
            and conn.pre in chosen_set
            and conn.post in chosen_set
        ),
        key=lambda c: (-c.contacts, c.pre, c.post),
    )[:max_edges]
    for index, conn in enumerate(strong):
        start = coordinates[conn.pre]
        end = coordinates[conn.post]
        lines.append(
            {
                "type": "line",
                "id": f"edge{index + 1}",
                "pointA": [round(v, 1) for v in start],
                "pointB": [round(v, 1) for v in end],
                "description": f"{conn.contacts} contacts",
            }
        )

    dimensions = {"x": [1e-9, "m"], "y": [1e-9, "m"], "z": [1e-9, "m"]}
    return {
        "dimensions": dimensions,
        "position": [0.0, 0.0, 0.0],
        "crossSectionScale": 40.0,
        "projectionScale": 24_000.0,
        "layers": [
            {
                "type": "annotation",
                "name": "cells",
                "source": {"url": "local://annotations", "transform": {"outputDimensions": dimensions}},
                "annotations": points,
                "annotationColor": CELL_CLASS_COLOURS[str(CellClass.EXCITATORY)],
                "shader": "void main() { setColor(defaultColor()); setPointMarkerSize(4.0); }",
            },
            {
                "type": "annotation",
                "name": "strong connections",
                "source": {"url": "local://annotations", "transform": {"outputDimensions": dimensions}},
                "annotations": lines,
                "annotationColor": "#c1554b",
            },
        ],
        "layout": "3d",
        "selectedLayer": {"layer": "cells", "visible": True},
        "aura": {
            "cells_drawn": len(points),
            "edges_drawn": len(lines),
            "min_contacts": min_contacts,
            "layers_present": list(LAYER_ORDER) if assignment else [],
            "digest": snapshot.digest(),
        },
    }


def write_export(
    snapshot: ConnectomeSnapshot,
    directory: str | Path,
    assignment: LaminarAssignment | None = None,
    *,
    labels: Mapping[str, str] | None = None,
    source: str = "core.connectome.neuroglancer",
) -> dict[str, str]:
    """Write the segment properties and the viewer state through the gateway."""
    from core.runtime.file_write_gateway import get_file_write_gateway

    target = Path(directory)
    gateway = get_file_write_gateway()
    gateway.ensure_directory(target, source=source)
    properties = segment_properties(snapshot, assignment, labels=labels)
    state = viewer_state(snapshot, assignment)
    files = {
        "segment_properties": str(target / "segment_properties.json"),
        "viewer_state": str(target / "neuroglancer_state.json"),
    }
    gateway.write_text(
        files["segment_properties"],
        json.dumps(properties, indent=2, sort_keys=True),
        source=source,
    )
    gateway.write_text(
        files["viewer_state"], json.dumps(state, indent=2, sort_keys=True), source=source
    )
    return files
