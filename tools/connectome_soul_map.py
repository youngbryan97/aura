#!/usr/bin/env python3
"""Export the connectome in the shape the soul map already reads.

She has a viewer: `interface/static/mycelial.html`, a 3D force graph fed by
`/api/mycelial/graph`. It draws the mycelium. The connectome is a different
graph of the same organism, measured rather than declared, and it has never had
a picture.

This writes one payload in that viewer's own shape — `nodes` with an id, a
label, a type, a description and a centrality, `links` with a source, a target
and a weight — so the same viewer can draw it with no changes to the viewer.

Operationally: this measures nothing about a soul. "Soul map" is the name of an
existing viewer in this repository — `interface/static/mycelial.html` — and this
writes a payload in that viewer's shape so the connectome can be drawn by it.
What is drawn is one node per source file, sized by how many call sites land on
its edges, and one link per pair of files that call each other.

Aggregated to the module. Forty-eight thousand cells in a force-directed scene
is the hairball the viewer's own comments describe fighting; the module is the
level a person can read, and it is the level the layer analyses already work at.

The mesh comes too. `G_neural` put the 64 cortical columns into the same graph
as the code, and a picture that leaves them out would be a picture of the half
of her that is written in Python.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

#: One colour per region, from the regions the reconstruction actually names.
#: The first version matched substrings against a list of eight architectural
#: words and 435 of 699 drawn modules fell through to the default, so the plate
#: was a grey blob with a legend that said "elsewhere". These are the regions by
#: size, and anything past the twelfth is genuinely small.
REGION_COLOURS: dict[str, str] = {
    "brain": "#4f9dff",
    "runtime": "#8892a6",
    "learning": "#7ee787",
    "cognition": "#c77dff",
    "consciousness": "#8a2be2",
    "memory": "#37d3a4",
    "skills": "#ff8c42",
    "agency": "#ffc857",
    "interiority": "#ff6b8a",
    "conversation": "#5ad1e6",
    "environment": "#b4c96b",
    "perception": "#f2a65a",
    "mesh": "#ff4d6d",
}
DEFAULT_COLOUR = "#59636f"


def _region_colour(region: str) -> str:
    """Exact region first. A substring match put every module whose region
    merely contained a listed word into that colour, which is how `auth` and
    `autonomic` ended up the same shade as `agency`."""
    exact = REGION_COLOURS.get(region)
    if exact:
        return exact
    return DEFAULT_COLOUR


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "artifacts" / "connectome" / "soul_map_graph.json",
    )
    parser.add_argument("--observed", type=Path, default=None)
    parser.add_argument("--max-links", type=int, default=2600)
    parser.add_argument("--no-mesh", action="store_true")
    args = parser.parse_args()

    from core.connectome.volume import VolumeReconstructor

    reconstructor = VolumeReconstructor(REPO)
    reconstructor.scan()
    snapshot = reconstructor.build()

    if args.observed and args.observed.exists():
        from core.connectome.activity import ObservedEdges
        from core.connectome.proofreading import repair_observed_splits

        observed = ObservedEdges()
        payload = json.loads(args.observed.read_text())
        for key, count in payload.get("counts", {}).items():
            pre, _, post = key.partition(">")
            observed.counts[(pre, post)] = int(count)
        ledger = repair_observed_splits(snapshot, observed.without_self_pairs())
        snapshot = ledger.apply(snapshot)

    cells_per_module: dict[str, int] = defaultdict(int)
    region_of: dict[str, str] = {}
    for unit in snapshot.units.values():
        cells_per_module[unit.neuropil] += 1
        region_of.setdefault(unit.neuropil, unit.region)

    module_of = {uid: unit.neuropil for uid, unit in snapshot.units.items()}
    weights: dict[tuple[str, str], int] = defaultdict(int)
    for (pre, post, _kind), connection in snapshot.connections.items():
        source = module_of.get(pre)
        target = module_of.get(post)
        if source is None or target is None or source == target:
            continue
        weights[(source, target)] += connection.contacts

    degree: dict[str, int] = defaultdict(int)
    for (source, target), weight in weights.items():
        degree[source] += weight
        degree[target] += weight
    busiest = max(degree.values()) if degree else 1

    nodes: list[dict[str, Any]] = []
    for module, cells in sorted(cells_per_module.items()):
        region = region_of.get(module, "core")
        nodes.append(
            {
                "id": module,
                "label": module.split(".")[-1],
                "type": region,
                "color": _region_colour(region),
                "description": (
                    f"{cells} cells in {module}, {degree.get(module, 0)} call sites "
                    "on its edges"
                ),
                "centrality": round(degree.get(module, 0) / busiest, 4),
                "confidence": 1.0,
                "cells": cells,
            }
        )

    links = [
        {"source": source, "target": target, "weight": weight}
        for (source, target), weight in weights.items()
    ]
    links.sort(key=lambda link: -link["weight"])
    if args.max_links and len(links) > args.max_links:
        links = links[: args.max_links]

    mesh_summary: dict[str, Any] = {}
    if not args.no_mesh:
        from core.connectome.neural import (
            build_mesh_layer,
            join_to_code,
            signal_can_cross,
        )

        layer = join_to_code(build_mesh_layer(), snapshot)
        mesh_summary = {"layer": layer.summary(), "crossing": signal_can_cross(layer)}
        for column in layer.columns:
            tier = layer.tiers.get(column, "?")
            nodes.append(
                {
                    "id": column,
                    "label": column.rsplit(":", 1)[-1],
                    "type": f"mesh_{tier}",
                    "color": REGION_COLOURS["mesh"],
                    "description": f"cortical column, {tier} tier, 64 neurons",
                    "centrality": 0.2,
                    "confidence": 1.0,
                    "cells": 64,
                }
            )
        for (pre, post), weight in layer.edges.items():
            links.append({"source": pre, "target": post, "weight": abs(weight) * 10})
        # The seam is the point of including the mesh at all: these are the
        # edges that make the substrate and the code one graph.
        known = {node["id"] for node in nodes}
        for (caller, column), method in layer.seam_in.items():
            source = module_of.get(caller)
            if source in known:
                links.append(
                    {"source": source, "target": column, "weight": 2, "seam": method}
                )
        for (column, caller), method in layer.seam_out.items():
            target = module_of.get(caller)
            if target in known:
                links.append(
                    {"source": column, "target": target, "weight": 2, "seam": method}
                )

    payload = {
        "nodes": nodes,
        "links": links,
        "meta": {
            "source": "core/connectome",
            "cells": len(snapshot.units),
            "modules": len(cells_per_module),
            "connections": len(snapshot.connections),
            "links_drawn": len(links),
            "mesh": mesh_summary,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload))
    print(
        json.dumps(
            {
                "written": str(args.out),
                "nodes": len(nodes),
                "links": len(links),
                "bytes": args.out.stat().st_size,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
