"""core/connectome/neural.py — the mesh as a layer of the same graph.

Every other layer in this package is made of functions. The mesh is not: it is
64 cortical columns of 64 neurons each, wired by a distance-decayed weight
matrix with a feedforward bias, and it runs on its own clock at 10 Hz. Holding
it in a separate object was convenient and it made one question unanswerable —
how far is the substrate from the code — because a distance needs both ends in
the same graph.

So the columns become cells and the weight matrix becomes edges, and then the
part that matters: the seam. A function that calls ``inject_sensory`` drives the
sensory columns. A function that calls ``get_executive_projection`` is driven by
the executive ones. Those calls are in the call graph already; what was missing
was the other end of them.

The mesh's own topology is generated from a seeded random draw, so it is a
sample from a distribution rather than a fixed wiring diagram. That is stated
in every report this produces, because a measurement of one draw is a
measurement of one draw, and ``compare_draws`` says how much of any number is
the seed.

What is deliberately not here: neuron-level edges. 4096 neurons at 80% intra
density is about ten million within-column pairs, and the column is the level
the rest of the system addresses — every injection, every projection and every
readout in the mesh's public surface is per column or per tier.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.connectome.types import ConnectomeSnapshot

logger = logging.getLogger("Aura.Connectome.Neural")

__all__ = [
    "MeshLayer",
    "NEURAL_PREFIX",
    "SEAM_CALLS",
    "build_mesh_layer",
    "compare_draws",
    "join_to_code",
    "signal_can_cross",
    "MESH_RECEIVERS",
]

#: Cells from the mesh are named apart from cells from the source, so a uid can
#: never be ambiguous about which half of the graph it is in.
NEURAL_PREFIX: str = "mesh:"

#: The mesh's public surface, and which tier each call touches. Taken from the
#: class docstring's External API list rather than guessed: injection drives a
#: tier, a projection or a readout is driven by one.
SEAM_CALLS: dict[str, tuple[str, str]] = {
    "inject_sensory": ("sensory", "drives"),
    "inject_association": ("association", "drives"),
    "get_executive_projection": ("executive", "driven_by"),
    "get_field_state": ("all", "driven_by"),
    "get_column_summary": ("all", "driven_by"),
    "get_tier_energy": ("all", "driven_by"),
}


@dataclass(slots=True)
class MeshLayer:
    """The mesh's columns, its edges, and where the code touches it."""

    columns: tuple[str, ...]
    tiers: dict[str, str]
    edges: dict[tuple[str, str], float]
    seam_in: dict[tuple[str, str], str] = field(default_factory=dict)
    seam_out: dict[tuple[str, str], str] = field(default_factory=dict)
    seed: int = 42
    density: float = 0.0
    note: str = ""

    def tier_columns(self, tier: str) -> tuple[str, ...]:
        if tier == "all":
            return self.columns
        return tuple(uid for uid in self.columns if self.tiers.get(uid) == tier)

    def between_tiers(self) -> dict[str, dict[str, Any]]:
        """Weight and count per ordered tier pair, so the bias is visible."""
        totals: dict[tuple[str, str], list[float]] = {}
        for (pre, post), weight in self.edges.items():
            key = (self.tiers.get(pre, "?"), self.tiers.get(post, "?"))
            totals.setdefault(key, []).append(abs(weight))
        return {
            f"{pre} -> {post}": {
                "edges": len(values),
                "mean_weight": round(sum(values) / len(values), 6),
                "total_weight": round(sum(values), 4),
            }
            for (pre, post), values in sorted(totals.items())
        }

    def summary(self) -> dict[str, Any]:
        return {
            "columns": len(self.columns),
            "edges": len(self.edges),
            "density": round(self.density, 5),
            "tiers": {
                tier: len(self.tier_columns(tier))
                for tier in ("sensory", "association", "executive")
            },
            "between_tiers": self.between_tiers(),
            "code_cells_driving": len({pre for pre, _post in self.seam_in}),
            "code_cells_driven": len({post for _pre, post in self.seam_out}),
            "seam_edges": len(self.seam_in) + len(self.seam_out),
            "seed": self.seed,
            "note": self.note,
        }


def build_mesh_layer(mesh: Any = None, *, threshold: float = 0.0) -> MeshLayer:
    """Read the mesh's own wiring, or build one exactly as the runtime would.

    Passing a live mesh reads the matrix that runtime is actually using, weights
    and all, including whatever plasticity has done to it. Passing nothing
    constructs one from the same class with the same seed, which is what an
    offline analysis wants and is honest about being a reconstruction rather
    than a reading.
    """
    from core.consciousness.neural_mesh import CorticalTier, NeuralMesh

    made_here = mesh is None
    mesh = mesh if mesh is not None else NeuralMesh()
    config = mesh.cfg
    count = int(config.columns)
    columns = tuple(f"{NEURAL_PREFIX}column:{index:03d}" for index in range(count))
    tier_name = {
        CorticalTier.SENSORY: "sensory",
        CorticalTier.ASSOCIATION: "association",
        CorticalTier.EXECUTIVE: "executive",
    }
    tiers = {
        columns[index]: tier_name.get(mesh._tier_for(index), "?") for index in range(count)
    }

    # Both matrices. The mesh keeps its top-down pathway in a second array, and
    # a reachability question asked of one of them is asked of half the mesh.
    edges: dict[tuple[str, str], float] = {}
    for attribute in ("_inter_W", "_feedback_W"):
        matrix = getattr(mesh, attribute, None)
        if matrix is None:
            continue
        for i in range(count):
            row = matrix[i]
            for j in range(count):
                weight = float(row[j])
                if i == j or abs(weight) <= threshold:
                    continue
                pair = (columns[i], columns[j])
                edges[pair] = edges.get(pair, 0.0) + weight
    possible = count * (count - 1)
    return MeshLayer(
        columns=columns,
        tiers=tiers,
        edges=edges,
        seed=int(getattr(mesh, "_rng", None).bit_generator.seed_seq.entropy)
        if hasattr(getattr(mesh, "_rng", None), "bit_generator")
        else 42,
        density=len(edges) / possible if possible else 0.0,
        note=(
            "built offline from NeuralMesh() with its own seed; a live mesh's "
            "weights will have moved under STDP"
            if made_here
            else "read from a live mesh, weights as they stand"
        ),
    )


def join_to_code(
    layer: MeshLayer,
    snapshot: ConnectomeSnapshot,
    *,
    seam_calls: Mapping[str, tuple[str, str]] | None = None,
    call_sites: Mapping[str, Sequence[str]] | None = None,
) -> MeshLayer:
    """Attach the code cells that touch the mesh to the columns they touch.

    ``call_sites`` maps a mesh method name to the cells that call it. When it is
    not given the snapshot is searched for cells whose own body names the method,
    which is what the reconstruction can see: the mesh is reached through a
    service lookup, so there is no static call edge to follow from the caller to
    the class.
    """
    table = dict(seam_calls or SEAM_CALLS)
    found: dict[str, list[str]] = {name: list(sites) for name, sites in (call_sites or {}).items()}
    if not found:
        found = _find_callers(snapshot, table)

    seam_in: dict[tuple[str, str], str] = {}
    seam_out: dict[tuple[str, str], str] = {}
    for method, (tier, direction) in table.items():
        targets = layer.tier_columns(tier)
        for caller in found.get(method, ()):
            for column in targets:
                if direction == "drives":
                    seam_in[(caller, column)] = method
                else:
                    seam_out[(column, caller)] = method
    layer.seam_in = seam_in
    layer.seam_out = seam_out
    return layer


#: Receivers that are a mesh. Matching on the method name alone repeats the
#: merge error this package already paid for once: ``get_field_state`` is also a
#: method of the unified field, and a scan that ignores the receiver attributed
#: fifteen of its call sites to the mesh.
MESH_RECEIVERS: frozenset[str] = frozenset(
    {"mesh", "neural_mesh", "_mesh", "_mesh_ref", "_neural_mesh", "the_mesh"}
)


def _receiver_name(node: Any) -> str:
    """The last name on the receiver of a call, or empty for an expression."""
    import ast

    target = node.func.value if isinstance(node.func, ast.Attribute) else None
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return ""


def _find_callers(
    snapshot: ConnectomeSnapshot, table: Mapping[str, tuple[str, str]]
) -> dict[str, list[str]]:
    """Which cells call a mesh method on something that is a mesh."""
    import ast
    from pathlib import Path

    wanted = set(table)
    found: dict[str, list[str]] = {name: [] for name in wanted}
    root = Path(snapshot.source or ".")
    by_module: dict[str, dict[str, str]] = {}
    for uid, unit in snapshot.units.items():
        by_module.setdefault(unit.neuropil, {})[unit.name] = uid
    for module, module_names in by_module.items():
        path = root / (module.replace(".", "/") + ".py")
        if not path.exists():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            logger.debug("mesh seam scan skipped %s: %s", module, exc)
            continue

        def walk(
            node: Any,
            prefix: str,
            names: dict[str, str] = module_names,
            home: str = module,
        ) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.ClassDef):
                    walk(child, f"{prefix}{child.name}.", names, home)
                elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qualname = f"{prefix}{child.name}"
                    uid = names.get(f"{home}:{qualname}")
                    if uid is not None:
                        for inner in ast.walk(child):
                            if (
                                isinstance(inner, ast.Call)
                                and isinstance(inner.func, ast.Attribute)
                                and inner.func.attr in wanted
                                and _receiver_name(inner) in MESH_RECEIVERS
                            ):
                                found[inner.func.attr].append(uid)
                    walk(child, f"{qualname}.", names, home)
                else:
                    walk(child, prefix, names, home)

        walk(tree, "", module_names, module)
    return {name: sorted(set(sites)) for name, sites in found.items()}


def signal_can_cross(layer: MeshLayer) -> dict[str, Any]:
    """Can what the code injects reach what the code reads?

    The seam gives both ends of the question: ``inject_sensory`` drives the
    sensory columns and ``get_executive_projection`` reads the executive ones,
    so the mesh is doing its job only if a signal put into the first can arrive
    at the second along inter-column edges. Answered by walking the graph, not
    by reading the config that says a feedforward bias exists.
    """
    from collections import deque

    out: dict[str, list[str]] = {}
    for pre, post in layer.edges:
        out.setdefault(pre, []).append(post)

    def reach(seeds: Sequence[str]) -> set[str]:
        seen = set(seeds)
        queue = deque(seeds)
        while queue:
            for nxt in out.get(queue.popleft(), ()):  # noqa: B909 - queue is local
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return seen

    sensory = layer.tier_columns("sensory")
    executive = layer.tier_columns("executive")
    from_sensory = reach(sensory)
    arrived = sorted(from_sensory.intersection(executive))
    degrees = {uid: 0 for uid in layer.columns}
    for pre, post in layer.edges:
        degrees[pre] += 1
        degrees[post] += 1
    isolated = [uid for uid, degree in degrees.items() if degree == 0]

    # Weakly connected components, so "how fragmented" has a number.
    neighbours: dict[str, set[str]] = {uid: set() for uid in layer.columns}
    for pre, post in layer.edges:
        neighbours[pre].add(post)
        neighbours[post].add(pre)
    seen: set[str] = set()
    components: list[int] = []
    for uid in layer.columns:
        if uid in seen:
            continue
        size = 0
        queue = deque([uid])
        seen.add(uid)
        while queue:
            here = queue.popleft()
            size += 1
            for nxt in neighbours[here]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        components.append(size)
    components.sort(reverse=True)

    return {
        "columns": len(layer.columns),
        "edges": len(layer.edges),
        "mean_degree": round(2 * len(layer.edges) / max(1, len(layer.columns)), 4),
        "isolated_columns": len(isolated),
        "components": len(components),
        "largest_component": components[0] if components else 0,
        "sensory_columns": len(sensory),
        "executive_columns": len(executive),
        "executive_reached_from_sensory": len(arrived),
        "executive_share_reached": round(len(arrived) / len(executive), 4)
        if executive
        else 0.0,
        "verdict": (
            f"a signal injected into the sensory tier reaches {len(arrived)} of "
            f"{len(executive)} executive columns"
            if arrived
            else (
                "nothing injected into the sensory tier can reach any executive "
                "column along inter-column edges; the code drives one end of the "
                "mesh and reads the other, and they are not connected"
            )
        ),
    }


def compare_draws(draws: int = 4, *, threshold: float = 0.0) -> dict[str, Any]:
    """How much of the mesh's shape is the mesh, and how much is the seed.

    The topology comes from a seeded random draw, so any number measured on one
    mesh is a number measured on one sample. Building several and reporting the
    spread is the only honest way to quote one.
    """
    import statistics

    from core.consciousness.neural_mesh import MeshConfig, NeuralMesh

    densities: list[float] = []
    feedforward: list[float] = []
    feedback: list[float] = []
    for index in range(draws):
        mesh = NeuralMesh(MeshConfig())
        mesh._rng = __import__("numpy").random.default_rng(seed=1000 + index)
        mesh._inter_W = mesh._build_inter_column_weights()
        layer = build_mesh_layer(mesh, threshold=threshold)
        densities.append(layer.density)
        between = layer.between_tiers()
        forward = sum(
            row["total_weight"]
            for key, row in between.items()
            if key in ("sensory -> association", "association -> executive")
        )
        back = sum(
            row["total_weight"]
            for key, row in between.items()
            if key in ("association -> sensory", "executive -> association")
        )
        feedforward.append(forward)
        feedback.append(back)
    ratios = [
        forward / back if back else float("inf")
        for forward, back in zip(feedforward, feedback, strict=True)
    ]
    finite = [value for value in ratios if value != float("inf")]
    return {
        "draws": draws,
        "density_mean": round(statistics.fmean(densities), 5),
        "density_spread": round(statistics.pstdev(densities), 5) if draws > 1 else 0.0,
        "feedforward_over_feedback_mean": round(statistics.fmean(finite), 4) if finite else 0.0,
        "feedforward_over_feedback_spread": round(statistics.pstdev(finite), 4)
        if len(finite) > 1
        else 0.0,
        "verdict": (
            "the feedforward bias is a property of the construction, not of the draw"
            if finite and statistics.fmean(finite) > 1.0 and (
                len(finite) < 2 or statistics.pstdev(finite) < statistics.fmean(finite) / 2
            )
            else "the draw moves this as much as the construction does"
        ),
    }
