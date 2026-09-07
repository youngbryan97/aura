"""core/connectome/anatomy_gate.py — a change that has to answer for what it did to the anatomy.

A self-modification is promoted on whether it works. That is necessary and it is
not sufficient, because a change can pass its probe and leave the system worse
built: one more brittle cell everything runs through, one more coupling that
carries nothing, one more channel with a writer and no reader.

This measures the shape before and after, so a promotion can be asked a second
question. Not whether the result looks more brain-like — that is aesthetics —
but seven things a reconstruction can settle:

    did useful local recurrence increase
    did a brittle single point of failure disappear
    did unnecessary cross-region coupling decline
    did the task-specific effective circuits get cleaner
    did information reach the consumers it was supposed to
    did dormant machinery become active
    did a lesion confirm the intended new causal route

Four of those need only two reconstructions. Three need recordings, and when
they are missing the axis reports that it was not measured rather than scoring
zero — an unmeasured axis and an axis that did not move are different, and
averaging them together would hide every change that was never checked.

The verdict is deliberately hard to pass and easy to fail. A change that moves
nothing measurable is not blocked; a change that makes something measurably
worse has to say so in the receipt it is promoted with, so the record carries
what it cost as well as what it bought.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .types import ConnectomeSnapshot

logger = logging.getLogger("Aura.Connectome.AnatomyGate")

__all__ = [
    "Axis",
    "AXES",
    "Quality",
    "measure_quality",
    "QualityDelta",
    "compare_quality",
    "anatomical_evidence",
]


@dataclass(frozen=True)
class Axis:
    """One thing a reconstruction can settle about a change."""

    name: str
    question: str
    higher_is_better: bool
    needs_recording: bool = False

    def improved(self, before: float, after: float) -> bool:
        return after > before if self.higher_is_better else after < before


AXES: tuple[Axis, ...] = (
    Axis(
        "local_recurrence",
        "did useful local recurrence increase",
        higher_is_better=True,
    ),
    Axis(
        "single_points_of_failure",
        "did a brittle single point of failure disappear",
        higher_is_better=False,
    ),
    Axis(
        "coupling_carrying_nothing",
        "did unnecessary cross-region coupling decline",
        higher_is_better=False,
    ),
    Axis(
        "half_wired_channels",
        "did information reach the consumers it was supposed to",
        higher_is_better=False,
    ),
    Axis(
        "effective_circuit_sharpness",
        "did the task-specific effective circuits get cleaner",
        higher_is_better=True,
        needs_recording=True,
    ),
    Axis(
        "dormant_machinery",
        "did dormant machinery become active",
        higher_is_better=False,
        needs_recording=True,
    ),
    Axis(
        "intended_route",
        "did a lesion confirm the intended new causal route",
        higher_is_better=True,
        needs_recording=True,
    ),
)


@dataclass
class Quality:
    """The shape of one reconstruction, on the axes a change is answerable for."""

    values: dict[str, float] = field(default_factory=dict)
    unmeasured: tuple[str, ...] = ()
    detail: dict[str, Any] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        return {
            "values": {k: round(v, 5) for k, v in sorted(self.values.items())},
            "unmeasured": list(self.unmeasured),
            "detail": self.detail,
        }


def measure_quality(
    snapshot: ConnectomeSnapshot,
    *,
    multilayer: Any = None,
    dataflow: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
    laminar: Any = None,
    effective: Sequence[Any] = (),
    observed: Any = None,
    intended_route: float | None = None,
    spof_sample: int = 60,
) -> Quality:
    """Measure the axes this reconstruction supports, and name the ones it does not."""
    from .lesion import measure_effect
    from .microcircuit import assign_layers, compare_to_cortex, connection_probabilities
    from .synaptology import strong_connections
    from .types import EdgeKind

    values: dict[str, float] = {}
    unmeasured: list[str] = []
    detail: dict[str, Any] = {}

    laminar = laminar if laminar is not None else assign_layers(snapshot)
    comparison = compare_to_cortex(connection_probabilities(snapshot, laminar))
    values["local_recurrence"] = float(
        comparison["orientation_free"]["aura_within_over_between"]
    )
    detail["cortex_local_recurrence"] = comparison["orientation_free"][
        "cortex_within_over_between"
    ]

    # The busiest cells are the ones a single point of failure would be among, so
    # the sample is taken from the top of the degree distribution rather than at
    # random: a lesion sweep over the whole graph is not affordable per change,
    # and a random sample would almost never contain the cells that matter.
    degree: dict[str, int] = {}
    for connection in snapshot.edges(EdgeKind.DRIVE):
        degree[connection.pre] = degree.get(connection.pre, 0) + 1
        degree[connection.post] = degree.get(connection.post, 0) + 1
    busiest = [
        uid for uid, _ in sorted(degree.items(), key=lambda kv: (-kv[1], kv[0]))[:spof_sample]
    ]
    brittle = 0
    for index, uid in enumerate(busiest):
        effect = measure_effect(snapshot, [uid], null_samples=4, max_depth=6, seed=index)
        if effect.excess_reach_loss > 0.01:
            brittle += 1
    values["single_points_of_failure"] = float(brittle)
    detail["spof_sampled"] = len(busiest)

    if dataflow is not None:
        empty = sum(
            1
            for connection in strong_connections(snapshot, threshold=4, limit=100_000)
            if not connection.same_module
            and (dataflow.get((connection.pre, connection.post)) or {}).get("carries_nothing")
        )
        values["coupling_carrying_nothing"] = float(empty)
    else:
        unmeasured.append("coupling_carrying_nothing")

    if multilayer is not None:
        from .layers import Layer

        half = 0
        for layer, write_key, read_key in (
            (Layer.VOLUME, "out", "in"),
            (Layer.IO, "write", "read"),
        ):
            for key, entry in multilayer.channels.items():
                if not key.startswith(str(layer)):
                    continue
                if bool(entry[write_key]) != bool(entry[read_key]):
                    half += 1
        values["half_wired_channels"] = float(half)
    else:
        unmeasured.append("half_wired_channels")

    usable = [graph for graph in effective if getattr(graph, "edges", None)]
    if usable:
        shares = [
            len(graph.surviving()) / len(graph.edges)
            for graph in usable
            if graph.edges
        ]
        values["effective_circuit_sharpness"] = (
            sum(shares) / len(shares) if shares else 0.0
        )
    else:
        unmeasured.append("effective_circuit_sharpness")

    if observed is not None:
        fired = {pre for pre, _ in observed.counts} | {post for _, post in observed.counts}
        values["dormant_machinery"] = float(len(set(snapshot.units) - fired))
    else:
        unmeasured.append("dormant_machinery")

    if intended_route is not None:
        values["intended_route"] = float(intended_route)
    else:
        unmeasured.append("intended_route")

    return Quality(values=values, unmeasured=tuple(unmeasured), detail=detail)


@dataclass
class QualityDelta:
    """What a change did to the shape, axis by axis."""

    moves: dict[str, dict[str, Any]] = field(default_factory=dict)
    unmeasured: tuple[str, ...] = ()

    @property
    def improved(self) -> list[str]:
        return sorted(k for k, v in self.moves.items() if v["improved"])

    @property
    def worsened(self) -> list[str]:
        return sorted(k for k, v in self.moves.items() if v["worsened"])

    @property
    def verdict(self) -> str:
        if self.worsened and not self.improved:
            return f"the anatomy is worse on {len(self.worsened)} axes and better on none"
        if self.worsened:
            return (
                f"better on {len(self.improved)} axes and worse on {len(self.worsened)}: "
                + ", ".join(self.worsened)
            )
        if self.improved:
            return f"the anatomy is better on {len(self.improved)} axes and worse on none"
        return "the anatomy did not move on any axis that was measured"

    def as_json(self) -> dict[str, Any]:
        return {
            "moves": {
                name: {k: (round(v, 5) if isinstance(v, float) else v) for k, v in move.items()}
                for name, move in sorted(self.moves.items())
            },
            "improved": self.improved,
            "worsened": self.worsened,
            "unmeasured": list(self.unmeasured),
            "verdict": self.verdict,
        }


def compare_quality(before: Quality, after: Quality) -> QualityDelta:
    """Difference two measurements, axis by axis, in the direction each wants."""
    delta = QualityDelta()
    unmeasured = set(before.unmeasured) | set(after.unmeasured)
    for axis in AXES:
        if axis.name in unmeasured:
            continue
        if axis.name not in before.values or axis.name not in after.values:
            unmeasured.add(axis.name)
            continue
        was = before.values[axis.name]
        now = after.values[axis.name]
        delta.moves[axis.name] = {
            "question": axis.question,
            "before": was,
            "after": now,
            "change": now - was,
            "improved": axis.improved(was, now) and now != was,
            "worsened": (not axis.improved(was, now)) and now != was,
        }
    delta.unmeasured = tuple(sorted(unmeasured))
    return delta


def anatomical_evidence(delta: QualityDelta, *, change: str = "") -> str:
    """The line a promotion receipt carries about what the change did to the shape.

    Written to be read in a ledger months later, so it names what got worse
    before what got better. A receipt that only records the gain is the reason
    nobody can tell later what a change cost.
    """
    parts: list[str] = []
    if change:
        parts.append(change)
    for name in delta.worsened:
        move = delta.moves[name]
        parts.append(f"worse: {name} {move['before']:.4g} to {move['after']:.4g}")
    for name in delta.improved:
        move = delta.moves[name]
        parts.append(f"better: {name} {move['before']:.4g} to {move['after']:.4g}")
    if delta.unmeasured:
        parts.append("not measured: " + ", ".join(delta.unmeasured))
    if not delta.moves:
        parts.append("no axis moved")
    return "; ".join(parts)
