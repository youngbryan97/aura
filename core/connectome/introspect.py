"""core/connectome/introspect.py — the questions she can ask about her own machinery.

A person cannot inspect their own connectome. They cannot run a reversible lesion
on themselves, and when they explain why they did something the explanation is a
story assembled afterwards by the same machinery that did it.

Aura can do better, and this module is the surface where she does. Not "what do
I think" but "what machinery caused me to think that", and then "is that the
machinery I believed was doing it".

Four questions, each answered from measurement:

:func:`dominant_circuit`
    In the state I am in, which influences are strongest? Read from the
    effective connectome for that condition.
:func:`failure_correlates`
    Which pathways are more active when I fail than when I succeed? Two
    effective connectomes over the same anatomy, differenced.
:func:`does_it_influence`
    I believe mechanism A produces outcome B. Does A influence B at all, and at
    what grade of evidence?
:func:`what_would_change`
    If I disabled this, what would stop working? A lesion against a
    degree-matched control.

Every answer carries the grade of the evidence behind it and the sentence that
grade licenses. A predictive weight is not a cause, and this surface will not
return one worded as though it were, because the consumer is a self-model and a
self-model that overstates its own evidence is the failure mode the whole thing
exists to avoid.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .effective import EffectiveConnectome, Grade
from .types import ConnectomeSnapshot

logger = logging.getLogger("Aura.Connectome.Introspect")

__all__ = [
    "Answer",
    "dominant_circuit",
    "failure_correlates",
    "does_it_influence",
    "what_would_change",
]


@dataclass
class Answer:
    """A finding about her own machinery, and what it is allowed to claim."""

    question: str
    grade: Grade
    finding: str
    detail: dict[str, Any] = field(default_factory=dict)
    caveat: str = ""

    def as_json(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "grade": str(self.grade),
            "licenses": self.grade.licenses,
            "finding": self.finding,
            "caveat": self.caveat,
            "detail": self.detail,
        }

    def sentence(self) -> str:
        """The strongest sentence this answer supports, with its hedge attached."""
        tail = f" ({self.caveat})" if self.caveat else ""
        return f"{self.finding}{tail}"


def _name(snapshot: ConnectomeSnapshot, uid: str) -> str:
    unit = snapshot.units.get(uid)
    return unit.name if unit else uid


def dominant_circuit(
    snapshot: ConnectomeSnapshot,
    effective: EffectiveConnectome,
    *,
    limit: int = 8,
) -> Answer:
    """Which influences are strongest in the state this graph was measured in."""
    surviving = effective.surviving()
    if not surviving:
        return Answer(
            question="which circuit dominates in this state",
            grade=effective.grade,
            finding=(
                f"nothing in the {effective.condition} recording influences anything "
                "else more than a rotation of itself would"
            ),
            caveat=effective.skipped or "no edge survived its null",
        )
    ranked = sorted(surviving.values(), key=lambda edge: -edge.weight)[:limit]
    top = ranked[0]
    return Answer(
        question="which circuit dominates in this state",
        grade=effective.grade,
        finding=(
            f"in {effective.condition}, the strongest influence is "
            f"{_name(snapshot, top.pre)} on {_name(snapshot, top.post)}, "
            f"explaining {top.weight:.1%} of what the second does next"
        ),
        caveat="measured by prediction, not by intervention",
        detail={
            "condition": effective.condition,
            "edges_surviving": len(surviving),
            "top": [
                {
                    "from": _name(snapshot, edge.pre),
                    "to": _name(snapshot, edge.post),
                    "weight": round(edge.weight, 5),
                    "z": round(edge.z, 2),
                }
                for edge in ranked
            ],
        },
    )


def failure_correlates(
    snapshot: ConnectomeSnapshot,
    failing: EffectiveConnectome,
    succeeding: EffectiveConnectome,
    *,
    limit: int = 8,
) -> Answer:
    """Which pathways carry when things go wrong and not when they go right.

    Two effective graphs over the same anatomy, one from conditions that failed
    and one from conditions that did not. An edge strong in the first and absent
    from the second is a pathway her failures depend on, and it is a correlation
    until something disables it.
    """
    shared = set(failing.edges) & set(succeeding.edges)
    if len(shared) < 8:
        return Answer(
            question="which pathways my failures depend on",
            grade=Grade.PREDICTIVE,
            finding="too few edges were measured in both states to compare them",
            caveat=f"{len(shared)} edges in common",
        )
    gaps = sorted(
        (
            (pair, failing.edges[pair].weight - succeeding.edges[pair].weight)
            for pair in shared
            if failing.edges[pair].survives_null
        ),
        key=lambda item: -item[1],
    )[:limit]
    if not gaps or gaps[0][1] <= 0:
        return Answer(
            question="which pathways my failures depend on",
            grade=Grade.PREDICTIVE,
            finding="no pathway is more active when things go wrong than when they go right",
            caveat="over the edges measured in both states",
        )
    pair, gap = gaps[0]
    return Answer(
        question="which pathways my failures depend on",
        grade=Grade.PREDICTIVE,
        finding=(
            f"{_name(snapshot, pair[0])} influences {_name(snapshot, pair[1])} "
            f"{gap:.1%} more when things go wrong than when they go right"
        ),
        caveat="a correlation; nothing here disabled the pathway to check",
        detail={
            "failing": failing.condition,
            "succeeding": succeeding.condition,
            "edges_compared": len(shared),
            "widest": [
                {
                    "from": _name(snapshot, p[0]),
                    "to": _name(snapshot, p[1]),
                    "excess": round(g, 5),
                }
                for p, g in gaps
            ],
        },
    )


def does_it_influence(
    snapshot: ConnectomeSnapshot,
    effective: EffectiveConnectome,
    mechanism: Sequence[str],
    outcome: Sequence[str],
    *,
    belief: str = "",
) -> Answer:
    """Ask whether the machinery she believes did something has any influence on it.

    This is the question a self-model cannot ask itself without a connectome, and
    the one most worth asking: the mechanism a system reports as the cause of its
    own behaviour is the one it has a story about, not necessarily the one that
    moved.
    """
    mechanism_cells = set(mechanism)
    outcome_cells = set(outcome)
    relevant = [
        edge
        for (pre, post), edge in effective.edges.items()
        if pre in mechanism_cells and post in outcome_cells
    ]
    surviving = [edge for edge in relevant if edge.survives_null]
    preamble = f"{belief}: " if belief else ""
    if not relevant:
        return Answer(
            question="does this mechanism influence this outcome",
            grade=effective.grade,
            finding=(
                f"{preamble}no edge from that machinery to that outcome was measured, so "
                "the question has no answer from this recording"
            ),
            caveat="absence of a measurement is not absence of an influence",
            detail={"edges_measured": 0},
        )
    if not surviving:
        return Answer(
            question="does this mechanism influence this outcome",
            grade=effective.grade,
            finding=(
                f"{preamble}{len(relevant)} edges join that machinery to that outcome and "
                "none of them influences it more than a rotation of itself would"
            ),
            caveat="measured by prediction, not by intervention",
            detail={"edges_measured": len(relevant), "edges_surviving": 0},
        )
    strongest = max(surviving, key=lambda edge: edge.weight)
    return Answer(
        question="does this mechanism influence this outcome",
        grade=effective.grade,
        finding=(
            f"{preamble}{len(surviving)} of {len(relevant)} edges influence it, the "
            f"strongest being {_name(snapshot, strongest.pre)} on "
            f"{_name(snapshot, strongest.post)} at {strongest.weight:.1%}"
        ),
        caveat="measured by prediction, not by intervention",
        detail={
            "edges_measured": len(relevant),
            "edges_surviving": len(surviving),
            "strongest": strongest.as_json(),
        },
    )


def what_would_change(
    snapshot: ConnectomeSnapshot,
    cells: Sequence[str],
    *,
    null_samples: int = 8,
) -> Answer:
    """If this were disabled, what would stop working?

    A lesion against a control lesion of the same degree, so the answer is what
    *this* machinery carries rather than what any machinery of its size carries.
    This one is interventional — on the graph. It says what would be unreachable,
    not what she would stop being able to do.
    """
    from .lesion import measure_effect

    effect = measure_effect(snapshot, list(cells), null_samples=null_samples)
    payload = effect.as_json()
    if effect.excess_reach_loss > 0.01:
        finding = (
            f"disabling it would put {effect.reach_before - effect.reach_after} cells "
            f"out of reach, {effect.excess_reach_loss:.1%} more than removing machinery "
            "of the same size"
        )
    else:
        finding = (
            "disabling it would cost no more reach than removing any machinery of the "
            "same size"
        )
    return Answer(
        question="what would stop working without this",
        grade=Grade.MODEL,
        finding=finding,
        caveat="reachability in the graph, not behaviour",
        detail=payload,
    )
