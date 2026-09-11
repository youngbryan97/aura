"""A null passes or fails the conjunction, not one line of it.

`beats_every_null` says "the whole battery separates the system from every
matched null" and the test behind it asked whether irreducibility cleared its
bar. That is not the conjunction, and it gave the wrong answer: the hub null —
a broker that carries its own state across steps — scores 0.070 against a bar
of 0.05 and was counted as passing, though it fails the graph on vertex
connectivity exactly as it was designed to. A stateful broker is genuinely hard
to partition. The battery tells it from a mind on the shape of its graph, and
the criterion has to ask what the battery asks.

The other half is the positive control. A reference architecture that is
recurrent by construction has to pass everything the nulls fail, or the
instrument is only capable of returning no and nobody can tell a hard organism
from a blunt instrument. It could not: a flat displacement of 0.5 moved the
reference's workspace domain by 0.29 standard deviations and its self-state by
0.52, so some domains were poked twice as hard as others and the reference's
own declared wiring did not come back out of the measurement.
"""

from __future__ import annotations

import pytest

from core.subject.graph import analyse_graph
from core.subject.nulls import ARCHITECTURES, architecture, toy_doses, toy_edges
from core.subject.state import DOMAINS


def _graph(name: str, seed: int = 7):
    system = architecture(name, seed=seed)
    return analyse_graph(list(DOMAINS), toy_edges(system, trials=12, seed=seed))


def test_the_reference_architecture_shows_its_own_wiring() -> None:
    """The positive control, and the whole point of having one."""
    graph = _graph("recurrent")
    assert graph.one_component, "the reference recurrent system is not one component"
    assert graph.connectivity >= 2, (
        f"the reference's vertex connectivity is {graph.connectivity}"
    )
    assert graph.every_node_reenters


#: The one null the graph cannot tell from a mind. Every path runs through a
#: broker that is not one of the ten domains, so every domain depends on every
#: other one and the graph comes back with the reference's own answers. What
#: separates them is causal closure, which is why the conjunction is a
#: conjunction — see `test_the_null_suite_covers_the_ways_to_fake_it.py`.
GRAPH_CANNOT_SEPARATE: frozenset[str] = frozenset({"hidden_broker"})


@pytest.mark.parametrize(
    "name",
    [n for n in ARCHITECTURES if n != "recurrent" and n not in GRAPH_CANNOT_SEPARATE],
)
def test_no_null_passes_the_graph_the_reference_passes(name: str) -> None:
    """Each null keeps something superficial and loses one thing that matters.

    None of these may look like the reference on the graph, or the graph is not
    measuring the thing the battery says it measures.
    """
    graph = _graph(name)
    passes = (
        graph.one_component
        and graph.connectivity >= 2
        and graph.every_node_reenters
    )
    assert not passes, f"the {name} null is indistinguishable from the reference"


def test_every_domain_is_displaced_to_the_same_dose() -> None:
    """A displacement is not a dose until the thing displaced has moved.

    A flat delta is not a matched intervention: what would differ between two
    architectures is then how hard each was hit as much as what escaped.
    """
    system = architecture("recurrent", seed=7)
    deltas, _ = toy_doses(system, seed=7)
    assert set(deltas) == set(DOMAINS)
    assert all(value > 0.0 for value in deltas.values())
    # They are not all the same, because the domains are not equally stiff.
    assert max(deltas.values()) > min(deltas.values()) * 1.1


def test_a_stateful_broker_is_hard_to_partition_and_still_not_a_mind() -> None:
    """Why the conjunction is the right question.

    The hub is the null the battery is least able to tell apart on phi alone,
    and the one it tells apart most clearly on the graph.
    """
    graph = _graph("hub")
    assert not graph.one_component or graph.connectivity < 2
