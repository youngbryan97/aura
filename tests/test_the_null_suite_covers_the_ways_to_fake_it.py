"""Each null keeps something superficial and destroys one thing that matters.

A statistic that only ever runs on the thing it was written for cannot be
trusted, because there is no way to tell whether it detects integration or
detects complexity. Two of the ways to look integrated had no null at all.

A broker that sits outside K and remembers nearly all of its own past makes
every domain depend on every other one, and the system's memory is not in K —
so K's own future depends on a variable no reading of K contains. That is the
hardest case for causal closure and it must not read as a mind.

And ten domains of independent noise: same width, same decay, same noise,
nothing crossing. Every measure in the battery has to report nothing here, and
a measure that rewards dimensionality rather than integration will not.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.subject.graph import analyse_graph
from core.subject.nulls import ARCHITECTURES, architecture, toy_edges, toy_recording
from core.subject.state import DOMAINS


def test_both_new_nulls_are_in_the_suite() -> None:
    assert "hidden_broker" in ARCHITECTURES
    assert "independent" in ARCHITECTURES


def test_the_independent_system_has_no_coupling_to_find() -> None:
    """The sanity null. Nothing crosses, so nothing may be measured crossing."""
    system = architecture("independent", seed=5)
    assert all(not row for row in system.coupling.values()), "a domain reads another"
    assert system.hub_width == 0, "there is a broker in a system with no coupling"
    graph = analyse_graph(list(DOMAINS), toy_edges(system, trials=10, seed=5))
    assert not graph.one_component
    assert graph.connectivity == 0


def test_the_independent_system_still_moves() -> None:
    """A null that is simply dead proves nothing.

    Each domain has to have its own dynamics and its own noise, or "no edges"
    is a statement about a flat recording rather than about the absence of
    coupling.
    """
    recording = toy_recording(architecture("independent", seed=5), steps=800, seed=5)
    spread = [float(np.mean(recording.domain(key).std(axis=0))) for key in DOMAINS]
    assert min(spread) > 1e-6, "a domain of the independent null never moves"


def test_the_hidden_broker_keeps_the_memory_outside_the_core() -> None:
    """Its state is not one of the ten, and it remembers nearly all of its own
    past — which is the whole difference from the star's pure relay."""
    broker = architecture("hidden_broker", seed=5)
    star = architecture("star", seed=5)
    assert broker.hub_width > 0
    assert broker.hub_decay > star.hub_decay
    assert broker.hub_decay >= 0.9
    assert all(not row for row in broker.coupling.values()), (
        "a domain reads another directly, so not every path is through the broker"
    )


def test_the_independent_null_does_not_look_like_the_reference() -> None:
    reference = analyse_graph(
        list(DOMAINS), toy_edges(architecture("recurrent", seed=5), trials=10, seed=5)
    )
    assert reference.one_component and reference.connectivity >= 2

    graph = analyse_graph(
        list(DOMAINS), toy_edges(architecture("independent", seed=5), trials=10, seed=5)
    )
    passes = graph.one_component and graph.connectivity >= 2 and graph.every_node_reenters
    assert not passes


def test_the_hidden_broker_looks_exactly_like_a_mind_on_the_graph() -> None:
    """The finding this null exists to make.

    Every domain depends on every other one, so the graph comes back one
    component, vertex connectivity three, every node re-entering — the same
    answers the recurrent reference gives. No graph measure tells them apart,
    which is why the conjunction is a conjunction.
    """
    graph = analyse_graph(
        list(DOMAINS), toy_edges(architecture("hidden_broker", seed=5), trials=10, seed=5)
    )
    assert graph.one_component
    assert graph.connectivity >= 2
    assert graph.every_node_reenters


@pytest.mark.parametrize("name", ["hidden_broker", "hub", "star"])
def test_a_broker_outside_the_core_leaves_the_core_open(name: str) -> None:
    """What does tell them apart.

    K's future depends on a variable no reading of K contains. The shuffled arm
    keeps every broker column and destroys only its alignment in time, so the
    gain is information rather than the extra freedom a wider model brings.
    """
    from core.subject.closure import closure_gain
    from core.subject.nulls import toy_periphery

    system = architecture(name, seed=5)
    recording = toy_recording(system, steps=2000, seed=5)
    outside = toy_periphery(system, steps=2000, seed=5)
    assert outside.shape[1] > 0, f"the {name} null has no broker to read"
    report = closure_gain(
        recording,
        outside,
        tuple(f"broker.{index}" for index in range(outside.shape[1])),
        seed=5,
    )
    assert not report.closed, f"the {name} null's core reads as closed"
    assert report.leak > report.shuffled_leak


def test_the_reference_core_is_closed() -> None:
    """It has no broker, so there is nothing outside it to leak from."""
    from core.subject.closure import closure_gain
    from core.subject.nulls import toy_periphery

    system = architecture("recurrent", seed=5)
    outside = toy_periphery(system, steps=2000, seed=5)
    assert outside.shape[1] == 0
    report = closure_gain(
        toy_recording(system, steps=2000, seed=5), outside, (), seed=5
    )
    assert report.closed


def test_the_conjunction_asks_whether_the_null_core_is_closed() -> None:
    source = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools"
        / "run_subject_core.py"
    ).read_text()
    assert 'row.get("closed", True)' in source
    assert '"closed": closed' in source
