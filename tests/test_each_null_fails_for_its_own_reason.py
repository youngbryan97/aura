"""A null that fails for the wrong reason is not a control.

Each architecture in the suite keeps something superficial and loses one thing
that matters, and the line it loses is the line it was built to test. A null
that happens to fail on something else has stopped testing what its name says,
and the criterion it was supposed to defend is undefended while the suite still
reports green.

Two of these were caught that way while they were being written.

`low_rank` was meant to be recurrent and minimally differentiated. Built with a
fresh pair of vectors per edge it had ten independent one-dimensional channels
and an effective dimension of 19.6 — a system with a repertoire, failing on
nothing it was written for. Sharing one read and one write direction across
every edge makes the whole network rank one, and it now sits at 1.26 with 89%
of the variance in one component.

`all_to_all` was meant to be a broadcast. Built with a random matrix per pair
it was a dense random recurrent network, effective dimension 3.4, and it passed
differentiation. A broadcast means everyone receives the same thing, and it now
does.
"""

from __future__ import annotations

import pytest

from core.subject.battery import THRESHOLDS
from core.subject.differentiation import effective_dimension
from core.subject.graph import analyse_graph
from core.subject.irreducibility import phi_do
from core.subject.nulls import architecture, toy_edges, toy_recording
from core.subject.state import DOMAINS

pytestmark = pytest.mark.unit

SEED = 7
STEPS = 900
TRIALS = 6


def _measure(name: str) -> dict[str, object]:
    system = architecture(name, seed=SEED)
    recording = toy_recording(system, steps=STEPS, seed=SEED)
    graph = analyse_graph(list(DOMAINS), toy_edges(system, trials=TRIALS, seed=SEED))
    spectrum = effective_dimension(recording)
    return {
        "phi": float(phi_do(recording).phi),
        "one_component": bool(graph.one_component),
        "connectivity": int(graph.connectivity),
        "reentry": bool(graph.every_node_reenters),
        "top_share": float(spectrum.top_share),
        "ratio": float(spectrum.normalised),
    }


def _fails_on(row: dict[str, object]) -> set[str]:
    """Which lines of the conjunction this system does not clear.

    Differentiation is read on its two scale-free halves. `d_eff >= 3` is a bar
    on a number that grows with the width of the system, and a toy of forty
    columns is not comparable to an organism of two hundred and eight on it —
    the recurrent reference scores 2.93 and would fail the line it exists to
    pass.
    """
    out: set[str] = set()
    if row["phi"] <= THRESHOLDS["phi_do"]:
        out.add("irreducibility")
    if not row["one_component"]:
        out.add("one_component")
    if row["connectivity"] < THRESHOLDS["vertex_connectivity"]:
        out.add("connectivity")
    if not row["reentry"]:
        out.add("reentry")
    if (
        row["top_share"] >= THRESHOLDS["component_share"]
        or row["ratio"] >= THRESHOLDS["d_eff_normalised"]
    ):
        out.add("differentiation")
    return out


@pytest.fixture(scope="module")
def measured() -> dict[str, dict[str, object]]:
    names = (
        "recurrent", "ring", "all_to_all", "low_rank",
        "common_driver", "high_dimensional_independent",
    )
    return {name: _measure(name) for name in names}


def test_the_reference_clears_every_line(measured) -> None:
    """Without this the instrument can only say no, and nobody can tell a hard
    organism from a blunt instrument."""
    assert _fails_on(measured["recurrent"]) == set()


#: What each null is for. The key is the line it must lose; anything else it
#: also loses is fine, but losing this one is the whole reason it is here.
MUST_FAIL_ON: dict[str, str] = {
    # Strongly connected and every node re-enters, and removing any single node
    # splits it. Connectivity is the line.
    "ring": "connectivity",
    # Genuinely integrated and genuinely redundant. Differentiation is the line.
    "all_to_all": "differentiation",
    # Recurrent and riding on one latent. Differentiation is the line.
    "low_rank": "differentiation",
    # Every pair moves together and no pair moves the other, so the
    # interventional graph is empty where an observational one would be full.
    "common_driver": "one_component",
    # The highest effective dimension of anything in the suite, and nothing
    # crossing between the domains.
    "high_dimensional_independent": "differentiation",
}


@pytest.mark.parametrize("name,line", sorted(MUST_FAIL_ON.items()))
def test_each_null_loses_the_line_it_was_built_to_lose(measured, name: str, line: str) -> None:
    lost = _fails_on(measured[name])
    assert line in lost, (
        f"the {name} null was built to fail on {line} and failed on "
        f"{sorted(lost) or 'nothing'} instead, so it is not testing that line"
    )


@pytest.mark.parametrize("name", sorted(MUST_FAIL_ON))
def test_no_null_clears_the_whole_conjunction(measured, name: str) -> None:
    assert _fails_on(measured[name]), f"the {name} null passes the whole battery"


def test_a_broadcast_is_integrated_and_that_is_not_the_objection(measured) -> None:
    """all_to_all scores above the irreducibility bar, and should.

    Its graph is complete and its irreducibility is real. What is wrong with it
    is that every domain hears the same signal, and differentiation is the only
    line that says so.
    """
    row = measured["all_to_all"]
    assert row["phi"] > THRESHOLDS["phi_do"]
    assert row["one_component"] and row["connectivity"] >= 2
    assert row["top_share"] >= THRESHOLDS["component_share"]


def test_the_two_degenerate_nulls_really_are_degenerate(measured) -> None:
    """The defect this file was written after: a null that had a repertoire."""
    for name in ("all_to_all", "low_rank"):
        assert measured[name]["top_share"] > 0.5, (
            f"{name} spreads its variance over more than one component, so it is "
            "a system with a repertoire rather than the degenerate control it claims"
        )


def test_the_independent_null_scores_highest_on_the_one_sided_reading(measured) -> None:
    """Why the differentiation bar is applied as an upper bound.

    Ten domains of independent noise have the most effective dimensions of
    anything here. A criterion that rewarded that number would be passed by the
    least mind-like system in the suite.
    """
    assert measured["high_dimensional_independent"]["ratio"] > measured["recurrent"]["ratio"]
