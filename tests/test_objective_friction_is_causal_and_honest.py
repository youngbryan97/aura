"""A grand name over a broken controller, writing to nobody.

`core/brain/autopoiesis.py` described itself as a "self-creating topology"
that performs "mitosis", "apoptosis" and "spontaneous generation of a new
pathway". It is a capped list of strings with two floats each. That gap
between vocabulary and algorithm is the criticism, and it turned out to be
hiding three defects rather than merely overselling one:

1. Pruning was UNREACHABLE. `_apoptosis` fired on `friction < 0.0`, but
   friction only accumulated (`friction *= 0.9; friction += dissonance`)
   and both live call sites pass positive values (0.05 and 0.45). Nothing
   was ever retired for obsolescence.
2. Splitting made DUPLICATES. Every split appended a node literally named
   `Nuance_of_<concept>`, so forty observations of one failing objective
   produced twenty entries with two distinct names — measured, not assumed.
3. Nothing READ it. Two writers in `cognitive_engine`, zero readers in the
   entire codebase. The accumulated friction could not influence any
   output, which makes the signal unmeasurable rather than just unused —
   the half-wired shape this repo keeps finding.

These tests hold the rewrite to the claim it now makes.
"""
from __future__ import annotations

from pathlib import Path

from core.brain.autopoiesis import (
    MAX_NODES,
    MAX_REFINEMENT_DEPTH,
    OBSOLETE_FRICTION,
    OBSOLETE_WEIGHT,
    PRESSURE_THRESHOLD,
    AutopoieticGraph,
    ObjectiveFrictionGraph,
)

ROOT = Path(__file__).resolve().parents[1]


# ───────────────────────────────────────── 1: retirement is reachable


def test_pressure_can_be_relieved_so_retirement_can_happen():
    """The old branch could never execute at all."""
    graph = ObjectiveFrictionGraph()
    graph.experience_friction("chore", 0.0)  # creates the entry, weight 0.1
    node = graph.nodes[0]
    node.weight = OBSOLETE_WEIGHT / 2
    node.friction = OBSOLETE_FRICTION / 2

    # A negative observation is relief. Without a relief path the obsolete
    # branch is decoration.
    graph.experience_friction("chore", -0.01)

    assert graph.nodes == [], "an inert key was never retired"
    assert graph.pruned == 1


def test_accumulating_friction_never_retires_an_active_key():
    """Retirement must not fire on a key that is under pressure."""
    graph = ObjectiveFrictionGraph()
    graph.experience_friction("hard", 0.45)
    for _ in range(5):
        graph.experience_friction("hard", 0.45)

    assert graph.nodes, "an actively failing objective was dropped"


# ───────────────────────────────────────── 2: splitting stops duplicating


def test_repeated_failure_does_not_produce_identical_entries():
    """The measured defect: 40 observations gave 20 nodes, 2 distinct names."""
    graph = ObjectiveFrictionGraph()
    for _ in range(40):
        graph.experience_friction("stuck", 0.45)

    names = [n.concept for n in graph.nodes]

    assert len(names) == len(set(names)), f"duplicate keys: {names}"


def test_refinement_is_depth_bounded():
    """Unbounded refinement is the same defect with unique names."""
    graph = ObjectiveFrictionGraph()
    for _ in range(200):
        graph.experience_friction("stuck", 0.45)

    depths = [n.depth for n in graph.nodes]

    assert max(depths) <= MAX_REFINEMENT_DEPTH
    assert len(graph.nodes) <= MAX_REFINEMENT_DEPTH + 1, (
        f"one objective produced {len(graph.nodes)} tracked keys"
    )


def test_refinement_still_happens():
    """The bound must not work by disabling the mechanism."""
    graph = ObjectiveFrictionGraph()
    for _ in range(10):
        graph.experience_friction("stuck", 0.45)

    assert graph.splits > 0
    assert any(n.depth > 0 for n in graph.nodes)


# ───────────────────────────────────────── 3: there is now a reader


def test_friction_can_be_read_back():
    """The whole point. Two writers and no reader is not a signal."""
    graph = ObjectiveFrictionGraph()
    graph.experience_friction("flaky", 0.45)
    graph.experience_friction("flaky", 0.45)

    assert graph.friction_for("flaky") > 0.0


def test_an_unknown_objective_reads_as_no_pressure():
    graph = ObjectiveFrictionGraph()

    assert graph.friction_for("never seen") == 0.0
    assert graph.is_under_pressure("never seen") is False


def test_sustained_failure_registers_as_pressure():
    graph = ObjectiveFrictionGraph()
    graph.experience_friction("flaky", 0.45)
    graph.experience_friction("flaky", 0.45)

    assert graph.friction_for("flaky") >= PRESSURE_THRESHOLD
    assert graph.is_under_pressure("flaky") is True


def test_a_succeeding_objective_does_not_register_as_pressure():
    """0.05 is what the success path writes. It must not look like failure."""
    graph = ObjectiveFrictionGraph()
    for _ in range(20):
        graph.experience_friction("fine", 0.05)

    assert graph.is_under_pressure("fine") is False


def test_the_report_names_what_is_failing():
    graph = ObjectiveFrictionGraph()
    for _ in range(3):
        graph.experience_friction("failing objective", 0.45)
    graph.experience_friction("ok objective", 0.05)

    report = graph.pressure_report()

    assert report["under_pressure"] >= 1
    assert report["top"], "the report names nothing"
    assert report["top"][0]["concept"].startswith("failing objective")


def test_the_report_is_bounded():
    """It is attached to telemetry; an unbounded list is its own incident."""
    graph = ObjectiveFrictionGraph()
    for i in range(50):
        graph.experience_friction(f"objective-{i}", 0.45)
        graph.experience_friction(f"objective-{i}", 0.45)

    assert len(graph.pressure_report(limit=5)["top"]) <= 5


def test_the_engine_actually_reads_it():
    """A reader nothing calls is the same defect one layer up."""
    from tests.source_contract import family_text_at

    # With the modules lifted out of it: the read is in cognitive_engine_thinking_loop.
    source = family_text_at(ROOT / "core" / "brain" / "cognitive_engine.py")

    assert "self.autopoiesis.is_under_pressure(" in source, (
        "cognitive_engine still only writes to the friction graph"
    )
    assert "pressure_report()" in source
    pressure_block = source[source.index("if self.autopoiesis.is_under_pressure(") :]
    pressure_block = pressure_block[: pressure_block.index(")\n        self._learn_spiking")]
    assert "enforce_failure_policy=False" in pressure_block, (
        "diagnostic friction can still invoke CognitiveEngine's fail-closed policy"
    )


# ───────────────────────────────────────── the claim matches the code


def test_the_module_no_longer_claims_to_be_self_creating():
    """The criticism was about vocabulary describing a process not performed."""
    source = (ROOT / "core" / "brain" / "autopoiesis.py").read_text("utf-8")

    for overclaim in (
        "Self-creating topology",
        "Spontaneous generation",
        "Cell death",
    ):
        assert overclaim not in source, f"still claims {overclaim!r}"


def test_the_old_name_still_works_for_live_callers():
    """Renaming a class out from under a caller is a separate change."""
    assert AutopoieticGraph is ObjectiveFrictionGraph
    assert AutopoieticGraph().friction_for("x") == 0.0


# ───────────────────────────────────────── bounds still hold


def test_the_list_stays_capped():
    graph = ObjectiveFrictionGraph()
    for i in range(MAX_NODES + 120):
        graph.experience_friction(f"k{i}", 0.1)

    assert len(graph.nodes) <= MAX_NODES


def test_capacity_pruning_keeps_the_heaviest_keys():
    graph = ObjectiveFrictionGraph()
    for i in range(MAX_NODES + 50):
        graph.experience_friction(f"k{i}", 0.1)
    graph.nodes[-1].weight = 1.0
    heavy = graph.nodes[-1].concept
    for i in range(60):
        graph.experience_friction(f"extra{i}", 0.1)

    assert any(n.concept == heavy for n in graph.nodes)


def test_the_index_does_not_leak_after_pruning():
    """A stale index would report friction for a key that no longer exists."""
    graph = ObjectiveFrictionGraph()
    for i in range(MAX_NODES + 100):
        graph.experience_friction(f"k{i}", 0.1)

    tracked = {n.concept for n in graph.nodes}
    indexed = set(graph._by_concept)

    assert indexed == tracked


def test_an_empty_concept_is_ignored():
    graph = ObjectiveFrictionGraph()
    graph.experience_friction("", 0.45)
    graph.experience_friction("   ", 0.45)

    assert graph.nodes == []
