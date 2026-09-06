"""Four graph stores, one shape, one id for what a node is about.

The stores stay: an AtomSpace with PLN and an attention economy is not a
worse entity index. What could not stay is a reference from one into another
being a string with no shared meaning, so that "the same person" in two of
them was two things and nothing could tell.
"""
from __future__ import annotations

import pytest

# Importing the store registers its adapter. That is the mechanism, and
# `which_stores_have_not_registered` is what makes forgetting it visible
# rather than silently reporting "two graphs, all consistent" for four.
import core.world.knowledge_graph  # noqa: F401
from core.knowledge.one_graph import (
    A_PLACEHOLDER,
    THE_PROMISES,
    WHERE_THE_ADAPTERS_LIVE,
    ALink,
    ANode,
    SemanticGraph,
    TheAtomSpaceAsAGraph,
    every_graph,
    references_that_lead_nowhere,
    what_a_graph_promises,
    which_stores_have_not_registered,
)
from core.knowledge.who_this_is import an_id_for
from core.world.knowledge_graph import TheKnowledgeGraphAsAGraph

THE_ADAPTERS = (
    ("knowledge_graph", TheKnowledgeGraphAsAGraph),
    ("atomspace", TheAtomSpaceAsAGraph),
)


@pytest.mark.parametrize("name,make", THE_ADAPTERS)
def test_every_store_satisfies_the_shape(name, make):
    assert isinstance(make(), SemanticGraph), name


@pytest.mark.parametrize("name,make", THE_ADAPTERS)
def test_every_store_keeps_every_promise(name, make):
    """The conformance suite. The same list for every store, unchanged."""
    kept = what_a_graph_promises(make, called=name)
    broken = {promise: why for promise, why in kept.items() if why != "kept"}
    assert not broken, f"{name} broke: {broken}"
    assert set(kept) == set(THE_PROMISES)


@pytest.mark.parametrize("name,make", THE_ADAPTERS)
def test_a_fresh_graph_per_promise(name, make):
    """Shared state makes a failure ambiguous.

    Did the second promise fail, or did the first leave something behind?
    """
    graph = make()
    graph.put_node(ANode.of("person", "Left Behind"))
    kept = what_a_graph_promises(make, called=name)
    assert all(why == "kept" for why in kept.values())


# --------------------------------------------------------- the shared id


def test_two_stores_give_one_thing_the_same_id():
    """The finding. Before, they gave it two."""
    graphs = every_graph()
    node = ANode.of("person", "Bryan")
    for graph in graphs.values():
        graph.put_node(node)
    ids = {
        graph.node(node.node_id).node_id
        for graph in graphs.values()
    }
    assert len(ids) == 1


def test_the_id_survives_a_different_spelling():
    assert ANode.of("person", "Bryan").node_id == ANode.of("person", " BRYAN ").node_id


def test_the_kind_is_part_of_the_id():
    """A person called Aura and a project called Aura are two things."""
    assert an_id_for("person", "Aura") != an_id_for("project", "Aura")


# ----------------------------------------------- referential integrity


def test_a_reference_across_stores_resolves_when_both_are_looked_at():
    graphs = every_graph()
    bryan = ANode.of("person", "Bryan")
    aura = ANode.of("project", "Aura")
    graphs["knowledge_graph"].put_node(bryan)
    graphs["atomspace"].put_node(aura)
    graphs["knowledge_graph"].put_link(ALink("builds", bryan.node_id, aura.node_id))

    assert references_that_lead_nowhere(graphs) == []


def test_the_same_reference_leads_nowhere_when_only_one_store_is_looked_at():
    """The case a per-store check cannot see, which is why the set matters."""
    graphs = every_graph()
    bryan = ANode.of("person", "Bryan")
    aura = ANode.of("project", "Aura")
    graphs["knowledge_graph"].put_node(bryan)
    graphs["atomspace"].put_node(aura)
    graphs["knowledge_graph"].put_link(ALink("builds", bryan.node_id, aura.node_id))

    alone = references_that_lead_nowhere({"knowledge_graph": graphs["knowledge_graph"]})
    assert len(alone) == 1
    assert alone[0]["id"] == aura.node_id
    assert alone[0]["why"] == "a placeholder"


def test_a_placeholder_is_what_dangling_looks_like_here():
    """Neither store refuses an edge, which is right and also hides this."""
    graph = TheKnowledgeGraphAsAGraph()
    bryan = ANode.of("person", "Bryan")
    graph.put_node(bryan)
    graph.put_link(ALink("knows", bryan.node_id, an_id_for("person", "Nobody")))

    invented = graph.node(an_id_for("person", "Nobody"))
    assert invented is not None
    assert invented.kind == A_PLACEHOLDER


def test_a_graph_with_nothing_in_it_reports_nothing_wrong():
    assert references_that_lead_nowhere(every_graph()) == []


def test_the_check_reports_each_bad_end_once():
    """A link read from both ends must not be counted twice."""
    graph = TheKnowledgeGraphAsAGraph()
    bryan = ANode.of("person", "Bryan")
    graph.put_node(bryan)
    missing = an_id_for("person", "Nobody")
    graph.put_link(ALink("knows", bryan.node_id, missing))
    graph.put_link(ALink("knows", bryan.node_id, missing))

    found = references_that_lead_nowhere({"kg": graph})
    assert len([one for one in found if one["id"] == missing]) == 1


# ------------------------------------------------------------ the wiring


def test_every_declared_store_registered_itself():
    """A store whose module nothing imports never registers.

    A registry that could not say so would report two graphs, all consistent,
    for a system running four.
    """
    assert which_stores_have_not_registered() == []
    assert set(WHERE_THE_ADAPTERS_LIVE) == {"atomspace", "knowledge_graph"}


def test_the_live_stores_are_what_gets_checked():
    """An integrity check over fresh graphs measures nothing."""
    assert every_graph(live=True) == {} or all(
        isinstance(one, SemanticGraph) for one in every_graph(live=True).values()
    )


def test_the_check_is_in_the_health_report():
    from core.runtime.health_contract import runtime_health_report

    block = runtime_health_report()["integrity"]["one_graph"]
    assert set(block) >= {"stores", "nodes", "how_many_lead_nowhere"}


def test_a_store_that_breaks_a_promise_is_named_and_not_only_counted():
    class Forgetful:
        def put_node(self, node):
            return node.node_id

        def put_link(self, link):
            pass

        def node(self, node_id):
            return None

        def out_of(self, node_id):
            return []

        def into(self, node_id):
            return []

        def all_nodes(self):
            return []

    kept = what_a_graph_promises(Forgetful, called="forgetful")
    broken = [promise for promise, why in kept.items() if why != "kept"]
    assert "a node put can be read back" in broken
    assert "broken:" in kept["a node put can be read back"]
