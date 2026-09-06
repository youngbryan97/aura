"""One shape for a graph, and one id for what its nodes are about.

Aura has four graph-shaped stores and they do not agree about what a node is.
The AtomSpace has typed Atoms with truth and attention values. The knowledge
graph has ``node_id`` strings and typed edges. The mycelial network has bare
strings for memories and skills. The entity memory has content-addressed ids.

Keeping four is right — an AtomSpace with PLN and an attention economy is not
a worse entity index, it is a different thing. What is not right is that a
reference from one to another is a string with no shared meaning, so "the same
person" in two of them is two things, and nothing can tell.

So this is an interface, not a replacement. A store keeps everything it has
and grows a small adapter; ``an_id_for`` from ``who_this_is`` mints the id, so
a node about Bryan has the same id in all four. Then a reference across stores
can be checked, and ``references_that_lead_nowhere`` is the check.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from core.knowledge.who_this_is import an_id_for, the_same

logger = logging.getLogger("Aura.OneGraph")

__all__ = [
    "A_PLACEHOLDER",
    "ALink",
    "ANode",
    "SemanticGraph",
    "TheAtomSpaceAsAGraph",
    "THE_ADAPTERS",
    "WHERE_THE_ADAPTERS_LIVE",
    "a_store_that_is_a_graph",
    "every_graph",
    "references_that_lead_nowhere",
    "what_a_graph_promises",
    "which_stores_have_not_registered",
]


@dataclass(frozen=True)
class ANode:
    """One thing a graph knows about, under the id every store agrees on."""

    node_id: str
    kind: str
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def of(cls, kind: str, name: str, **attributes: Any) -> "ANode":
        return cls(an_id_for(kind, name), str(kind), str(name), dict(attributes))


@dataclass(frozen=True)
class ALink:
    """A relation between two nodes, by their canonical ids."""

    kind: str
    source_id: str
    target_id: str
    attributes: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SemanticGraph(Protocol):
    """What every graph-shaped store can be asked, whatever else it does."""

    def put_node(self, node: ANode) -> str:
        ...

    def put_link(self, link: ALink) -> None:
        ...

    def node(self, node_id: str) -> ANode | None:
        ...

    def out_of(self, node_id: str) -> list[ALink]:
        ...

    def into(self, node_id: str) -> list[ALink]:
        ...

    def all_nodes(self) -> list[ANode]:
        ...


#: Every store that has grown an adapter, by service name. A store registers
#: itself here rather than being imported: the knowledge graph lives in
#: ``core.world``, which this package may not reach, and an interface that
#: forced the layering open to describe itself would be describing a different
#: system.
THE_ADAPTERS: dict[str, Any] = {}

#: Where each adapter lives, as a string, so this file can say what should
#: have registered without importing it. A store whose module is never
#: imported never registers, and a registry that could not tell you that
#: would report "two graphs, all consistent" for a system running four.
WHERE_THE_ADAPTERS_LIVE: dict[str, str] = {
    "atomspace": "core.knowledge.one_graph",
    "knowledge_graph": "core.world.knowledge_graph",
}


def which_stores_have_not_registered() -> list[str]:
    """Declared adapters whose module nothing has imported yet."""
    return sorted(set(WHERE_THE_ADAPTERS_LIVE) - set(THE_ADAPTERS))


def a_store_that_is_a_graph(name: str) -> Any:
    """Register a store's adapter under its service name.

    ``adapt(live=...)`` returns a SemanticGraph over the live instance, or
    over a fresh one when ``live`` is false.
    """

    def keep(adapt: Any) -> Any:
        THE_ADAPTERS[name] = adapt
        return adapt

    return keep


#: What a graph must do to be one, in the words a failure should use. Every
#: adapter runs the same list; a store that passes its own tests and not these
#: is a store nothing can safely reference.
THE_PROMISES: tuple[str, ...] = (
    "a node put can be read back",
    "the same kind and name give the same id",
    "a link is found from both ends",
    "a link to a node that was never put creates it rather than dangling",
    "reading an unknown id gives nothing rather than raising",
    "a node put twice is one node",
)


def what_a_graph_promises(
    make: Any, *, called: str = ""
) -> dict[str, str]:
    """Run the promises against a factory. A fresh graph for each.

    Shared state between promises makes a failure ambiguous — did the second
    fail, or did the first leave something behind?
    """
    name = called or getattr(make, "__name__", "a graph")
    kept: dict[str, str] = {}

    def _try(promise: str, check: Any) -> None:
        try:
            check(make())
            kept[promise] = "kept"
        except AssertionError as exc:
            kept[promise] = f"broken: {exc}"
        except Exception as exc:  # noqa: BLE001 — a raise is a broken promise
            kept[promise] = f"broken: {exc!r}"

    def _read_back(graph: Any) -> None:
        node = ANode.of("person", "Bryan")
        graph.put_node(node)
        found = graph.node(node.node_id)
        assert found is not None, "put a node, read back nothing"
        assert found.name == "Bryan", f"read back {found.name!r}"

    def _same_id(graph: Any) -> None:
        one = ANode.of("person", "Bryan")
        other = ANode.of("person", "  bryan ")
        assert one.node_id == other.node_id, "two ids for one name"

    def _both_ends(graph: Any) -> None:
        a, b = ANode.of("person", "Bryan"), ANode.of("project", "Aura")
        graph.put_node(a)
        graph.put_node(b)
        graph.put_link(ALink("builds", a.node_id, b.node_id))
        assert [one.target_id for one in graph.out_of(a.node_id)] == [b.node_id]
        assert [one.source_id for one in graph.into(b.node_id)] == [a.node_id]

    def _no_dangling(graph: Any) -> None:
        a = ANode.of("person", "Bryan")
        graph.put_node(a)
        graph.put_link(ALink("knows", a.node_id, an_id_for("person", "Nobody")))
        assert graph.node(an_id_for("person", "Nobody")) is not None, (
            "a link to an absent node left it dangling"
        )

    def _unknown_is_nothing(graph: Any) -> None:
        assert graph.node("no-such-id") is None
        assert graph.out_of("no-such-id") == []
        assert graph.into("no-such-id") == []

    def _twice_is_once(graph: Any) -> None:
        node = ANode.of("person", "Bryan")
        graph.put_node(node)
        graph.put_node(ANode.of("person", "BRYAN"))
        assert len(graph.all_nodes()) == 1, (
            f"one thing under two spellings became {len(graph.all_nodes())} nodes"
        )

    _try(THE_PROMISES[0], _read_back)
    _try(THE_PROMISES[1], _same_id)
    _try(THE_PROMISES[2], _both_ends)
    _try(THE_PROMISES[3], _no_dangling)
    _try(THE_PROMISES[4], _unknown_is_nothing)
    _try(THE_PROMISES[5], _twice_is_once)
    logger.debug("%s kept %d of %d promises", name, sum(
        1 for one in kept.values() if one == "kept"), len(THE_PROMISES))
    return kept


# ------------------------------------------------------------- adapters


@a_store_that_is_a_graph("atomspace")
def _the_atomspace(*, live: bool = False) -> "TheAtomSpaceAsAGraph":
    inner = None
    if live:
        from core.container import ServiceContainer
        from core.service_names import ServiceNames

        inner = ServiceContainer.get(ServiceNames.ATOMSPACE, default=None)
        if inner is None:
            raise RuntimeError("no live atomspace")
    return TheAtomSpaceAsAGraph(inner)


class TheAtomSpaceAsAGraph:
    """core/knowledge/atomspace.py, under the shared shape.

    An Atom carries truth and attention that this interface does not mention,
    and that is the point of keeping the store: the interface is what other
    systems may rely on, not everything the store can do.
    """

    def __init__(self, inner: Any = None) -> None:
        if inner is None:
            from core.knowledge.atomspace import AtomSpace

            inner = AtomSpace()
        self._inner = inner
        self._by_id: dict[str, ANode] = {}
        self._links: list[ALink] = []

    def put_node(self, node: ANode) -> str:
        from core.knowledge.atomspace import Node as AnAtomNode

        self._inner.add(AnAtomNode(node.kind.upper() + "_NODE", node.name))
        self._by_id[node.node_id] = node
        return node.node_id

    def put_link(self, link: ALink) -> None:
        for end in (link.source_id, link.target_id):
            if end not in self._by_id:
                self.put_node(ANode(end, "unknown", end))
        self._links.append(link)

    def node(self, node_id: str) -> ANode | None:
        return self._by_id.get(the_same(node_id))

    def out_of(self, node_id: str) -> list[ALink]:
        here = the_same(node_id)
        return [one for one in self._links if one.source_id == here]

    def into(self, node_id: str) -> list[ALink]:
        here = the_same(node_id)
        return [one for one in self._links if one.target_id == here]

    def all_nodes(self) -> list[ANode]:
        return list(self._by_id.values())


# ------------------------------------------------- referential integrity


#: What a store calls a node it invented to hold up a link. Both of these
#: stores create one rather than refusing the link, which is the right choice
#: — refusing would lose the fact — and it is also what makes a dangling
#: reference invisible unless somebody looks for the placeholder.
A_PLACEHOLDER = "unknown"


def references_that_lead_nowhere(
    graphs: dict[str, SemanticGraph],
) -> list[dict[str, str]]:
    """Link ends that no graph in this set really holds.

    Two ways an end leads nowhere. It resolves to nothing at all, or it
    resolves only to a placeholder the store invented to keep the link — which
    is what dangling looks like in a store that will not refuse an edge.

    Checked across the whole set rather than per store: a reference from the
    knowledge graph into the AtomSpace is exactly the case a per-store check
    cannot see, and the reason a shared id layer is worth having.
    """
    real: set[str] = set()
    for graph in graphs.values():
        for node in graph.all_nodes():
            if node is not None and node.kind != A_PLACEHOLDER:
                real.add(the_same(node.node_id))

    nowhere: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for where, graph in sorted(graphs.items()):
        for node in graph.all_nodes():
            if node is None:
                continue
            for link in graph.out_of(node.node_id):
                for end, which in (
                    (link.source_id, "source"),
                    (link.target_id, "target"),
                ):
                    if the_same(end) in real:
                        continue
                    key = (where, link.kind, end)
                    if key in seen:
                        continue
                    seen.add(key)
                    held = graph.node(end)
                    nowhere.append(
                        {
                            "in": where,
                            "link": link.kind,
                            "end": which,
                            "id": end,
                            # The node's own kind, rather than a word for it.
                            # Every id here is absent from `real`, so a node
                            # that exists at all is one of the unknown kind —
                            # naming the constant says which and says it in
                            # the vocabulary the graph already uses.
                            "why": A_PLACEHOLDER if held is not None else "nothing",
                        }
                    )
    return nowhere


def every_graph(*, live: bool = False) -> dict[str, SemanticGraph]:
    """The stores, wrapped. New ones belong here or nothing can reference them.

    ``live`` wraps the instances the running system is using, through the
    service names, rather than empty ones. An integrity check over fresh
    graphs measures nothing, and that is the shape a check like this usually
    fails in.
    """
    wrapped: dict[str, SemanticGraph] = {}
    for name, adapt in sorted(THE_ADAPTERS.items()):
        try:
            wrapped[name] = adapt(live=live)
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("%s could not be wrapped as a graph: %s", name, exc)
    return {name: one for name, one in wrapped.items() if one is not None}


