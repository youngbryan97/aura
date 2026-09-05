"""core/cognition/abstraction_lineage.py — every abstraction back to what it came from.

A concept, a rule, a procedure, a compressed representation: each is a claim
that some episodes had something in common. Aura can produce all four and none
of them keeps the episodes. So when an abstraction turns out to be wrong, there
is no way to ask which experiences supported it, and when it turns out to be
right there is no way to ask what would have to change for it to stop being.

The ledger is a directed graph from episodes upward. Two properties it
enforces, because an abstraction that lacks either is a guess with a lineage
attached:

* **A parent that was never observed cannot support a child.** Deriving from a
  derivation is fine; deriving from nothing is a root, and a root has to name
  the episodes directly.
* **A lineage cannot loop.** An abstraction that transitively supports itself
  has laundered a hypothesis into evidence, and :meth:`Lineage.link` refuses
  the edge that would close the cycle.

Because the sources are kept, :meth:`Lineage.would_survive` answers the
question a retraction needs: if these episodes turn out to be wrong, what else
falls? That is the difference between deleting one belief and finding out how
much of the building was resting on it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from core.runtime.lockdep import checked_lock

__all__ = ["Kind", "Node", "Lineage", "CycleRefused"]


class Kind(StrEnum):
    EPISODE = "episode"
    CONCEPT = "concept"
    RULE = "rule"
    PROCEDURE = "procedure"
    REPRESENTATION = "representation"
    NEURAL = "neural"


class CycleRefused(ValueError):
    """An abstraction would have supported itself."""


@dataclass
class Node:
    """One abstraction, or one episode at the bottom."""

    node_id: str
    kind: Kind
    parents: set[str] = field(default_factory=set)
    label: str = ""
    retracted: bool = False

    @property
    def is_root(self) -> bool:
        return self.kind is Kind.EPISODE


class Lineage:
    """Which episodes each abstraction rests on, and what falls if they go."""

    def __init__(self) -> None:
        self._lock = checked_lock("core.cognition.abstraction_lineage.Lineage", reentrant=True)
        self._nodes: dict[str, Node] = {}
        self._children: dict[str, set[str]] = {}

    def add(self, node_id: str, kind: Kind, *, label: str = "") -> Node:
        with self._lock:
            node = self._nodes.setdefault(node_id, Node(node_id=node_id, kind=kind, label=label))
            return node

    def link(self, child: str, parents: Sequence[str]) -> Node:
        """Record that ``child`` was derived from ``parents``."""
        with self._lock:
            node = self._nodes.get(child)
            if node is None:
                raise KeyError(f"{child!r} is not registered")
            for parent in parents:
                if parent not in self._nodes:
                    raise KeyError(
                        f"{child!r} claims support from {parent!r}, which was never "
                        "observed; a root must name its episodes directly"
                    )
                if parent == child or self._reaches_locked(parent, child):
                    raise CycleRefused(
                        f"{child!r} would support itself through {parent!r}; that "
                        "launders a hypothesis into evidence"
                    )
            node.parents |= set(parents)
            for parent in parents:
                self._children.setdefault(parent, set()).add(child)
            return node

    def _reaches_locked(self, start: str, target: str) -> bool:
        seen, frontier = set(), [start]
        while frontier:
            current = frontier.pop()
            if current == target:
                return True
            if current in seen:
                continue
            seen.add(current)
            node = self._nodes.get(current)
            if node:
                frontier.extend(node.parents)
        return False

    def episodes_behind(self, node_id: str) -> frozenset[str]:
        """Every episode this abstraction transitively rests on."""
        with self._lock:
            seen, frontier, episodes = set(), [node_id], set()
            while frontier:
                current = frontier.pop()
                if current in seen:
                    continue
                seen.add(current)
                node = self._nodes.get(current)
                if node is None:
                    continue
                if node.is_root:
                    episodes.add(current)
                frontier.extend(node.parents)
            return frozenset(episodes)

    def would_survive(self, retracted_episodes: Iterable[str]) -> dict[str, Any]:
        """If these episodes were wrong, what else falls.

        An abstraction falls when every episode behind it is retracted. One
        supported by three episodes of which one is retracted stands, weaker,
        and saying which is which is the whole use of keeping the lineage.
        """
        gone = set(retracted_episodes)
        with self._lock:
            nodes = [n for n in self._nodes.values() if not n.is_root]
        falls, weakened, untouched = [], [], []
        for node in nodes:
            episodes = self.episodes_behind(node.node_id)
            if not episodes:
                untouched.append(node.node_id)
            elif episodes <= gone:
                falls.append(node.node_id)
            elif episodes & gone:
                weakened.append(
                    {
                        "node": node.node_id,
                        "lost": len(episodes & gone),
                        "remaining": len(episodes - gone),
                    }
                )
            else:
                untouched.append(node.node_id)
        return {
            "retracted_episodes": sorted(gone),
            "falls": sorted(falls),
            "weakened": sorted(weakened, key=lambda row: row["node"]),
            "untouched": sorted(untouched),
        }

    def report(self) -> dict[str, Any]:
        with self._lock:
            nodes = list(self._nodes.values())
        by_kind: dict[str, int] = {}
        for node in nodes:
            by_kind[node.kind.value] = by_kind.get(node.kind.value, 0) + 1
        orphans = [
            n.node_id for n in nodes
            if not n.is_root and not self.episodes_behind(n.node_id)
        ]
        return {
            "nodes": len(nodes),
            "by_kind": dict(sorted(by_kind.items())),
            "abstractions_with_no_episode_behind_them": sorted(orphans),
            "all_traceable": not orphans,
        }
