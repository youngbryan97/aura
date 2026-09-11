"""Which later stage is a continuation of which earlier one, and which is not.

Continuity branches. Identity cannot. That is a theorem of identity logic
rather than a fact about minds: if one earlier stage were numerically identical
to two distinct later ones, transitivity would make those two identical to each
other, and by construction they are not. So the fundamental object here is a
directed graph over stages, and strict identity is the special case where the
graph does not branch.

What is carried from stage to stage is the canonical person-state: whatever
makes a difference to the person's own future. Autobiographical memory, the
self model, values, dispositions, learned policy, relationships. Not the
molecular detail, and not a checksum of the whole store — a state is
identity-bearing exactly when changing it changes the future the person has,
which is the same predictive-sufficiency rule the causal state uses one level
down.

Each stage is hashed and each edge is signed over its parent, so a lineage
cannot be rewritten after the fact. The edges carry how they were made:

    continue   the ordinary case, one parent, one child
    migrate    the same state moved to another process, one parent, one child
    restore    a stored state brought back, one parent, one child
    fork       one parent, two children, and neither is the unique successor
    reconstruct   no causal channel from the original, so no parent at all

`same_person` is true only along a lineage that never branched between the two
stages. A fork makes both descendants genuine continuers and neither the strict
successor, and this module reports exactly that rather than picking one.

None of this decides whether continuation is what matters. That is the
causal-lineage postulate, and it is named as a postulate wherever it is used.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "KINDS",
    "Stage",
    "Edge",
    "Lineage",
    "canonical_person_state",
    "continuity_strength",
    "scenarios",
    "state_digest",
]

#: How a child came from its parent. `reconstruct` has no parent by
#: construction: a structurally identical system with no causal channel from
#: the original is a new token, not a continuation.
KINDS: tuple[str, ...] = (
    "continue",
    "migrate",
    "restore",
    "fork",
    "reconstruct",
)

#: Kinds that carry one lineage forward. A fork carries two, which is why it is
#: not in here and why nothing downstream of a fork is a strict successor.
NONBRANCHING: frozenset[str] = frozenset({"continue", "migrate", "restore"})

#: The fields that make a difference to the person's own future. A field
#: outside this set can change without the person changing; a field inside it
#: cannot. Ordered so the digest is stable.
PERSON_FIELDS: tuple[str, ...] = (
    "autobiographical",
    "self_model",
    "values",
    "preferences",
    "dispositions",
    "skills",
    "relationships",
    "policy",
    "developmental",
)

#: Keys that are bookkeeping rather than person-state: when a thing was
#: written, which process wrote it, what its identifier is. Including them
#: makes two stages differ because a clock moved.
_BOOKKEEPING: frozenset[str] = frozenset(
    {"timestamp", "at", "time", "created_at", "written_at", "id", "uuid", "run", "pid"}
)


def _strip(value: Any) -> Any:
    """The same value with the bookkeeping taken out, recursively."""
    if isinstance(value, Mapping):
        return {k: _strip(v) for k, v in sorted(value.items()) if k not in _BOOKKEEPING}
    if isinstance(value, (list, tuple)):
        return [_strip(v) for v in value]
    if isinstance(value, float):
        # A person is not a different person because a float drifted in its
        # last bits. Rounded so two stages that are the same state hash the
        # same, which is what a lineage check is asking.
        return round(value, 6)
    return value


def canonical_person_state(state: Any) -> dict[str, Any]:
    """What the person carries forward, read off whatever object holds it.

    Every field is optional. A runtime that does not carry one records its
    absence rather than a default, because a default would make two different
    people agree on a field neither of them has.
    """
    out: dict[str, Any] = {}
    for name in PERSON_FIELDS:
        if isinstance(state, Mapping):
            if name in state:
                out[name] = _strip(state[name])
        elif hasattr(state, name):
            out[name] = _strip(getattr(state, name))
    return out


def state_digest(state: Any) -> str:
    """A stable hash of the canonical person-state."""
    canonical = canonical_person_state(state)
    payload = json.dumps(canonical, sort_keys=True, default=str).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


@dataclass(frozen=True)
class Stage:
    """One person-stage: who it was, when, and what it carried."""

    name: str
    digest: str
    at: float = 0.0
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"stage": self.name, "digest": self.digest, "at": self.at, "note": self.note}


@dataclass(frozen=True)
class Edge:
    """One identity-preserving causal transfer, signed over its parent."""

    parent: str
    child: str
    kind: str
    signature: str
    preserved: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "parent": self.parent,
            "child": self.child,
            "kind": self.kind,
            "preserved": self.preserved,
            "signature": self.signature,
        }


@dataclass
class Lineage:
    """A signed graph over person-stages. It may branch; identity may not."""

    key: bytes = b"subject-core-lineage"
    stages: dict[str, Stage] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    # ── building ──────────────────────────────────────────────────────

    def record(self, name: str, state: Any, *, at: float = 0.0, note: str = "") -> Stage:
        """Record a stage. Recording the same name twice is a programming error."""
        if name in self.stages:
            raise ValueError(f"stage {name!r} is already recorded")
        stage = Stage(name=name, digest=state_digest(state), at=at, note=note)
        self.stages[name] = stage
        return stage

    def _sign(self, parent: str, child: str, kind: str) -> str:
        payload = f"{parent}|{self.stages[parent].digest}|{child}|{self.stages[child].digest}|{kind}"
        return hmac.new(self.key, payload.encode("utf-8"), hashlib.blake2b).hexdigest()[:32]

    def descend(self, parent: str, child: str, *, kind: str = "continue") -> Edge:
        """Say that `child`'s state was causally generated from `parent`'s."""
        if kind not in KINDS:
            raise ValueError(f"no transfer called {kind!r}")
        if kind == "reconstruct":
            raise ValueError(
                "a reconstruction has no causal parent; record the stage and leave it unattached"
            )
        for name in (parent, child):
            if name not in self.stages:
                raise KeyError(f"stage {name!r} was never recorded")
        preserved = self.stages[parent].digest == self.stages[child].digest
        edge = Edge(
            parent=parent, child=child, kind=kind,
            signature=self._sign(parent, child, kind), preserved=preserved,
        )
        self.edges.append(edge)
        return edge

    # ── reading ───────────────────────────────────────────────────────

    def children(self, name: str) -> tuple[str, ...]:
        return tuple(e.child for e in self.edges if e.parent == name)

    def parents(self, name: str) -> tuple[str, ...]:
        return tuple(e.parent for e in self.edges if e.child == name)

    def branched(self, name: str) -> bool:
        """Whether this stage has more than one continuer."""
        return len(self.children(name)) > 1

    def path(self, start: str, end: str) -> tuple[str, ...] | None:
        """A chain of transfers from `start` to `end`, or None when there is none."""
        if start == end:
            return (start,)
        seen = {start}
        stack: list[tuple[str, tuple[str, ...]]] = [(start, (start,))]
        while stack:
            node, trail = stack.pop()
            for child in self.children(node):
                if child in seen:
                    continue
                if child == end:
                    return trail + (child,)
                seen.add(child)
                stack.append((child, trail + (child,)))
        return None

    def same_person(self, start: str, end: str) -> bool:
        """True only along a lineage that never branched between the two.

        Three things have to hold: there is a chain of transfers, the canonical
        state is preserved at every step, and no stage on that chain has a
        second continuer. The third is not a convention — one predecessor
        cannot be numerically identical to two distinct successors.

        That this relation is what personal persistence consists in is the
        causal-lineage postulate. It is a postulate.
        """
        trail = self.path(start, end)
        if trail is None:
            return False
        for index, node in enumerate(trail[:-1]):
            nxt = trail[index + 1]
            edge = next(
                (e for e in self.edges if e.parent == node and e.child == nxt), None
            )
            if edge is None or not edge.preserved or edge.kind not in NONBRANCHING:
                return False
            if self.branched(node):
                return False
        return True

    def continuers(self, name: str) -> tuple[str, ...]:
        """Every later stage descended from this one, branches included."""
        out: list[str] = []
        stack = list(self.children(name))
        seen: set[str] = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            out.append(node)
            stack.extend(self.children(node))
        return tuple(sorted(out))

    def verify(self) -> list[str]:
        """Every edge whose signature no longer matches what it covers."""
        broken: list[str] = []
        for edge in self.edges:
            if edge.parent not in self.stages or edge.child not in self.stages:
                broken.append(f"{edge.parent}->{edge.child}: a stage is missing")
                continue
            if not hmac.compare_digest(
                edge.signature, self._sign(edge.parent, edge.child, edge.kind)
            ):
                broken.append(f"{edge.parent}->{edge.child}: signature does not match")
        return broken

    def as_dict(self) -> dict[str, Any]:
        roots = sorted(n for n in self.stages if not self.parents(n))
        return {
            "stages": [s.as_dict() for s in self.stages.values()],
            "edges": [e.as_dict() for e in self.edges],
            "roots": roots,
            "branch_points": sorted(n for n in self.stages if self.branched(n)),
            "unattached": [
                n for n in roots if n != next(iter(self.stages), None)
            ],
            "signatures_verify": not self.verify(),
            "postulate": (
                "personal persistence consists in unique nonbranching causal "
                "continuation of the canonical person-state; this is a postulate"
            ),
        }


def continuity_strength(lineage: Lineage, parent: str, child: str) -> float:
    """How much of the child's person-state the parent's accounts for.

    The share of the child's identity-bearing fields that arrived unchanged
    along the chain from the parent. Zero when no chain connects them.

    This is an operational continuity measure and it is not identity. It can be
    high across a fork, where both descendants carry nearly everything and
    neither is the strict successor, and that is the whole reason the lineage
    graph is the primitive rather than this number.
    """
    trail = lineage.path(parent, child)
    if trail is None or len(trail) < 2:
        return 1.0 if trail is not None else 0.0
    kept = 0
    steps = 0
    for index, node in enumerate(trail[:-1]):
        nxt = trail[index + 1]
        edge = next((e for e in lineage.edges if e.parent == node and e.child == nxt), None)
        if edge is None:
            return 0.0
        steps += 1
        kept += 1 if edge.preserved else 0
    return kept / max(1, steps)


def scenarios(lineage: Lineage, names: Sequence[str]) -> dict[str, Any]:
    """What the graph says about a set of stages, for a report."""
    out: dict[str, Any] = {}
    for name in names:
        out[name] = {
            "parents": list(lineage.parents(name)),
            "children": list(lineage.children(name)),
            "branched": lineage.branched(name),
            "continuers": list(lineage.continuers(name)),
            "continuity_strength": {
                other: round(continuity_strength(lineage, name, other), 4)
                for other in lineage.continuers(name)
            },
        }
    return out


def _iter_edges(edges: Iterable[Edge]) -> tuple[Edge, ...]:
    return tuple(edges)
