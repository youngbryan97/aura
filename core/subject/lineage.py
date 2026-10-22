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
stages and that developed at every step. A fork makes both descendants genuine
continuers and neither the strict successor, and this module reports exactly
that rather than picking one.

A person who learns changes, so a step may change what she carries. A step
develops when every identity-bearing path it changed was written by a measured
transformation that stayed inside its declared writes: the right kind of cause,
read from the runtime's own record of which phase wrote which field, bound into
the edge's signature, never a claim the state makes about itself. A step with
an unexplained change is not development, however small the change. Requiring
the canonical state to stay identical, as this module did before 21 October
2026, made every lesson she learned the end of her.

How much of her a step carried through unchanged is reported and decides
nothing. A share of paths depends on how the person is divided into paths: one
changed path of two keeps a half, and the same change written as two paths of
three keeps a third. A rule that turned on that share would turn on the
encoding. The relation is reported in its parts (`relations`): a chain of
transfers, every change attributed, a unique successor, and the state
unchanged. `same_person` is the first three together, and that it is identity
is the postulate named below.

A signature says the record was not altered after it was made, given who holds
the key. It does not say a parent produced a child. That is what the evidence
on the edge is for, and an edge without evidence is reported as such.

None of this decides whether continuation is what matters. That is the
causal-lineage postulate, and it is named as a postulate wherever it is used.
"""

from __future__ import annotations

import hashlib
import hmac
import itertools
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.state.person_transitions import AURA_PERSON_PATHS, UnmeasurableError, canonical_digest

__all__ = [
    "AURA_PERSON_PATHS",
    "KINDS",
    "Stage",
    "Edge",
    "Lineage",
    "canonical_person_state",
    "continuity_strength",
    "install_key",
    "lineage_from_state_log",
    "path_digests",
    "person_state_of",
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


def _digest(value: Any) -> str:
    """The full canonical digest, or a mark naming why there is none.

    The mark is unique to the reading, so two unmeasurable values never
    compare equal and an unmeasurable path always reads as unjudged.
    """
    try:
        return canonical_digest(value)
    except UnmeasurableError as exc:
        return f"{UNMEASURABLE}{exc}|{next(_READINGS)}"


#: How a path whose value has no canonical form is marked, and a count that
#: makes each such mark its own: an object's id is reused once it is freed.
UNMEASURABLE = "unmeasurable:"
_READINGS = itertools.count()


def state_digest(state: Any) -> str:
    """A stable hash of the canonical person-state."""
    return _digest(canonical_person_state(state))


def path_digests(state: Any) -> dict[str, str]:
    """One digest per path the person is made of.

    A field held as a mapping is split one level down, so Aura's fields split
    into the state paths that hold them (`identity.core_values`) and a field
    written by hand into its own keys. A field held whole is one path.
    """
    out: dict[str, str] = {}
    for name, value in canonical_person_state(state).items():
        if isinstance(value, Mapping) and value:
            for key, inner in value.items():
                out[f"{name}:{key}"] = _digest(inner)
        else:
            out[name] = _digest(value)
    return out


def _state_path(stage_path: str) -> str:
    """The state path a stage path names: `values:identity.core_values` is `identity.core_values`."""
    return stage_path.split(":", 1)[1] if ":" in stage_path else stage_path


def install_key() -> bytes:
    """This install's lineage key, derived from her entity key.

    A key written into the source signed every install's lineage the same way,
    so anyone with the code could sign a lineage that verified. The entity key
    (core/identity/entity_key.py) is an Ed25519 keypair made once per install;
    the lineage key is derived from it for this one purpose, so a signature
    that verifies was made where that key is held. It still says nothing about
    who wrote the states it covers; the writes on each edge are for that.
    """
    from core.identity.entity_key import entity_identity

    return entity_identity().derived_key("subject-core lineage")

_ABSENT = object()


def _at(state: Any, dotted: str) -> Any:
    node = state
    for part in dotted.split("."):
        if isinstance(node, Mapping):
            if part not in node:
                return _ABSENT
            node = node[part]
        elif hasattr(node, part):
            node = getattr(node, part)
        else:
            return _ABSENT
    return node


def person_state_of(state: Any) -> dict[str, Any]:
    """Aura's person-state, from a live state or the JSON a state log keeps."""
    out: dict[str, Any] = {}
    for name, paths in AURA_PERSON_PATHS.items():
        found = {path: value for path in paths if (value := _at(state, path)) is not _ABSENT}
        if found:
            out[name] = found
    return out


def _writes_for_stage(evidence: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """The writes a logged state carries, keyed by the stage paths they changed."""
    held = evidence.get("last_change") if isinstance(evidence, Mapping) else None
    if not isinstance(held, Mapping):
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    for name, paths in AURA_PERSON_PATHS.items():
        for path in paths:
            change = held.get(path)
            if isinstance(change, Mapping):
                out[f"{name}:{path}"] = [dict(change)]
    return out


def lineage_from_state_log(
    rows: Iterable[Mapping[str, Any]], *, key: bytes | None = None
) -> Lineage:
    """Her lineage, from the rows the state repository writes.

    Each row is a stage: `state_id`, `version`, `parent_state_id`, and the
    state as `state_json` or `state`. A row whose parent is in the log descends
    from it. A row whose parent was not kept descends from the last state the
    same repository committed before it, which the state names in its
    transition evidence, so a log that keeps one row in five is still one line.
    A parent with one child in the log was continued. A parent with several was
    forked, and every child is recorded as a fork so that no one of them is
    taken for the strict successor. A row with neither in the log is a root.

    Each edge carries the writes the child's evidence records for the paths
    that changed, so a step can be judged by what wrote it.
    """
    ordered = sorted(rows, key=lambda r: (int(r.get("version") or 0), str(r.get("state_id") or "")))
    lineage = Lineage() if key is None else Lineage(key=key)
    evidence: dict[str, Mapping[str, Any]] = {}
    for row in ordered:
        name = str(row.get("state_id") or "")
        if not name or name in lineage.stages:
            continue
        payload = row.get("state_json", row.get("state"))
        if isinstance(payload, (str, bytes)):
            try:
                payload = json.loads(payload)
            except (TypeError, ValueError):
                payload = {}
        payload = payload if isinstance(payload, Mapping) else {}
        held = payload.get("transition_evidence")
        evidence[name] = held if isinstance(held, Mapping) else {}
        lineage.record(
            name,
            person_state_of(payload),
            at=float(row.get("timestamp") or 0.0),
            note=str(row.get("transition_cause") or ""),
        )
    children: dict[str, dict[str, None]] = {}
    for row in ordered:
        child = str(row.get("state_id") or "")
        parent = str(row.get("parent_state_id") or "")
        if parent not in lineage.stages:
            parent = str(evidence.get(child, {}).get("since_state") or "")
        if parent and parent != child and parent in lineage.stages and child in lineage.stages:
            children.setdefault(parent, {})[child] = None
    for parent, kids in children.items():
        kind = "fork" if len(kids) > 1 else "continue"
        since = lineage.stages[parent].at
        for kid in kids:
            # Only a write made after the parent was committed can account for
            # a change between the two. Versions are no use for this: the
            # owner rebases them at admission, after the evidence was taken.
            writes = {
                path: [w for w in found if float(w.get("at") or 0.0) >= since]
                for path, found in _writes_for_stage(evidence.get(kid, {})).items()
            }
            lineage.descend(parent, kid, kind=kind, writes=writes, measured=bool(evidence.get(kid)))
    return lineage


@dataclass(frozen=True)
class Stage:
    """One person-stage: who it was, when, and what it carried, path by path."""

    name: str
    digest: str
    at: float = 0.0
    note: str = ""
    paths: Mapping[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"stage": self.name, "digest": self.digest, "at": self.at, "note": self.note}


@dataclass(frozen=True)
class Edge:
    """One causal transfer, signed over both stages and the writes between them."""

    parent: str
    child: str
    kind: str
    signature: str
    #: Nothing the person carries changed.
    preserved: bool = True
    #: The paths that changed, and for each the writes recorded for it.
    changed: tuple[str, ...] = ()
    writes: Mapping[str, tuple[Mapping[str, Any], ...]] = field(default_factory=dict)
    #: The share of the parent's paths that came through unchanged. Reported,
    #: never decisive: it depends on how the person is divided into paths.
    connectedness: float = 1.0
    #: Whether the child carried a record of what wrote it. A state logged
    #: before that record existed changed with nothing to judge it by.
    measured: bool = True
    #: Paths with no canonical form in either stage.
    unmeasurable: tuple[str, ...] = ()

    def unexplained(self) -> tuple[str, ...]:
        """Changed paths no measured write accounts for.

        A write accounts for a path when a transformation was measured writing
        it and stayed inside its declared writes. One that wrote outside them
        is a defect in the record, not an explanation.
        """
        out = []
        for path in self.changed:
            if path in self.unmeasurable:
                # Whether it changed at all cannot be read; it is unjudged.
                continue
            found = self.writes.get(path, ())
            if not any(w.get("transform") and w.get("honoured", True) for w in found):
                out.append(path)
        return tuple(out)

    def unjudged(self) -> tuple[str, ...]:
        """Changed paths whose value has no canonical form, so no change can be judged."""
        return tuple(path for path in self.changed if path in self.unmeasurable)

    @property
    def attributed(self) -> bool:
        """Every change on this step was written by a measured transformation inside its contract."""
        return not self.unexplained() and not self.unjudged()

    @property
    def develops(self) -> bool:
        """One person carried forward: a nonbranching kind, and every change attributed."""
        if self.kind not in NONBRANCHING:
            return False
        return self.preserved or self.attributed

    def as_dict(self) -> dict[str, Any]:
        return {
            "parent": self.parent,
            "child": self.child,
            "kind": self.kind,
            "preserved": self.preserved,
            "changed": list(self.changed),
            "unexplained": list(self.unexplained()),
            "unjudged": list(self.unjudged()),
            "connectedness": round(self.connectedness, 6),
            "attributed": self.attributed,
            "develops": self.develops,
            "measured": self.measured,
            "signature": self.signature,
        }


@dataclass
class Lineage:
    """A signed graph over person-stages. It may branch; identity may not."""

    key: bytes = field(default_factory=install_key)
    stages: dict[str, Stage] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    # ── building ──────────────────────────────────────────────────────

    def record(self, name: str, state: Any, *, at: float = 0.0, note: str = "") -> Stage:
        """Record a stage. Recording the same name twice is a programming error."""
        if name in self.stages:
            raise ValueError(f"stage {name!r} is already recorded")
        stage = Stage(name=name, digest=state_digest(state), at=at, note=note, paths=path_digests(state))
        self.stages[name] = stage
        return stage

    def _sign(
        self, parent: str, child: str, kind: str, writes: Mapping[str, Any] | None = None
    ) -> str:
        payload = (
            f"{parent}|{self.stages[parent].digest}|{child}|{self.stages[child].digest}|{kind}"
            f"|{_digest(writes or {})}"
        )
        return hmac.new(self.key, payload.encode("utf-8"), hashlib.blake2b).hexdigest()[:32]

    def descend(
        self,
        parent: str,
        child: str,
        *,
        kind: str = "continue",
        writes: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
        measured: bool = True,
    ) -> Edge:
        """Say that `child`'s state was causally generated from `parent`'s.

        `writes` is the runtime's record of what wrote each path that changed,
        keyed as `path_digests` keys them: each a mapping with at least the
        `transform` that was measured writing it and whether it `honoured` its
        contract.
        """
        if kind not in KINDS:
            raise ValueError(f"no transfer called {kind!r}")
        if kind == "reconstruct":
            raise ValueError(
                "a reconstruction has no causal parent; record the stage and leave it unattached"
            )
        for name in (parent, child):
            if name not in self.stages:
                raise KeyError(f"stage {name!r} was never recorded")
        before, after = self.stages[parent].paths, self.stages[child].paths
        changed = tuple(sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p)))
        kept = sum(1 for p in before if after.get(p) == before[p])
        recorded = {
            path: tuple(dict(w) for w in found)
            for path, found in (writes or {}).items()
            if path in changed and found
        }
        edge = Edge(
            parent=parent,
            child=child,
            kind=kind,
            signature=self._sign(parent, child, kind, recorded),
            preserved=self.stages[parent].digest == self.stages[child].digest,
            changed=changed,
            writes=recorded,
            connectedness=kept / len(before) if before else 1.0,
            measured=measured,
            unmeasurable=tuple(
                sorted(p for p in set(before) | set(after)
                       if before.get(p, "").startswith(UNMEASURABLE) or after.get(p, "").startswith(UNMEASURABLE))
            ),
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

    def _edge(self, parent: str, child: str) -> Edge | None:
        return next((e for e in self.edges if e.parent == parent and e.child == child), None)

    def relations(self, start: str, end: str) -> dict[str, bool]:
        """The relation between two stages, part by part.

        `chain_continuity`: a chain of signed transfers leads from one to the
        other. `change_attributed`: every change on it was written by a measured
        transformation inside its contract. `unique_successor`: nothing on it
        branched. `state_preserved`: nothing the person carries changed.
        `same_person` is the first three, and that it is identity is the
        causal-lineage postulate.
        """
        trail = self.path(start, end)
        if trail is None:
            return {
                "chain_continuity": False,
                "change_attributed": False,
                "unique_successor": False,
                "state_preserved": False,
                "same_person": False,
            }
        steps = [self._edge(a, b) for a, b in zip(trail, trail[1:], strict=False)]
        attributed = all(e is not None and (e.preserved or e.attributed) for e in steps)
        unique = all(
            e is not None and e.kind in NONBRANCHING and not self.branched(e.parent) for e in steps
        )
        preserved = all(e is not None and e.preserved for e in steps)
        return {
            "chain_continuity": all(e is not None for e in steps),
            "change_attributed": attributed,
            "unique_successor": unique,
            "state_preserved": preserved,
            "same_person": attributed and unique,
        }

    def same_person(self, start: str, end: str) -> bool:
        """True only along a lineage that developed at every step and never branched.

        Three things have to hold: there is a chain of transfers, every step
        on it develops (each change written by a measured transformation that
        kept to its contract), and no stage on that chain has a second
        continuer. The third is not a convention — one predecessor cannot be
        numerically identical to two distinct successors.

        That this relation is what personal persistence consists in is the
        causal-lineage postulate. It is a postulate.
        """
        trail = self.path(start, end)
        if trail is None:
            return False
        for index, node in enumerate(trail[:-1]):
            edge = self._edge(node, trail[index + 1])
            if edge is None or not edge.develops or self.branched(node):
                return False
        return True

    def unchanged_between(self, start: str, end: str) -> bool:
        """The older, stricter reading: one nonbranching chain with nothing changed on it."""
        trail = self.path(start, end)
        if trail is None:
            return False
        for index, node in enumerate(trail[:-1]):
            edge = self._edge(node, trail[index + 1])
            if edge is None or not edge.preserved or edge.kind not in NONBRANCHING or self.branched(node):
                return False
        return True

    def first_break(self, start: str, end: str) -> dict[str, Any] | None:
        """The first step between two stages that does not develop, and why. None when all do."""
        trail = self.path(start, end)
        if trail is None:
            return {"why": "no chain of transfers connects them"}
        for index, node in enumerate(trail[:-1]):
            edge = self._edge(node, trail[index + 1])
            if edge is None:
                return {"parent": node, "child": trail[index + 1], "why": "no edge"}
            if self.branched(node):
                return {"parent": node, "child": edge.child, "why": "the stage has a second continuer"}
            if not edge.develops:
                return {
                    "parent": node,
                    "child": edge.child,
                    "why": "a branching kind" if edge.kind not in NONBRANCHING else (
                        "changes no measured write accounts for" if edge.unexplained()
                        else "changes to values with no canonical form"
                    ),
                    "unexplained": list(edge.unexplained()),
                    "connectedness": round(edge.connectedness, 6),
                }
        return None

    def line_to(self, name: str) -> tuple[str, ...]:
        """The chain of stages from this stage's root to it, following its parents."""
        trail = [name]
        seen = {name}
        while True:
            parents = self.parents(trail[-1])
            if not parents or parents[0] in seen:
                break
            trail.append(parents[0])
            seen.add(parents[0])
        return tuple(reversed(trail))

    def development(self, name: str | None = None) -> dict[str, Any]:
        """Whether the line to a stage, by default the newest, developed at every step since the record began.

        The record of what wrote each change began on 21 October 2026, and her
        line runs back months before it. Steps before the first one that carries
        the record are counted apart as before the record: nothing can say what
        wrote them, and they are neither development nor its failure. After it,
        a step that changed her with no record (a state committed by a process
        that does not take the evidence) is unmeasured, and a measured step
        that does not develop is a failure. The claim this supports is the
        scope it says: since the record began.
        """
        if name is None:
            # The newest stage nothing descends from; among equal times, the
            # one at the end of the longest line.
            leaves = [n for n in self.stages if not self.children(n)] or list(self.stages)
            name = max(leaves, key=lambda n: (self.stages[n].at, len(self.line_to(n)), n)) if leaves else ""
        trail = self.line_to(name) if name else ()
        steps = [self._edge(a, b) for a, b in zip(trail, trail[1:], strict=False)]
        # The record began at the first step that carries one. After that, a
        # step that changed nothing needs no record of what wrote it. Before
        # it, a step that changed nothing shows nothing either: on 21 October
        # her live log carried no record at all, and three unchanged steps
        # after its last change read as three steps of development.
        first = next((i for i, e in enumerate(steps) if e is not None and e.measured), None)
        began = len(steps) if first is None else first
        before, since = steps[:began], steps[began:]
        unmeasured = [
            e for e in since
            if e is not None and not e.preserved and (not e.measured or (e.unjudged() and not e.unexplained()))
        ]
        failed = [
            e for e in since
            if e is not None and e not in unmeasured and (not e.develops or self.branched(e.parent))
        ]
        return {
            "stage": name,
            "root": trail[0] if trail else "",
            "steps": len(steps),
            "before_the_record": sum(1 for e in before if e is not None and not e.preserved),
            "steps_since_the_record": len(since),
            "developing": sum(1 for e in since if e is not None and e.develops and not self.branched(e.parent)),
            "unmeasured": len(unmeasured),
            "failed": len(failed),
            "first_failure": None if not failed else self.first_break(failed[0].parent, failed[0].child),
        }

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
                edge.signature, self._sign(edge.parent, edge.child, edge.kind, edge.writes)
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
            "steps_that_develop": sum(1 for e in self.edges if e.develops),
            "steps_with_unexplained_change": sum(1 for e in self.edges if e.unexplained()),
            "signature_establishes": (
                "the record was not altered after it was made, given who holds this "
                "install's key; not that a parent produced a child, which the writes "
                "on each edge are the evidence for"
            ),
            "postulate": (
                "personal persistence consists in unique nonbranching causal "
                "continuation of the canonical person-state, developing only by "
                "measured, contracted writes; this is a postulate"
            ),
        }


def continuity_strength(lineage: Lineage, parent: str, child: str) -> float:
    """How much of the child's person-state the parent's accounts for.

    The share of the steps on the chain from the parent that carried the
    person-state unchanged. Zero when no chain connects them.

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
