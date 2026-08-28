"""What was learned in one world, available on entering the next.

Persistence of learned structure has been per-world, which is honest and costs
transfer: knowledge of world A does not become better initial learning in world
B. The unit that transfers is not the relation — a world exchanging positions 0
and 2 is a different relation from one exchanging 1 and 3 — it is the *shape*
the relation has. Pairwise exchange, an offset that wraps, a mirror, a constant.

So this keeps the shapes, counts how often each has accounted for a world, and
offers them in that order to the next world. That is a prior over transition
structure, carried across environments, which is the thing per-world
persistence cannot give.

Why an order is worth anything
------------------------------
One observation is often ambiguous between shapes. ``(1,2) -> (2,1)`` is a
pairwise exchange, and a mirror, and an offset of one: all three fit, and
telling them apart needs a second observation of a different length. A system
that has seen mirrors before picks the mirror from the first observation and is
right more often than chance — and wrong sometimes, which is measured here
rather than assumed away.

The measurement this file exists to support is ``observations_needed``: how many
observations a world takes to pin down, with and without a prior. A prior that
does not lower that number on unseen worlds is not transfer, whatever else it
is.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.cognition.primitive_invention import (
    InventedRelation,
    Transition,
    explains,
    invent_relation,
)

__all__ = ["RelationLanguage", "observations_needed"]


@dataclass
class RelationLanguage:
    """Shapes of transition that have accounted for a world before.

    Holds no world, no state and no domain: only which shapes have worked and
    how often. That is what makes it carry across environments.
    """

    counts: dict[str, int] = field(default_factory=dict)
    #: The shapes themselves, by their description, so the next world can
    #: compose with them. Held in memory: a rule over indices is a function and
    #: the counts above are what survive a restart.
    forms: dict[str, tuple[str, object, tuple[str, ...]]] = field(default_factory=dict)
    path: Path | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def admit(self, relation: InventedRelation | None) -> None:
        """Record that this shape accounted for a world."""

        if relation is None or not relation.family:
            return
        with self._lock:
            self.counts[relation.family] = self.counts.get(relation.family, 0) + 1
            if relation.index_rule is not None and relation.form:
                self.forms[relation.form] = (
                    relation.family,
                    relation.index_rule,
                    tuple(relation.components) or (relation.form,),
                )

    def families(self) -> list[str]:
        """The shapes, most often useful first."""

        with self._lock:
            return [
                name
                for name, _count in sorted(
                    self.counts.items(), key=lambda row: (-row[1], row[0])
                )
            ]

    def prefers(self, family: str) -> int:
        """How much evidence there is for this shape. Zero for an unseen one."""

        with self._lock:
            return int(self.counts.get(str(family), 0))

    def explain(
        self,
        transitions: Sequence[Transition],
        *,
        held_out: Sequence[Transition] = (),
        without: frozenset[str] = frozenset(),
    ) -> InventedRelation | None:
        """The relation these transitions need, preferring a shape seen before.

        The invention does the work; this decides which of several fitting
        answers to keep when the observations do not separate them. With an
        empty language it is exactly the invention, which is what makes the
        comparison in ``observations_needed`` a fair one.
        """

        with self._lock:
            prior = dict(self.counts)
            known = [
                (family, description, rule, parts)
                for description, (family, rule, parts) in self.forms.items()
            ]
        return invent_relation(
            transitions,
            held_out=held_out,
            prefer={} if "prior" in without else prior,
            known_forms=known,
            without=without,
        )

    def refactor(self) -> str:
        """Admit structure several solved shapes share, which none of them is.

        A library that only keeps whole winners can hold nothing it has not
        already seen entire, and the long-term studies of chunking in Soar and
        ACT-R report where that ends: symbolic learning eventually stops. What
        keeps DreamCoder's library growing is this step — refactor the
        solutions, find the sub-structure common across them, and admit that.

        The sub-sequence chosen is the one that saves most: a run appearing in
        several shapes is worth (occurrences - 1) x (its length), because each
        occurrence after the first collapses to one part. That is the
        description-length argument, done by counting rather than by taste.

        Returns the description admitted, or "" when nothing is shared.
        """

        with self._lock:
            structures = [parts for _f, _r, parts in self.forms.values()]
            rules = {
                description: rule for description, (_f, rule, _p) in self.forms.items()
            }
        if len(structures) < 2:
            return ""
        counts: dict[tuple[str, ...], int] = {}
        for parts in structures:
            seen_here: set[tuple[str, ...]] = set()
            for start in range(len(parts)):
                for stop in range(start + 2, len(parts) + 1):
                    run = tuple(parts[start:stop])
                    if run in seen_here:
                        continue
                    seen_here.add(run)
                    counts[run] = counts.get(run, 0) + 1
        shared = {
            run: count
            for run, count in counts.items()
            if count >= 2 and run not in {tuple(p) for p in structures}
        }
        if not shared:
            return ""
        best = max(shared, key=lambda run: ((shared[run] - 1) * len(run), -len(run)))
        rebuilt = _apply_in_order(best, rules)
        if rebuilt is None:
            return ""
        description = ", then ".join(best)
        with self._lock:
            if description in self.forms:
                return ""
            self.forms[description] = ("refactored", rebuilt, tuple(best))
            self.counts["refactored"] = self.counts.get("refactored", 0) + 1
        return description

    def save(self) -> None:
        """Write the shapes down, through the runtime's own write path."""

        if self.path is None:
            return
        with self._lock:
            payload = json.dumps({"counts": dict(self.counts)}, indent=2, sort_keys=True)
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            with local_internal_governed_scope(
                "cognition.relation_language", domain="state_mutation"
            ):
                get_file_write_gateway().write_text(
                    self.path, payload, source="cognition.relation_language"
                )
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return

    @classmethod
    def load(cls, path: Path | str) -> RelationLanguage:
        """The shapes learned in earlier sessions, or an empty language."""

        target = Path(path)
        try:
            raw: Any = json.loads(target.read_text())
        except (OSError, ValueError):
            return cls(path=target)
        counts = raw.get("counts") if isinstance(raw, dict) else None
        if not isinstance(counts, dict):
            return cls(path=target)
        return cls(
            counts={str(k): int(v) for k, v in counts.items() if str(k)},
            path=target,
        )


def _apply_in_order(
    parts: Sequence[str],
    rules: Mapping[str, Any],
) -> Any:
    """One rule over indices for these parts applied innermost first.

    Parts are resolved from the library first and from the basis second, since
    most parts of a learned shape are basis atoms. None when a part names
    neither, which happens when a run crosses into a learned shape that has
    since been dropped.
    """

    from core.cognition.primitive_invention import rule_for_description

    found = [rules.get(part) or rule_for_description(part) for part in parts]
    if any(rule is None for rule in found):
        return None

    def rule(index: int, size: int, _chain=tuple(found)) -> int:
        position = index
        for step in reversed(_chain):
            position = step(position, size)
        return position

    return rule


def observations_needed(
    world: Sequence[Transition],
    *,
    language: RelationLanguage | None = None,
    minimum: int = 1,
) -> int | None:
    """How many observations of this world pin its relation down.

    Counts the shortest prefix after which the invented relation predicts every
    remaining observation. None when the world is never pinned down, which is
    the answer for a world with no relation in it.

    This is the measurement. A prior earns the word transfer by lowering this
    number on a world it has not seen, and the number is reported the same way
    with and without one.
    """

    observed = list(world)
    if len(observed) <= minimum:
        return None
    for size in range(max(1, int(minimum)), len(observed)):
        prefix, rest = observed[:size], observed[size:]
        speaker = language.explain if language is not None else invent_relation
        found = speaker(prefix)
        if found is None:
            continue
        if explains(found.apply, rest):
            return size
    return None
