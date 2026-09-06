"""One vocabulary for what a remembered thing is, and what actually gets used.

`core/memory/base.py` declares :class:`MemoryType` with five values. Exactly
one file imports it: the file that defines it. Meanwhile thirty-three modules
under ``core/memory`` write durably, and between them they write twenty-six
different strings into a field meaning "what kind of thing is this" — including
``episode`` and ``episodic`` for the same thing, ``skill`` against the enum's
``skills``, and ``goal`` against its ``goals``.

A taxonomy nobody uses is not a taxonomy, and the cost is specific: a reader
that wants every episodic memory has to know all the spellings, so it gets the
ones its author happened to think of. Consolidation, recall and forgetting all
read across stores, and each of them is a place where a spelling can be missed
silently.

Two things here, and the second is the one that keeps it honest:

* :func:`what_kind` maps any of the strings actually written onto one of the
  canonical kinds, so a reader asks for a kind rather than a spelling.
* :func:`strings_nothing_maps` names the ones it cannot place. That number is
  a ratchet: a new spelling shows up as an unmapped string rather than as a
  memory quietly missing from a query.
"""
from __future__ import annotations

import ast
import functools
import logging
import pathlib
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger("Aura.WhatKindOfMemoryIsThis")

__all__ = [
    "MemoryKind",
    "ARecord",
    "what_kind",
    "the_spellings",
    "strings_nothing_maps",
    "kinds_written_in_the_tree",
    "how_the_kinds_stand",
]


class MemoryKind(StrEnum):
    """What a remembered thing is. Every durable store writes one of these."""

    #: Something that happened, at a time, to her.
    EPISODIC = "episodic"
    #: Something that is the case, independent of when she learned it.
    SEMANTIC = "semantic"
    #: A person, and what is true of them.
    PERSON = "person"
    #: Something she means to do.
    GOAL = "goal"
    #: How to do something.
    SKILL = "skill"
    #: Something she asked and does not yet know.
    QUESTION = "question"
    #: Something she thought rather than said. Never served as an answer.
    PRIVATE_THOUGHT = "private_thought"
    #: A summary standing in for episodes that were compacted away.
    CONSOLIDATED = "consolidated"


#: Every string measured in the tree, mapped onto the kind it means. Built by
#: reading what the stores write, not by deciding what they should have.
_SPELLINGS: dict[str, MemoryKind] = {
    "episodic": MemoryKind.EPISODIC,
    "episode": MemoryKind.EPISODIC,
    "interaction_trace": MemoryKind.EPISODIC,
    "interaction_commit": MemoryKind.EPISODIC,
    "milestone": MemoryKind.EPISODIC,
    "facade_add_memory": MemoryKind.EPISODIC,
    "long_term_memory": MemoryKind.EPISODIC,
    "semantic": MemoryKind.SEMANTIC,
    "interaction_semantic": MemoryKind.SEMANTIC,
    "fact": MemoryKind.SEMANTIC,
    "knowledge": MemoryKind.SEMANTIC,
    "shared_ground": MemoryKind.SEMANTIC,
    "person": MemoryKind.PERSON,
    "relationship": MemoryKind.PERSON,
    "goal": MemoryKind.GOAL,
    "goals": MemoryKind.GOAL,
    "active": MemoryKind.GOAL,
    "queued": MemoryKind.GOAL,
    "skill": MemoryKind.SKILL,
    "skills": MemoryKind.SKILL,
    "question": MemoryKind.QUESTION,
    "unanswered": MemoryKind.QUESTION,
    "answered": MemoryKind.QUESTION,
    "curiosity": MemoryKind.QUESTION,
    "qa": MemoryKind.QUESTION,
    "private_thought": MemoryKind.PRIVATE_THOUGHT,
    "reflection": MemoryKind.PRIVATE_THOUGHT,
    "consolidated": MemoryKind.CONSOLIDATED,
    "consolidated_concept": MemoryKind.CONSOLIDATED,
    "episodic_compaction": MemoryKind.CONSOLIDATED,
    "day_summary": MemoryKind.CONSOLIDATED,
}


def the_spellings() -> dict[str, str]:
    """Every string that maps, and what it maps to."""
    return {word: str(kind) for word, kind in sorted(_SPELLINGS.items())}


def what_kind(said: Any, *, default: MemoryKind | None = None) -> MemoryKind | None:
    """The canonical kind for whatever a store wrote, or None if unplaceable.

    None rather than a guess: a memory filed under the wrong kind is worse
    than one filed under none, because a query for the wrong kind returns it
    and a query for the right kind does not.
    """
    word = str(said or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not word:
        return default
    found = _SPELLINGS.get(word)
    if found is not None:
        return found
    # A store that wrote the canonical value itself.
    try:
        return MemoryKind(word)
    except ValueError:
        return default


@dataclass(frozen=True, slots=True)
class ARecord:
    """The envelope every durable memory store writes.

    Deliberately small. A store keeps whatever else it needs in ``carries``;
    what it may not do is disagree with another store about what these five
    mean.
    """

    kind: MemoryKind
    #: What the thing says, in words.
    said: str
    #: When, as wall time, because a memory is compared against a person's day.
    at: float
    #: Where it came from, so a claim can be traced to what supports it.
    from_where: str = ""
    carries: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": str(self.kind),
            "said": self.said,
            "at": self.at,
            "from_where": self.from_where,
            "carries": dict(self.carries),
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "ARecord | None":
        """Read a row any store wrote. None where it names no placeable kind."""
        kind = what_kind(
            row.get("kind")
            or row.get("memory_type")
            or row.get("type")
            or row.get("category")
        )
        if kind is None:
            return None
        return cls(
            kind=kind,
            said=str(row.get("said") or row.get("content") or row.get("text") or ""),
            at=float(row.get("at") or row.get("timestamp") or row.get("t") or 0.0),
            from_where=str(row.get("from_where") or row.get("source") or ""),
            carries={
                k: v
                for k, v in row.items()
                if k
                not in {
                    "kind",
                    "memory_type",
                    "type",
                    "category",
                    "said",
                    "content",
                    "text",
                    "at",
                    "timestamp",
                    "t",
                    "from_where",
                    "source",
                }
            },
        )


_KINDISH = re.compile(r"^(memory_type|kind|category|type|record_type|entry_type)$")


@functools.lru_cache(maxsize=4)
def kinds_written_in_the_tree(repo: str = ".") -> dict[str, list[str]]:
    """Every kind-string written under core/memory, and which files write it."""
    base = pathlib.Path(repo) / "core" / "memory"
    found: dict[str, set[str]] = {}
    if not base.exists():
        return {}
    for path in sorted(base.glob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except (SyntaxError, ValueError, OSError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if (
                        isinstance(key, ast.Constant)
                        and isinstance(key.value, str)
                        and _KINDISH.match(key.value)
                        and isinstance(value, ast.Constant)
                        and isinstance(value.value, str)
                    ):
                        found.setdefault(value.value, set()).add(path.name)
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if (
                        kw.arg
                        and _KINDISH.match(kw.arg)
                        and isinstance(kw.value, ast.Constant)
                        and isinstance(kw.value.value, str)
                    ):
                        found.setdefault(kw.value.value, set()).add(path.name)
    return {word: sorted(files) for word, files in sorted(found.items())}


def strings_nothing_maps(repo: str = ".") -> tuple[str, ...]:
    """Kind-strings written in the tree that this vocabulary cannot place.

    The ratchet. A new spelling appears here rather than as a memory quietly
    missing from every query that asks for its kind.
    """
    return tuple(
        sorted(word for word in kinds_written_in_the_tree(repo) if what_kind(word) is None)
    )


def how_the_kinds_stand(repo: str = ".") -> dict[str, Any]:
    """For the health report: the vocabulary, the spellings, and the strays."""
    written = kinds_written_in_the_tree(repo)
    unmapped = strings_nothing_maps(repo)
    by_kind: dict[str, list[str]] = {}
    for word in written:
        kind = what_kind(word)
        if kind is not None:
            by_kind.setdefault(str(kind), []).append(word)
    return {
        "kinds": [str(k) for k in MemoryKind],
        "spellings_known": len(_SPELLINGS),
        "strings_written": len(written),
        "strings_nothing_maps": list(unmapped),
        "spellings_per_kind": {k: sorted(v) for k, v in sorted(by_kind.items())},
        "kinds_nothing_writes": sorted(
            str(k) for k in MemoryKind if str(k) not in by_kind
        ),
    }
