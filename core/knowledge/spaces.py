"""core/knowledge/spaces.py — one query surface over stores that share nothing.

Aura keeps knowledge in a metagraph, a vector store, a world model, a procedure
registry and a memory. Each has the right interface for what it is, and a
cognitive program that wants to ask all five the same question has to be
written five times. That is why introspection reads self-state rather than
program structure: there is no surface general enough to ask a structural
question over.

A :class:`Space` is that surface: add, remove, query, iterate, transaction,
provenance, attention. A store implements it by adapter and keeps its own
internals, so nothing is rewritten and a generic program - a compressor, a
metacognitive check, a query for everything that mentions a concept - runs over
all of them.

Nondeterminism is carried, not collapsed
----------------------------------------
:meth:`Space.query` returns weighted alternatives. Aura's rewriter already
produces several reductions and the rest of cognition collapses them to one
before anything downstream can weigh them, so an ambiguity discovered by
inference is lost at the first join. Alternatives travel here, and a caller
that wants one calls :meth:`Alternatives.best` explicitly, which makes the
collapse a decision somebody made.

Packages
--------
:class:`Package` gives a group of rules a name, a version and a rollback. A
learned rule set can be loaded, measured, and removed exactly - which is the
prerequisite for letting Aura change her own inference policy at all, and the
reason this is here rather than in the invention module.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "Alternatives",
    "Space",
    "DictSpace",
    "AtomSpaceAdapter",
    "Package",
    "PackageRegistry",
]


@dataclass(frozen=True, slots=True)
class Alternatives:
    """Several answers, weighted. Collapsing is the caller's decision, not this one's."""

    items: tuple[tuple[Any, float], ...] = ()

    def best(self) -> Any | None:
        """The highest-weighted answer. Calling this IS the collapse."""
        return max(self.items, key=lambda pair: pair[1])[0] if self.items else None

    @property
    def ambiguous(self) -> bool:
        if len(self.items) < 2:
            return False
        ordered = sorted((w for _, w in self.items), reverse=True)
        return ordered[0] - ordered[1] < 1e-9

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[tuple[Any, float]]:
        return iter(self.items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": len(self.items),
            "ambiguous": self.ambiguous,
            "items": [{"value": repr(v), "weight": w} for v, w in self.items],
        }


@runtime_checkable
class Space(Protocol):
    """What every knowledge store looks like to a generic cognitive program."""

    name: str

    def add(self, key: str, value: Any, *, source: str = "") -> None: ...

    def remove(self, key: str) -> bool: ...

    def query(self, predicate: Callable[[str, Any], bool]) -> Alternatives: ...

    def iterate(self) -> Iterator[tuple[str, Any]]: ...

    def provenance(self, key: str) -> frozenset[str]: ...

    def attention(self, key: str) -> float: ...


@dataclass
class DictSpace:
    """The reference implementation, and the one tests run against."""

    name: str
    _items: dict[str, Any] = field(default_factory=dict)
    _sources: dict[str, set[str]] = field(default_factory=dict)
    _attention: dict[str, float] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def add(self, key: str, value: Any, *, source: str = "") -> None:
        with self._lock:
            self._items[key] = value
            if source:
                self._sources.setdefault(key, set()).add(source)
            self._attention[key] = self._attention.get(key, 0.0) + 1.0

    def remove(self, key: str) -> bool:
        with self._lock:
            existed = key in self._items
            self._items.pop(key, None)
            self._sources.pop(key, None)
            self._attention.pop(key, None)
            return existed

    def query(self, predicate: Callable[[str, Any], bool]) -> Alternatives:
        with self._lock:
            hits = [(v, self._attention.get(k, 0.0)) for k, v in self._items.items()
                    if predicate(k, v)]
        return Alternatives(tuple(sorted(hits, key=lambda pair: -pair[1])))

    def iterate(self) -> Iterator[tuple[str, Any]]:
        with self._lock:
            return iter(list(self._items.items()))

    def provenance(self, key: str) -> frozenset[str]:
        with self._lock:
            return frozenset(self._sources.get(key, ()))

    def attention(self, key: str) -> float:
        with self._lock:
            return self._attention.get(key, 0.0)


@dataclass
class AtomSpaceAdapter:
    """The metagraph, behind the same surface. Its internals are untouched."""

    space: Any
    name: str = "atomspace"

    def add(self, key: str, value: Any, *, source: str = "") -> None:
        from core.knowledge.atomspace import TruthValue, concept

        strength = float(value) if isinstance(value, (int, float)) else 1.0
        self.space.add(concept(key), TruthValue(strength, 1.0), source=source or None)

    def remove(self, key: str) -> bool:
        from core.knowledge.atomspace import concept

        return bool(self.space.forget(concept(key))) if hasattr(self.space, "forget") else False

    def query(self, predicate: Callable[[str, Any], bool]) -> Alternatives:
        hits = []
        for atom in self.space.atoms_of_type("Concept"):
            name = getattr(atom, "name", str(atom))
            tv = self.space.get_tv(atom)
            value = tv.strength if tv else 0.0
            if predicate(name, value):
                av = self.space.get_av(atom)
                hits.append((name, av.sti if av else 0.0))
        return Alternatives(tuple(sorted(hits, key=lambda pair: -pair[1])))

    def iterate(self) -> Iterator[tuple[str, Any]]:
        for atom in self.space.atoms_of_type("Concept"):
            tv = self.space.get_tv(atom)
            yield getattr(atom, "name", str(atom)), (tv.strength if tv else 0.0)

    def provenance(self, key: str) -> frozenset[str]:
        from core.knowledge.atomspace import concept

        return self.space.evidence_sources(concept(key))

    def attention(self, key: str) -> float:
        from core.knowledge.atomspace import concept

        av = self.space.get_av(concept(key))
        return av.sti if av else 0.0


@dataclass
class Package:
    """A named, versioned group of entries that can be removed exactly."""

    name: str
    version: str
    entries: dict[str, Any] = field(default_factory=dict)
    space: str = ""
    loaded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "version": self.version, "space": self.space,
            "entries": len(self.entries), "loaded": self.loaded,
        }


class PackageRegistry:
    """Load and unload rule packages, exactly, so a change can be undone."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._packages: dict[tuple[str, str], Package] = {}

    def load(self, package: Package, space: Space) -> Package:
        """Install a package into a space, recording exactly what it added."""
        with self._lock:
            key = (package.name, package.version)
            if key in self._packages and self._packages[key].loaded:
                raise ValueError(f"{package.name} {package.version} is already loaded")
            for name, value in package.entries.items():
                space.add(name, value, source=f"package:{package.name}@{package.version}")
            package.loaded = True
            package.space = space.name
            self._packages[key] = package
            return package

    def unload(self, name: str, version: str, space: Space) -> dict[str, Any]:
        """Remove exactly what the package added, and say what was already gone."""
        with self._lock:
            package = self._packages.get((name, version))
            if package is None or not package.loaded:
                raise KeyError(f"{name} {version} is not loaded")
            removed = [k for k in package.entries if space.remove(k)]
            missing = [k for k in package.entries if k not in removed]
            package.loaded = False
            return {
                "package": name, "version": version,
                "removed": sorted(removed), "already_gone": sorted(missing),
                "clean": not missing,
            }

    def loaded(self) -> list[Package]:
        with self._lock:
            return [p for p in self._packages.values() if p.loaded]

    def report(self) -> dict[str, Any]:
        with self._lock:
            packages = list(self._packages.values())
        return {
            "packages": len(packages),
            "loaded": [p.to_dict() for p in packages if p.loaded],
            "available": [p.to_dict() for p in packages if not p.loaded],
        }
