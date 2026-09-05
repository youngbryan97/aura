"""core/cognition/library_compression.py — refactor the whole library, not one primitive.

Aura's representational growth adds. ``widening_the_language`` proposes a term,
``what_it_costs_to_say`` prices it, ``an_ecology_of_words`` retires it when it
stops paying. Each decision is local: does this one addition help. What nobody
asks is the global question - given every solution she has produced, what is
the smallest library that expresses all of them?

That question has a different answer. A subexpression appearing twice is not
worth naming; the same subexpression appearing in forty solutions across six
task families is the most valuable thing in the corpus, and no local rule can
see it because each of the forty looks unremarkable on its own.

The objective
-------------
Minimum description length over the corpus::

    cost = |library| + sum(|solution| rewritten against the library)

Naming an abstraction adds its own definition to the library and shortens every
solution that uses it. The abstraction pays when the shortening exceeds the
definition, and that arithmetic is the whole selection rule - no threshold on
frequency, no minimum size.

What it refuses
---------------
* **An abstraction used once.** It cannot shorten the corpus by more than its
  own definition costs, so MDL rejects it without a special case.
* **An abstraction that only appears in one family.** It is admitted, and
  flagged: a pattern that recurs inside one domain may be structure or may be
  one solver's habit, and the ``families`` count is what tells them apart
  later.
* **A rewrite that changes what a solution computes.** Every candidate rewrite
  is checked by re-evaluating the solution, and one that changes the result is
  discarded however much it would have compressed.

Iteration matters
-----------------
Compression is run to a fixed point. An abstraction extracted in round one
changes what the solutions look like, which exposes shared structure in round
two that was invisible before - that is where multi-layer abstractions come
from, rather than from a depth parameter.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from core.runtime.lockdep import checked_lock

__all__ = [
    "Expression",
    "Abstraction",
    "CompressionRound",
    "LibraryCompressor",
    "size",
    "subexpressions",
]

Expression = tuple  # ("op", arg, arg, ...) where an arg is an Expression or a leaf

#: What a reference to a library abstraction costs in the rewritten solution.
#: A call is written ``(name,)``, which is two symbols, and getting this wrong
#: is not a rounding error: at a cost of one, every size-two pattern appears to
#: pay, the compressor renames its own references forever, and the corpus grows
#: while the arithmetic reports a saving.
REFERENCE_COST = 2


def size(expression: Any) -> int:
    """Description length in symbols. A leaf is one, a node is one plus its parts."""
    if not isinstance(expression, tuple):
        return 1
    return 1 + sum(size(part) for part in expression)


def subexpressions(expression: Any, *, min_size: int = REFERENCE_COST + 1) -> list[Expression]:
    """Every subexpression big enough that naming it could possibly pay.

    Nothing smaller than a reference can shorten anything, so the floor is
    ``REFERENCE_COST + 1`` rather than an arbitrary two.
    """
    out: list[Expression] = []
    if not isinstance(expression, tuple):
        return out
    if size(expression) >= min_size:
        out.append(expression)
    for part in expression:
        out.extend(subexpressions(part, min_size=min_size))
    return out


def _substitute(expression: Any, pattern: Expression, name: str) -> Any:
    if expression == pattern:
        return (name,)
    if not isinstance(expression, tuple):
        return expression
    return tuple(_substitute(part, pattern, name) for part in expression)


@dataclass(frozen=True, slots=True)
class Abstraction:
    """One named piece of shared structure, and what naming it bought."""

    name: str
    body: Expression
    uses: int
    families: frozenset[str]
    saved: int
    round_found: int = 1

    @property
    def pays(self) -> bool:
        return self.saved > 0

    @property
    def cross_domain(self) -> bool:
        """Whether it recurs across families or is one solver's habit."""
        return len(self.families) > 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "body": self.body,
            "size": size(self.body),
            "uses": self.uses,
            "families": sorted(self.families),
            "cross_domain": self.cross_domain,
            "saved": self.saved,
            "round_found": self.round_found,
        }


@dataclass(frozen=True, slots=True)
class CompressionRound:
    """One pass over the corpus."""

    index: int
    extracted: tuple[Abstraction, ...]
    corpus_size_before: int
    corpus_size_after: int
    rejected_for_meaning: tuple[str, ...] = ()

    @property
    def saved(self) -> int:
        return self.corpus_size_before - self.corpus_size_after

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.index,
            "extracted": [a.to_dict() for a in self.extracted],
            "corpus_size_before": self.corpus_size_before,
            "corpus_size_after": self.corpus_size_after,
            "saved": self.saved,
            "rejected_for_meaning": list(self.rejected_for_meaning),
        }


class LibraryCompressor:
    """Compress a corpus of solutions into the smallest library that expresses it."""

    def __init__(
        self,
        *,
        evaluate: Callable[[Any], Any] | None = None,
        min_abstraction_size: int = REFERENCE_COST + 1,
    ) -> None:
        self._lock = checked_lock("core.cognition.library_compression.LibraryCompressor", reentrant=True)
        self._solutions: dict[str, tuple[Expression, str]] = {}
        self._library: dict[str, Abstraction] = {}
        self._rounds: list[CompressionRound] = []
        self._evaluate = evaluate
        self._min_size = int(min_abstraction_size)
        self._counter = 0

    def add_solution(self, key: str, expression: Expression, *, family: str = "") -> None:
        with self._lock:
            self._solutions[key] = (expression, family)

    def corpus_size(self) -> int:
        with self._lock:
            library = sum(size(a.body) for a in self._library.values())
            solutions = sum(size(e) for e, _ in self._solutions.values())
            return library + solutions

    def _candidates_locked(self) -> dict[Expression, tuple[int, set[str]]]:
        counts: dict[Expression, tuple[int, set[str]]] = {}
        for expression, family in self._solutions.values():
            for sub in subexpressions(expression, min_size=self._min_size):
                uses, families = counts.get(sub, (0, set()))
                counts[sub] = (uses + 1, families | {family or "unspecified"})
        return counts

    def _meaning_preserved(self, before: Any, after: Any) -> bool:
        """Whether a rewrite changed what the solution computes."""
        if self._evaluate is None:
            return True
        try:
            return self._evaluate(before) == self._evaluate(after)
        except Exception:  # noqa: BLE001 - a rewrite that cannot be evaluated is not safe
            return False

    def compress_once(self) -> CompressionRound:
        """Extract the single best-paying abstraction, if any pays."""
        with self._lock:
            before = self.corpus_size()
            candidates = self._candidates_locked()
            best: tuple[int, Expression, int, set[str]] | None = None
            for pattern, (uses, families) in candidates.items():
                if uses < 2:
                    continue
                # Each use collapses to a reference; the definition is paid once.
                saved = uses * (size(pattern) - REFERENCE_COST) - size(pattern)
                if saved <= 0:
                    continue
                if best is None or saved > best[0]:
                    best = (saved, pattern, uses, families)

            if best is None:
                round_result = CompressionRound(
                    index=len(self._rounds) + 1, extracted=(),
                    corpus_size_before=before, corpus_size_after=before,
                )
                self._rounds.append(round_result)
                return round_result

            saved, pattern, uses, families = best
            self._counter += 1
            name = f"f{self._counter}"

            rejected: list[str] = []
            rewritten: dict[str, tuple[Expression, str]] = {}
            for key, (expression, family) in self._solutions.items():
                candidate = _substitute(expression, pattern, name)
                if candidate != expression and not self._meaning_preserved(expression, candidate):
                    rejected.append(key)
                    rewritten[key] = (expression, family)
                    continue
                rewritten[key] = (candidate, family)

            abstraction = Abstraction(
                name=name, body=pattern, uses=uses - len(rejected),
                families=frozenset(families), saved=saved,
                round_found=len(self._rounds) + 1,
            )
            if abstraction.uses < 2:
                # Every use was rejected for meaning; naming it costs and buys nothing.
                round_result = CompressionRound(
                    index=len(self._rounds) + 1, extracted=(),
                    corpus_size_before=before, corpus_size_after=before,
                    rejected_for_meaning=tuple(rejected),
                )
                self._rounds.append(round_result)
                return round_result

            self._solutions = rewritten
            self._library[name] = abstraction
            after = self.corpus_size()
            round_result = CompressionRound(
                index=len(self._rounds) + 1, extracted=(abstraction,),
                corpus_size_before=before, corpus_size_after=after,
                rejected_for_meaning=tuple(rejected),
            )
            self._rounds.append(round_result)
            return round_result

    def compress(self, *, max_rounds: int = 20) -> list[CompressionRound]:
        """Run to a fixed point. Later rounds see structure earlier ones exposed."""
        rounds = []
        for _ in range(max_rounds):
            result = self.compress_once()
            rounds.append(result)
            if not result.extracted:
                break
        return rounds

    def library(self) -> list[Abstraction]:
        with self._lock:
            return sorted(self._library.values(), key=lambda a: (a.round_found, a.name))

    def solutions(self) -> dict[str, Expression]:
        with self._lock:
            return {k: v[0] for k, v in self._solutions.items()}

    def report(self) -> dict[str, Any]:
        with self._lock:
            library = list(self._library.values())
            rounds = list(self._rounds)
        layers = {a.round_found for a in library}
        return {
            "solutions": len(self._solutions),
            "library": [a.to_dict() for a in library],
            "abstractions": len(library),
            "layers": len(layers),
            "cross_domain_abstractions": sorted(a.name for a in library if a.cross_domain),
            "single_family_abstractions": sorted(a.name for a in library if not a.cross_domain),
            "total_saved": sum(r.saved for r in rounds),
            "rounds": [r.to_dict() for r in rounds],
        }
