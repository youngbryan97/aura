"""core/cognition/autodoc.py — a name for a learned abstraction, kept only if it helps.

Compression produces ``f1``, ``f2``, ``f17``. Those are correct and unreadable,
and unreadable matters for two reasons: a person cannot review a library they
cannot read, and a model asked to compose from it has nothing to go on but the
body.

Naming is easy to do and easy to fool yourself about. A generated name always
looks better than ``f17``, and looking better is not the claim. The claim is
that the name improves retrieval or synthesis, and :class:`NamingTrial` is the
A/B that settles it: the same tasks, the same library, once with names and once
without, scored on whether the right abstraction was found.

A name that does not win is discarded and the abstraction keeps its symbol.
That is the whole discipline, and it is why this module holds a trial rather
than a namer.

What makes a name honest
------------------------
:func:`describe` builds a name from the abstraction's own body and the contexts
it appears in - the operators it composes and the families that use it. It does
not invent a purpose. "map-then-double over arith and draw" is a description; a
namer that produced "normalise inputs" would be asserting an intent the
abstraction does not have, and a reader would then compose against the
assertion rather than the code.
"""

from __future__ import annotations

from core.runtime.lockdep import checked_lock
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.cognition.library_compression import Abstraction

__all__ = ["Naming", "NamingTrial", "AutoDoc", "describe"]


def _operators(expression: Any) -> list[str]:
    if not isinstance(expression, tuple) or not expression:
        return []
    out = []
    head = expression[0]
    if isinstance(head, str):
        out.append(head)
    for part in expression:
        out.extend(_operators(part))
    return out


def describe(abstraction: Abstraction) -> str:
    """A name built from what the abstraction is, not from what it might be for.

    Composed of the operators it applies, in order, and the families it recurs
    in. It reads awkwardly on purpose: a fluent name asserts an intent, and a
    reader who believes the intent composes against it rather than against the
    body.
    """
    operators = []
    for operator in _operators(abstraction.body):
        if operator not in operators:
            operators.append(operator)
    core = "-then-".join(operators[:4]) or abstraction.name
    families = "/".join(sorted(abstraction.families)[:3])
    return f"{core} over {families}" if families else core


@dataclass(frozen=True, slots=True)
class Naming:
    """One abstraction's symbol and its description."""

    symbol: str
    description: str
    #: Kept only after the trial says it helped.
    adopted: bool = False

    def label(self) -> str:
        return self.description if self.adopted else self.symbol


@dataclass(frozen=True, slots=True)
class NamingTrial:
    """The A/B that decides whether descriptions are worth carrying."""

    with_names_found: int
    without_names_found: int
    tasks: int

    @property
    def delta(self) -> float:
        return (self.with_names_found - self.without_names_found) / self.tasks if self.tasks else 0.0

    @property
    def helps(self) -> bool:
        return self.with_names_found > self.without_names_found

    def to_dict(self) -> dict[str, Any]:
        return {
            "tasks": self.tasks,
            "with_names_found": self.with_names_found,
            "without_names_found": self.without_names_found,
            "delta": self.delta,
            "helps": self.helps,
            "verdict": (
                "descriptions improve retrieval and are kept"
                if self.helps
                else "descriptions did not improve retrieval; the library keeps its symbols"
            ),
        }


class AutoDoc:
    """Describe a library, then find out whether the descriptions earned it."""

    def __init__(self) -> None:
        self._lock = checked_lock("core.cognition.autodoc.AutoDoc", reentrant=True)
        self._namings: dict[str, Naming] = {}
        self._trial: NamingTrial | None = None

    def propose(self, library: Sequence[Abstraction]) -> dict[str, Naming]:
        with self._lock:
            for abstraction in library:
                self._namings[abstraction.name] = Naming(
                    symbol=abstraction.name, description=describe(abstraction)
                )
            return dict(self._namings)

    def trial(
        self,
        tasks: Sequence[tuple[str, str]],
        library: Sequence[Abstraction],
        retrieve: Callable[[str, Sequence[str]], str | None],
    ) -> NamingTrial:
        """Run retrieval twice over the same tasks and library, names on and off.

        ``retrieve(query, labels)`` returns the label it picked. Injecting it
        keeps the trial honest about what is being compared: only the labels
        differ between the two arms.
        """
        with self._lock:
            namings = dict(self._namings)
        symbols = [a.name for a in library]
        described = [namings[a.name].description if a.name in namings else a.name for a in library]
        by_description = {
            namings[a.name].description: a.name for a in library if a.name in namings
        }

        with_names = without_names = 0
        for query, expected in tasks:
            picked_symbol = retrieve(query, symbols)
            without_names += 1 if picked_symbol == expected else 0
            picked_described = retrieve(query, described)
            resolved = by_description.get(picked_described or "", picked_described)
            with_names += 1 if resolved == expected else 0

        result = NamingTrial(with_names, without_names, len(tasks))
        with self._lock:
            self._trial = result
            if result.helps:
                self._namings = {
                    k: Naming(v.symbol, v.description, adopted=True)
                    for k, v in self._namings.items()
                }
        return result

    def label(self, symbol: str) -> str:
        with self._lock:
            naming = self._namings.get(symbol)
            return naming.label() if naming else symbol

    def report(self) -> dict[str, Any]:
        with self._lock:
            return {
                "described": len(self._namings),
                "adopted": sum(1 for n in self._namings.values() if n.adopted),
                "trial": self._trial.to_dict() if self._trial else None,
                "labels": {k: v.label() for k, v in sorted(self._namings.items())},
            }
