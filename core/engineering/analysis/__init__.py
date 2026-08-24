"""Every number a drawing is allowed to show, and where it came from.

A :class:`Finding` is one computed quantity together with the formula that
produced it, the inputs that went in, the reference the formula came from,
and a sentence a reader without an engineering degree can act on. Nothing
else may reach a drawing. A callout that cannot name the finding behind its
number does not render, which is the whole reason this package exists: the
cortex can propose a design, and it cannot state a result.

Analyses are registered functions over a :class:`~core.engineering.model.Design`.
Each declares the domains it applies to and returns whatever it can compute
from what the model holds. An analysis that lacks an input says so and
returns nothing rather than assuming a value, because an assumed input is
the same failure as a generated number with one more step in front of it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from core.engineering.units import Q, Quantity

__all__ = [
    "Finding",
    "Analysis",
    "register",
    "ANALYSES",
    "run_analyses",
    "findings_by_id",
    "AnalysisInputMissing",
    "quantify",
]


class AnalysisInputMissing(LookupError):
    """An analysis needed a value the model does not carry."""


@dataclass(frozen=True, slots=True)
class Finding:
    """One computed result, with everything needed to argue with it."""

    id: str
    name: str
    value: Quantity
    #: The formula as written in a textbook, so the arithmetic can be redone.
    formula: str = ""
    #: Every input by name, so the arithmetic can be redone with real numbers.
    inputs: dict[str, Quantity] = field(default_factory=dict)
    #: Where the formula comes from.
    method: str = ""
    #: What this means, for a reader who is not an engineer.
    plain: str = ""
    #: Which part, connection or subsystem this is about.
    subject: str = ""
    #: pass, fail, watch or none, when the finding settles something.
    verdict: str = ""
    #: How much room is left, as a ratio, when the finding has a limit.
    margin: float | None = None
    #: What to do about it, when the verdict is not a pass.
    advice: str = ""
    #: Assumptions the number rests on, named rather than buried.
    assumptions: tuple[str, ...] = ()

    def substituted(self) -> str:
        """The formula with the actual numbers in it."""
        if not self.inputs:
            return self.formula
        pieces = ", ".join(f"{k} = {v.text()}" for k, v in self.inputs.items())
        return f"{self.formula}   with {pieces}"

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "value": self.value.to_dict(),
            "formula": self.formula,
            "substituted": self.substituted(),
            "inputs": {k: v.to_dict() for k, v in self.inputs.items()},
            "method": self.method,
            "plain": self.plain,
            "subject": self.subject,
            "assumptions": list(self.assumptions),
        }
        if self.verdict:
            out["verdict"] = self.verdict
        if self.margin is not None:
            out["margin"] = self.margin
        if self.advice:
            out["advice"] = self.advice
        return out


@dataclass(frozen=True, slots=True)
class Analysis:
    """One registered calculation over a design."""

    key: str
    name: str
    #: What this tells you, for the panel that lists available analyses.
    question: str
    run: Callable[[Any], Iterable[Finding]]
    domains: tuple[str, ...] = ()
    discipline: str = "general"

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "question": self.question,
            "domains": list(self.domains),
            "discipline": self.discipline,
        }


ANALYSES: dict[str, Analysis] = {}


def register(
    key: str,
    name: str,
    question: str,
    *,
    domains: tuple[str, ...] = (),
    discipline: str = "general",
) -> Callable[[Callable[[Any], Iterable[Finding]]], Callable[[Any], Iterable[Finding]]]:
    """Register one analysis so the runner and the panel both find it."""

    def decorate(
        function: Callable[[Any], Iterable[Finding]],
    ) -> Callable[[Any], Iterable[Finding]]:
        ANALYSES[key] = Analysis(
            key=key,
            name=name,
            question=question,
            run=function,
            domains=domains,
            discipline=discipline,
        )
        return function

    return decorate


def quantify(value: Any, unit: str = "") -> Quantity:
    """Coerce an input into a quantity, refusing a bare number with no unit."""
    if isinstance(value, Quantity):
        return value
    if value is None:
        raise AnalysisInputMissing("no value")
    return Q(value, unit)


def run_analyses(design: Any, *, only: tuple[str, ...] = ()) -> tuple[Finding, ...]:
    """Run every registered analysis that has what it needs.

    An analysis that raises :class:`AnalysisInputMissing` is skipped, because
    a design in progress is missing inputs by definition. Any other exception
    is a defect in the analysis and is recorded as a degradation rather than
    swallowed, so a broken calculation is visible instead of quietly absent.
    """
    from core.runtime.errors import record_degradation

    wanted = set(only) if only else None
    results: list[Finding] = []
    for key, analysis in ANALYSES.items():
        if wanted is not None and key not in wanted:
            continue
        try:
            produced = list(analysis.run(design))
        except AnalysisInputMissing:
            continue
        except (ValueError, TypeError, KeyError, ZeroDivisionError, ArithmeticError) as exc:
            record_degradation(
                "engineering.analysis",
                exc,
                action=f"analysis {key} failed on design {getattr(design, 'name', '?')!r}",
            )
            continue
        results.extend(produced)
    # A stable order keeps two runs of the same design byte-identical, which
    # is what lets a drawing be compared with the one from last week.
    return tuple(sorted(results, key=lambda f: (f.subject, f.id)))


def findings_by_id(findings: Iterable[Finding]) -> dict[str, Finding]:
    return {finding.id: finding for finding in findings}


# Importing the modules is what registers them. Kept at the bottom so the
# decorator and the Finding type exist before any of them are read.
from core.engineering.analysis import (  # noqa: E402,F401
    conservation,
    controls,
    electrical,
    fluids,
    margins,
    mass,
    motion,
    process,
    structures,
    thermal,
)
