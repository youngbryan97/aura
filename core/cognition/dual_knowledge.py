"""core/cognition/dual_knowledge.py — the same skill in three forms, and the cost of each crossing.

Aura holds knowledge implicitly (RLC state, world-model latents, learned
tissue), explicitly (AtomSpace atoms, rules, procedures) and neurally
(distilled weights). All three are real and none of them can become another
without somebody writing a bespoke conversion. So a skill learned by practice
stays tacit and cannot be inspected, a rule that was reasoned out stays slow
forever, and a distilled shortcut cannot explain itself.

Three conversions, and each is only trustworthy with the same thing attached:

* **Extract** — implicit to explicit. Read a rule off a learned policy. Cheap
  to do badly: a rule that describes 60% of a policy's behaviour and is
  presented as *the* policy is worse than no rule, because it will be trusted
  in the 40%.
* **Distil** — explicit to neural. Compile a validated procedure into something
  fast. The risk is silent behaviour change, which is invisible precisely
  where the fast path is used most.
* **Explain** — neural to explicit. Say what a distilled thing does. The risk
  is a plausible account of something else entirely.

The thing attached is :class:`Equivalence`: agreement measured on held-out
cases, with the disagreements kept. A conversion whose agreement is below its
declared tolerance does not install; :meth:`DualKnowledge.convert` returns the
failure and the source form stays canonical. That refusal is the module.

Why disagreements are kept
--------------------------
An agreement rate is a summary and the interesting part is which cases fell
outside it. ``Equivalence.disagreements`` holds them, so a failed conversion is
a specification for the next attempt rather than a number that says try again.

Credit
------
:class:`CreditAssignment` answers card 217. When a chain of substrates produces
a success, each form's contribution is estimated by what its removal costs -
the same logic the lesion registry uses, applied to a conversion chain rather
than to an organ. Credit routed to a form that was not on the path is the
defect this prevents, and it is the reason contribution is measured by removal
rather than by presence.
"""

from __future__ import annotations

from core.runtime.lockdep import checked_lock
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "Form",
    "Equivalence",
    "ConversionResult",
    "DualKnowledge",
    "KnowledgeRegistry",
    "CreditAssignment",
    "get_knowledge_registry",
    "reset_knowledge_registry_for_test",
]


class Form(StrEnum):
    """How a piece of knowledge is held."""

    IMPLICIT = "implicit"    # a learned policy, a latent, tissue
    EXPLICIT = "explicit"    # a rule, an atom, a typed procedure
    NEURAL = "neural"        # distilled weights, a fast path

    @property
    def can_be_inspected(self) -> bool:
        return self is Form.EXPLICIT

    @property
    def is_fast(self) -> bool:
        return self in (Form.IMPLICIT, Form.NEURAL)


@dataclass(frozen=True, slots=True)
class Equivalence:
    """How far two forms of one skill actually agree."""

    cases: int
    agreements: int
    disagreements: tuple[tuple[Any, Any, Any], ...] = ()
    tolerance: float = 0.95
    measured_on: str = "held_out"

    @property
    def agreement(self) -> float:
        return self.agreements / self.cases if self.cases else 0.0

    @property
    def passes(self) -> bool:
        return self.cases > 0 and self.agreement >= self.tolerance

    def to_dict(self) -> dict[str, Any]:
        return {
            "cases": self.cases,
            "agreements": self.agreements,
            "agreement": self.agreement,
            "tolerance": self.tolerance,
            "passes": self.passes,
            "measured_on": self.measured_on,
            "disagreements": [
                {"input": repr(i), "source": repr(s), "target": repr(t)}
                for i, s, t in self.disagreements[:20]
            ],
        }


@dataclass(frozen=True, slots=True)
class ConversionResult:
    """One attempt to move a skill between forms."""

    skill: str
    source: Form
    target: Form
    installed: bool
    equivalence: Equivalence
    reason: str = ""
    at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill,
            "source": self.source.value,
            "target": self.target.value,
            "installed": self.installed,
            "reason": self.reason,
            "equivalence": self.equivalence.to_dict(),
        }


@dataclass
class DualKnowledge:
    """One skill, in every form it currently has, with the crossings measured."""

    skill: str
    forms: dict[Form, Any] = field(default_factory=dict)
    equivalences: dict[tuple[Form, Form], Equivalence] = field(default_factory=dict)
    #: Measured benefit of each form, in whatever unit the caller uses.
    benefit: dict[Form, float] = field(default_factory=dict)

    @property
    def reach(self) -> int:
        return len(self.forms)

    @property
    def travelled_all_three(self) -> bool:
        """Card 183's bar: the skill exists in all three forms, each verified."""
        if set(self.forms) != set(Form):
            return False
        return all(e.passes for e in self.equivalences.values())

    def each_form_earns_its_place(self) -> dict[str, Any]:
        """Whether every form measurably adds something.

        A form present with no measured benefit is a conversion that was made
        because it could be, and it costs storage, staleness and one more thing
        that can disagree.
        """
        idle = [f.value for f in self.forms if self.benefit.get(f, 0.0) <= 0.0]
        return {
            "forms": [f.value for f in self.forms],
            "benefit": {f.value: v for f, v in self.benefit.items()},
            "forms_earning_their_place": len(self.forms) - len(idle),
            "idle_forms": idle,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill,
            "forms": [f.value for f in self.forms],
            "reach": self.reach,
            "travelled_all_three": self.travelled_all_three,
            "equivalences": {
                f"{a.value}->{b.value}": e.to_dict() for (a, b), e in self.equivalences.items()
            },
            **self.each_form_earns_its_place(),
        }


def measure_equivalence(
    source: Callable[[Any], Any],
    target: Callable[[Any], Any],
    cases: Sequence[Any],
    *,
    tolerance: float = 0.95,
    compare: Callable[[Any, Any], bool] | None = None,
    measured_on: str = "held_out",
) -> Equivalence:
    """Run both forms over the same cases and record where they differ."""
    same = compare or (lambda a, b: a == b)
    agreements = 0
    disagreements: list[tuple[Any, Any, Any]] = []
    for case in cases:
        try:
            left = source(case)
        except Exception as exc:  # noqa: BLE001 - a raising source is a disagreement
            left = f"<raised {type(exc).__name__}>"
        try:
            right = target(case)
        except Exception as exc:  # noqa: BLE001
            right = f"<raised {type(exc).__name__}>"
        if same(left, right):
            agreements += 1
        else:
            disagreements.append((case, left, right))
    return Equivalence(
        cases=len(cases),
        agreements=agreements,
        disagreements=tuple(disagreements),
        tolerance=tolerance,
        measured_on=measured_on,
    )


class KnowledgeRegistry:
    """Every skill that exists in more than one form, and what the crossings cost."""

    def __init__(self) -> None:
        self._lock = checked_lock("core.cognition.dual_knowledge.KnowledgeRegistry", reentrant=True)
        self._skills: dict[str, DualKnowledge] = {}
        self._conversions: list[ConversionResult] = []
        self._max_history = 2048

    def hold(self, skill: str, form: Form, implementation: Any, *, benefit: float = 0.0) -> DualKnowledge:
        with self._lock:
            entry = self._skills.setdefault(skill, DualKnowledge(skill=skill))
            entry.forms[form] = implementation
            if benefit:
                entry.benefit[form] = benefit
            return entry

    def convert(
        self,
        skill: str,
        *,
        source: Form,
        target: Form,
        build: Callable[[Any], Any],
        cases: Sequence[Any],
        tolerance: float = 0.95,
        compare: Callable[[Any, Any], bool] | None = None,
        benefit: float = 0.0,
    ) -> ConversionResult:
        """Build the target form and install it only if it agrees with the source.

        The source stays canonical when the conversion fails. A fast path that
        silently disagrees is worse than no fast path, because it is used most
        exactly where nobody is checking.
        """
        with self._lock:
            entry = self._skills.get(skill)
            source_impl = entry.forms.get(source) if entry else None
        if source_impl is None:
            result = ConversionResult(
                skill, source, target, False,
                Equivalence(cases=0, agreements=0, tolerance=tolerance),
                reason=f"{skill!r} has no {source.value} form to convert from",
            )
            self._record(result)
            return result

        try:
            built = build(source_impl)
        except Exception as exc:  # noqa: BLE001
            result = ConversionResult(
                skill, source, target, False,
                Equivalence(cases=0, agreements=0, tolerance=tolerance),
                reason=f"conversion raised {type(exc).__name__}: {exc}",
            )
            self._record(result)
            return result

        equivalence = measure_equivalence(
            source_impl, built, cases, tolerance=tolerance, compare=compare
        )
        if not equivalence.passes:
            result = ConversionResult(
                skill, source, target, False, equivalence,
                reason=(
                    f"agreement {equivalence.agreement:.2%} below tolerance "
                    f"{tolerance:.2%}; {len(equivalence.disagreements)} case(s) differ"
                ),
            )
            self._record(result)
            return result

        with self._lock:
            entry = self._skills.setdefault(skill, DualKnowledge(skill=skill))
            entry.forms[target] = built
            entry.equivalences[(source, target)] = equivalence
            if benefit:
                entry.benefit[target] = benefit
        result = ConversionResult(skill, source, target, True, equivalence)
        self._record(result)
        return result

    def _record(self, result: ConversionResult) -> None:
        with self._lock:
            self._conversions.append(result)
            if len(self._conversions) > self._max_history:
                del self._conversions[: len(self._conversions) - self._max_history]

    def get(self, skill: str) -> DualKnowledge | None:
        with self._lock:
            return self._skills.get(skill)

    def report(self) -> dict[str, Any]:
        with self._lock:
            skills = list(self._skills.values())
            conversions = list(self._conversions)
        attempted = len(conversions)
        installed = sum(1 for c in conversions if c.installed)
        return {
            "skills": len(skills),
            "in_two_forms": sum(1 for s in skills if s.reach >= 2),
            "in_all_three": sum(1 for s in skills if s.travelled_all_three),
            "conversions_attempted": attempted,
            "conversions_installed": installed,
            "conversions_refused": attempted - installed,
            "refusal_reasons": [c.reason for c in conversions if not c.installed][-10:],
            "idle_forms": {
                s.skill: s.each_form_earns_its_place()["idle_forms"]
                for s in skills
                if s.each_form_earns_its_place()["idle_forms"]
            },
        }


class CreditAssignment:
    """Which substrate a success is owed to, measured by what removing it costs.

    Presence is not contribution. A form that was on the path and would not
    have been missed gets no credit here, which is the only way to stop the
    slowest and most visible component absorbing the credit for everything.
    """

    def __init__(self) -> None:
        self._lock = checked_lock("core.cognition.dual_knowledge.CreditAssignment", reentrant=True)
        self._credit: dict[str, dict[str, float]] = {}

    def assign(
        self, outcome: str, *, full_score: float, without: Mapping[str, float]
    ) -> dict[str, float]:
        """Split credit by each component's removal cost, normalised to the total.

        ``without[c]`` is the score with ``c`` removed. A component whose
        removal costs nothing gets zero, and when nothing costs anything the
        credit is empty rather than shared out.
        """
        deltas = {name: max(0.0, full_score - score) for name, score in without.items()}
        total = sum(deltas.values())
        share = {name: (delta / total if total > 0 else 0.0) for name, delta in deltas.items()}
        with self._lock:
            bucket = self._credit.setdefault(outcome, {})
            for name, value in share.items():
                bucket[name] = bucket.get(name, 0.0) + value
        return share

    def credit_for(self, outcome: str) -> dict[str, float]:
        with self._lock:
            return dict(self._credit.get(outcome, {}))

    def report(self) -> dict[str, Any]:
        with self._lock:
            outcomes = {k: dict(v) for k, v in self._credit.items()}
        totals: dict[str, float] = {}
        for shares in outcomes.values():
            for name, value in shares.items():
                totals[name] = totals.get(name, 0.0) + value
        return {
            "outcomes": len(outcomes),
            "by_component": dict(sorted(totals.items(), key=lambda kv: -kv[1])),
            "components_with_no_credit": sorted(k for k, v in totals.items() if v == 0.0),
        }


_lock = checked_lock("core.cognition.dual_knowledge.singleton")
_registry: KnowledgeRegistry | None = None


def get_knowledge_registry() -> KnowledgeRegistry:
    global _registry
    with _lock:
        if _registry is None:
            _registry = KnowledgeRegistry()
        return _registry


def reset_knowledge_registry_for_test() -> KnowledgeRegistry:
    global _registry
    with _lock:
        _registry = KnowledgeRegistry()
        return _registry
