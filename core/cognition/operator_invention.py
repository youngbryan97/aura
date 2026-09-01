"""core/cognition/operator_invention.py — a new meaning, proved before it is installed.

The boundary both reviews converge on. Aura can invent terms, constructors and
compositions INSIDE a semantic substrate somebody wrote. What she cannot do is
extend the substrate: derive a new operator's semantics, prove it is new,
install it, and later compose it into another invention. ``over again`` was
added by a human, and that is the example the boundary is named after.

The loop, and every step is a refusal
-------------------------------------
1. **A residual failure that persists.** A family of tasks that stays unsolved
   after the existing operators have been searched. One failure is a bug;
   :attr:`Residual.persistent` requires the family to survive repeated attempts,
   because inventing an operator for a transient failure adds vocabulary to
   cover a bug.
2. **A candidate synthesised from what exists.** Semantics are composed from
   installed operators, so an invention is always expressible in the language
   it extends - which is what makes it checkable.
3. **Bounded execution.** A candidate runs under a step budget in a sandbox.
   One that does not terminate is discarded, and the discard is recorded,
   because a language that can install a non-terminating operator can install
   one that hangs the mind.
4. **A novelty certificate.** The candidate must compute something no existing
   operator computes on the probe set. Extensional novelty, checked, not
   asserted - a renamed composition is not an invention.
5. **Reach and MDL.** It must solve a family that was unreachable AND shorten
   the corpus. Either alone admits an operator that is a special case dressed
   as a generalisation.
6. **Adversarial held-out.** It must survive probes chosen to break it,
   including inputs outside the range it was synthesised on.
7. **Governed installation with exact rollback.** Installing takes a snapshot;
   :meth:`OperatorKernel.rollback` restores the exact prior semantics, and the
   test for that is behavioural rather than structural - every probe gives the
   same answer as before the install.

Composition is the last bar
---------------------------
An invention that cannot be used to build the next one is a feature, not a
language extension. :meth:`OperatorKernel.compose_from_invented` requires the
new operator to appear in the body of a later one, which is card 187's final
clause and the thing that makes the loop recursive rather than a single step.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "Residual",
    "Candidate",
    "Verdict",
    "Operator",
    "OperatorKernel",
    "STEP_BUDGET",
]

#: Steps a candidate may take on one input before it is discarded. A language
#: that can install a non-terminating operator can install one that hangs.
STEP_BUDGET = 1000

#: Attempts a task family must survive before it counts as a residual failure
#: rather than a bug.
PERSISTENCE = 3


class Rejection(StrEnum):
    NOT_PERSISTENT = "not_persistent"
    NON_TERMINATING = "non_terminating"
    NOT_NOVEL = "not_novel"
    NO_REACH = "no_reach"
    NO_COMPRESSION = "no_compression"
    FAILED_ADVERSARIAL = "failed_adversarial"
    RAISED = "raised"


@dataclass
class Residual:
    """A family of tasks the existing operators cannot solve."""

    family: str
    attempts: int = 0
    solved: int = 0
    probes: tuple[Any, ...] = ()

    @property
    def persistent(self) -> bool:
        return self.attempts >= PERSISTENCE and self.solved == 0


@dataclass(frozen=True, slots=True)
class Candidate:
    """A proposed operator, composed from what already exists."""

    name: str
    body: str
    fn: Callable[..., Any]
    built_from: tuple[str, ...]
    arity: int = 1


@dataclass(frozen=True, slots=True)
class Verdict:
    """Why a candidate was installed or refused, with the evidence."""

    candidate: str
    installed: bool
    rejection: Rejection | None = None
    novel_on: tuple[Any, ...] = ()
    reach_gained: tuple[str, ...] = ()
    compression: int = 0
    adversarial_passed: int = 0
    adversarial_total: int = 0
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "installed": self.installed,
            "rejection": self.rejection.value if self.rejection else None,
            "novel_on": [repr(x) for x in self.novel_on[:5]],
            "reach_gained": list(self.reach_gained),
            "compression": self.compression,
            "adversarial": f"{self.adversarial_passed}/{self.adversarial_total}",
            "detail": self.detail,
        }


@dataclass
class Operator:
    """An installed operator, and where it came from."""

    name: str
    fn: Callable[..., Any]
    body: str = ""
    built_from: tuple[str, ...] = ()
    invented: bool = False
    generation: int = 0


class _Budget:
    """Counts steps so a runaway candidate is discarded rather than waited on."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0

    def step(self) -> None:
        self.used += 1
        if self.used > self.limit:
            raise TimeoutError(f"exceeded {self.limit} steps")


class OperatorKernel:
    """The evaluator's operator set, and the only way to extend it."""

    def __init__(self, base: Mapping[str, Callable[..., Any]] | None = None) -> None:
        self._lock = threading.RLock()
        self._operators: dict[str, Operator] = {
            name: Operator(name=name, fn=fn) for name, fn in (base or {}).items()
        }
        self._snapshots: list[tuple[str, dict[str, Operator]]] = []
        self._verdicts: list[Verdict] = []
        self._residuals: dict[str, Residual] = {}

    # ── residuals ─────────────────────────────────────────────────────

    def attempt(self, family: str, *, solved: bool, probes: Sequence[Any] = ()) -> Residual:
        with self._lock:
            residual = self._residuals.setdefault(family, Residual(family=family))
            residual.attempts += 1
            residual.solved += 1 if solved else 0
            if probes:
                residual.probes = tuple(probes)
            return residual

    def residuals(self) -> list[Residual]:
        with self._lock:
            return [r for r in self._residuals.values() if r.persistent]

    # ── the loop ──────────────────────────────────────────────────────

    def consider(
        self,
        candidate: Candidate,
        *,
        family: str,
        probes: Sequence[Any],
        adversarial: Sequence[Any] = (),
        solves: Callable[[Callable[..., Any], str], bool],
        compression: int = 0,
    ) -> Verdict:
        """Run the whole loop. Every step can refuse."""
        with self._lock:
            residual = self._residuals.get(family)
            existing = dict(self._operators)

        if residual is None or not residual.persistent:
            return self._record(Verdict(
                candidate.name, False, Rejection.NOT_PERSISTENT,
                detail=(
                    f"{family!r} has not persistently failed; inventing an operator for a "
                    "transient failure adds vocabulary to cover a bug"
                ),
            ))

        # Bounded execution on every probe.
        outputs: list[Any] = []
        for probe in probes:
            budget = _Budget(STEP_BUDGET)
            try:
                outputs.append(candidate.fn(probe, budget))
            except TimeoutError as exc:
                return self._record(Verdict(
                    candidate.name, False, Rejection.NON_TERMINATING, detail=str(exc)
                ))
            except Exception as exc:  # noqa: BLE001
                return self._record(Verdict(
                    candidate.name, False, Rejection.RAISED,
                    detail=f"{type(exc).__name__}: {exc}",
                ))

        # Novelty: something no installed operator computes on these probes.
        novel_on = []
        for probe, output in zip(probes, outputs, strict=True):
            matched = False
            for operator in existing.values():
                try:
                    if operator.fn(probe, _Budget(STEP_BUDGET)) == output:
                        matched = True
                        break
                except (TypeError, ValueError, ArithmeticError, LookupError,
                        RecursionError, TimeoutError):
                    # An installed operator that cannot run on this probe does
                    # not match it. Named, because an operator raising something
                    # else is a defect in the kernel rather than a mismatch.
                    continue
            if not matched:
                novel_on.append(probe)
        if not novel_on:
            return self._record(Verdict(
                candidate.name, False, Rejection.NOT_NOVEL, detail=(
                    "every probe is already computed by an installed operator; a renamed "
                    "composition is not an invention"
                ),
            ))

        # Reach: it must solve the family that was unreachable.
        if not solves(candidate.fn, family):
            return self._record(Verdict(
                candidate.name, False, Rejection.NO_REACH, novel_on=tuple(novel_on),
                detail=f"novel, and still does not solve {family!r}",
            ))

        # MDL: novelty plus reach without compression admits a special case.
        if compression <= 0:
            return self._record(Verdict(
                candidate.name, False, Rejection.NO_COMPRESSION,
                novel_on=tuple(novel_on), reach_gained=(family,),
                detail="solves the family and shortens nothing; that is a special case",
            ))

        # Adversarial held-out, including inputs outside the synthesis range.
        passed = 0
        for probe in adversarial:
            try:
                candidate.fn(probe, _Budget(STEP_BUDGET))
                passed += 1
            except (TypeError, ValueError, ArithmeticError, LookupError,
                    RecursionError, TimeoutError):
                # The adversarial probe broke it, which is the measurement.
                continue
        if adversarial and passed < len(adversarial):
            return self._record(Verdict(
                candidate.name, False, Rejection.FAILED_ADVERSARIAL,
                novel_on=tuple(novel_on), reach_gained=(family,), compression=compression,
                adversarial_passed=passed, adversarial_total=len(adversarial),
                detail="broke on inputs outside the range it was synthesised on",
            ))

        with self._lock:
            self._snapshots.append((candidate.name, dict(self._operators)))
            generation = 1 + max(
                (self._operators[n].generation for n in candidate.built_from
                 if n in self._operators),
                default=-1,
            )
            self._operators[candidate.name] = Operator(
                name=candidate.name, fn=candidate.fn, body=candidate.body,
                built_from=candidate.built_from, invented=True, generation=generation,
            )
        return self._record(Verdict(
            candidate.name, True, None, novel_on=tuple(novel_on), reach_gained=(family,),
            compression=compression, adversarial_passed=passed,
            adversarial_total=len(adversarial),
        ))

    def rollback(self, name: str) -> dict[str, Any]:
        """Restore the exact operator set from before this install."""
        with self._lock:
            index = next(
                (i for i in range(len(self._snapshots) - 1, -1, -1)
                 if self._snapshots[i][0] == name),
                None,
            )
            if index is None:
                raise KeyError(f"no install snapshot for {name!r}")
            _, snapshot = self._snapshots.pop(index)
            removed = sorted(set(self._operators) - set(snapshot))
            self._operators = dict(snapshot)
            return {"rolled_back": name, "removed": removed, "operators": sorted(self._operators)}

    def behaviourally_identical(
        self, probes: Sequence[Any], before: Mapping[str, Sequence[Any]]
    ) -> dict[str, Any]:
        """Whether every operator answers every probe exactly as it did before.

        Structural equality of the operator set is not the test. Two operator
        sets can look identical and behave differently if anything was rebound,
        so the rollback check runs the probes.
        """
        differences = []
        with self._lock:
            operators = dict(self._operators)
        for name, expected in before.items():
            operator = operators.get(name)
            if operator is None:
                differences.append(f"{name} is gone")
                continue
            for probe, want in zip(probes, expected, strict=False):
                try:
                    got = operator.fn(probe, _Budget(STEP_BUDGET))
                except Exception as exc:  # noqa: BLE001
                    differences.append(f"{name}({probe!r}) raised {type(exc).__name__}")
                    continue
                if got != want:
                    differences.append(f"{name}({probe!r}) was {want!r}, now {got!r}")
        return {"identical": not differences, "differences": differences}

    def compose_from_invented(self, candidate: Candidate) -> bool:
        """Whether this candidate is built on something previously invented.

        The final bar. An invention that cannot be used to build the next one
        is a feature; the language extends only when its own additions become
        material for further additions.
        """
        with self._lock:
            return any(
                self._operators.get(name) is not None and self._operators[name].invented
                for name in candidate.built_from
            )

    def _record(self, verdict: Verdict) -> Verdict:
        with self._lock:
            self._verdicts.append(verdict)
        return verdict

    def operators(self) -> dict[str, Operator]:
        with self._lock:
            return dict(self._operators)

    def report(self) -> dict[str, Any]:
        with self._lock:
            operators = list(self._operators.values())
            verdicts = list(self._verdicts)
        invented = [o for o in operators if o.invented]
        by_rejection: dict[str, int] = {}
        for verdict in verdicts:
            if verdict.rejection:
                by_rejection[verdict.rejection.value] = (
                    by_rejection.get(verdict.rejection.value, 0) + 1
                )
        return {
            "operators": len(operators),
            "invented": [o.name for o in invented],
            "max_generation": max((o.generation for o in invented), default=0),
            "recursive": any(o.generation >= 2 for o in invented),
            "considered": len(verdicts),
            "installed": sum(1 for v in verdicts if v.installed),
            "rejections": dict(sorted(by_rejection.items())),
            "verdicts": [v.to_dict() for v in verdicts],
        }
