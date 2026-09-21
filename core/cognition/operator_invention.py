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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from core.runtime.lockdep import checked_lock

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
    NAME_CONFLICT = "name_conflict"
    STATE_CHANGED = "state_changed"
    INVALID_LINEAGE = "invalid_lineage"


@dataclass
class Residual:
    """A family of tasks the existing operators cannot solve."""

    family: str
    attempts: int = 0
    solved: int = 0
    probes: tuple[Any, ...] = ()
    cases: tuple[tuple[Any, Any], ...] = ()

    @property
    def persistent(self) -> bool:
        return self.attempts >= PERSISTENCE and self.solved == 0


@dataclass(frozen=True, slots=True)
class Candidate:
    """A proposed operator, as a term rather than as a function.

    ``term`` is a floor term — see
    :mod:`core.cognition.the_floor_she_stands_on` — and it is what the kernel
    runs. A candidate arriving as a Python function is a candidate whose
    semantics somebody else wrote, which is the thing this whole file exists
    to be the last step of rather than the first; the kernel could check such
    a candidate but it could never have produced one.

    ``fn`` is kept for the callers that still pass one, and every one of them
    is a test. A candidate with a term uses the term; a candidate with neither
    is refused before anything is run.
    """

    name: str
    body: str
    fn: Callable[..., Any] | None = None
    built_from: tuple[str, ...] = ()
    arity: int = 1
    term: Any = None

    def how_it_computes(self) -> Callable[..., Any] | None:
        """What to run: the term where there is one, the function otherwise."""
        if self.term is not None:
            return _the_term_as_a_function(self.term)
        return self.fn


def _the_term_as_a_function(term: Any) -> Callable[..., Any]:
    """A floor term, wrapped so the kernel's probes can call it.

    Metered by the floor rather than by the kernel's own budget, because a
    universal language has terms that do not stop and the kernel's budget
    counts calls rather than reductions. A term that runs out is a term that
    does not compute here, which is the same answer the budget already gives.
    """
    from core.cognition.the_floor_she_stands_on import (
        NOTHING,
        Code,
        OutOfFuel,
        Pair,
        Stuck,
        as_list,
        from_list,
    )
    from core.cognition.the_floor_she_stands_on import run as run_on_the_floor

    def to_floor(value: Any) -> Any:
        if type(value) is int:
            return value
        if isinstance(value, (tuple, list)):
            return from_list([to_floor(part) for part in value])
        raise TypeError("operator examples require integers or nested sequences")

    def from_floor(value: Any) -> Any:
        if type(value) is int:
            return value
        if value is NOTHING or isinstance(value, Pair):
            return tuple(from_floor(part) for part in as_list(value))
        raise TypeError("operator returned neither an integer nor a sequence")

    def computes(one: Any, budget: Any = None) -> Any:
        if budget is not None:
            budget.step()
        try:
            return from_floor(
                run_on_the_floor(
                    Code("of", parts=(term, Code("the one it was given", value=0))),
                    env=(to_floor(one),),
                )
            )
        except (OutOfFuel, Stuck) as exc:
            raise TimeoutError(str(exc)) from exc

    return computes


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


@dataclass(frozen=True)
class Operator:
    """An installed operator, and where it came from."""

    name: str
    fn: Callable[..., Any]
    body: str = ""
    built_from: tuple[str, ...] = ()
    invented: bool = False
    generation: int = 0
    term: Any = None


class _Budget:
    """Counts steps so a runaway candidate is discarded rather than waited on."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0

    def step(self) -> None:
        self.used += 1
        if self.used > self.limit:
            raise TimeoutError(f"exceeded {self.limit} steps")


@dataclass(frozen=True)
class OperatorState:
    """Installed semantics and their rollback lineage, without audit history."""

    operators: dict[str, Operator]
    snapshots: tuple[tuple[str, dict[str, Operator]], ...]


class OperatorKernel:
    """The evaluator's operator set, and the only way to extend it."""

    def __init__(self, base: Mapping[str, Callable[..., Any]] | None = None) -> None:
        self._lock = checked_lock("core.cognition.operator_invention.OperatorKernel", reentrant=True)
        self._operators: dict[str, Operator] = {
            name: Operator(name=name, fn=fn) for name, fn in (base or {}).items()
        }
        self._snapshots: list[tuple[str, dict[str, Operator]]] = []
        self._verdicts: list[Verdict] = []
        self._residuals: dict[str, Residual] = {}

    def snapshot(self) -> OperatorState:
        with self._lock:
            return OperatorState(
                dict(self._operators),
                tuple((name, dict(before)) for name, before in self._snapshots),
            )

    def restore(self, state: OperatorState) -> None:
        """Restore semantics; observations and attempted verdicts stay in history."""
        with self._lock:
            self._operators = dict(state.operators)
            self._snapshots = [(name, dict(before)) for name, before in state.snapshots]

    @staticmethod
    def _written_operator(operator: Operator) -> dict[str, Any]:
        from core.cognition.the_floor_she_stands_on import Code, read_back, written_down

        if not isinstance(operator.term, Code):
            raise ValueError(f"operator {operator.name!r} has no serializable floor term")
        term = written_down(operator.term)
        if read_back(term) != operator.term:
            raise ValueError(f"operator {operator.name!r} has an unreadable floor term")
        return {
            "schema": "aura.retained_floor_operator.v1", "name": operator.name,
            "body": operator.body, "built_from": list(operator.built_from),
            "generation": operator.generation, "term": term,
        }

    def written_operators(self) -> list[dict[str, Any]]:
        """Keep terms in install order; opaque Python functions cannot be retained."""
        with self._lock:
            return [self._written_operator(operator) for operator in self._operators.values()
                    if operator.invented]

    def recall_operators(self, rows: Any) -> int:
        """Restore a complete recipe batch atomically, without granting new evidence.

        Existing names must have identical recipes. Rollback snapshots are rebuilt
        in installation order, so an ancestor rollback also removes descendants.
        """
        from core.cognition.the_floor_she_stands_on import read_back

        if not isinstance(rows, list):
            raise ValueError("retained operator batch must be a list")
        with self._lock:
            operators = dict(self._operators)
            snapshots = list(self._snapshots)
            seen = set()
            restored = 0
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("invalid retained operator record")
                name, generation = row.get("name"), row.get("generation")
                dependencies = row.get("built_from")
                if (row.get("schema") != "aura.retained_floor_operator.v1"
                        or not isinstance(name, str) or not name or name in seen
                        or not isinstance(row.get("body"), str)
                        or type(generation) is not int or generation < 0
                        or not isinstance(dependencies, list)
                        or any(not isinstance(part, str) or part not in operators or part == name
                               for part in dependencies)
                        or len(set(dependencies)) != len(dependencies)):
                    raise ValueError("invalid retained operator identity or lineage")
                expected = 1 + max((operators[part].generation for part in dependencies), default=-1)
                if generation != expected:
                    raise ValueError("retained operator generation differs from its lineage")
                term = read_back(row.get("term"))
                if term is None:
                    raise ValueError("retained operator has an unreadable floor term")
                seen.add(name)
                existing = operators.get(name)
                if existing is not None:
                    if not existing.invented or self._written_operator(existing) != row:
                        raise ValueError(f"retained operator name conflict: {name!r}")
                    continue
                snapshots.append((name, dict(operators)))
                operators[name] = Operator(
                    name=name, fn=_the_term_as_a_function(term), body=row["body"],
                    built_from=tuple(dependencies), invented=True, generation=generation, term=term,
                )
                restored += 1
            self._operators, self._snapshots = operators, snapshots
            return restored

    # ── residuals ─────────────────────────────────────────────────────

    def attempt(
        self, family: str, *, solved: bool, probes: Sequence[Any] = (),
        cases: Sequence[tuple[Any, Any]] = (),
    ) -> Residual:
        with self._lock:
            residual = self._residuals.setdefault(family, Residual(family=family))
            residual.attempts += 1
            residual.solved += 1 if solved else 0
            if probes:
                residual.probes = tuple(probes)
            # A coarse family key must not mix incompatible demonstrations.
            residual.cases = tuple(cases)
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

        if candidate.name in existing:
            return self._record(Verdict(candidate.name, False, Rejection.NAME_CONFLICT,
                                        detail="an invention cannot replace an installed name"))

        if (not isinstance(candidate.name, str) or not candidate.name
                or not isinstance(candidate.body, str)
                or not isinstance(candidate.built_from, tuple)
                or any(not isinstance(name, str) or name not in existing for name in candidate.built_from)
                or len(set(candidate.built_from)) != len(candidate.built_from)):
            return self._record(Verdict(candidate.name, False, Rejection.INVALID_LINEAGE,
                                        detail="an invention needs a unique installed dependency lineage"))

        if residual is None or not residual.persistent:
            return self._record(Verdict(
                candidate.name, False, Rejection.NOT_PERSISTENT,
                detail=(
                    f"{family!r} has not persistently failed; inventing an operator for a "
                    "transient failure adds vocabulary to cover a bug"
                ),
            ))

        computes = candidate.how_it_computes()
        if computes is None:
            return self._record(Verdict(
                candidate.name, False, Rejection.RAISED,
                detail=(
                    "a candidate with neither a term nor a function computes "
                    "nothing, and a name is not a semantics"
                ),
            ))

        # Bounded execution on every probe.
        outputs: list[Any] = []
        for probe in probes:
            budget = _Budget(STEP_BUDGET)
            try:
                outputs.append(computes(probe, budget))
            except TimeoutError as exc:
                return self._record(Verdict(
                    candidate.name, False, Rejection.NON_TERMINATING, detail=str(exc)
                ))
            except Exception as exc:  # noqa: BLE001
                return self._record(Verdict(
                    candidate.name, False, Rejection.RAISED,
                    detail=f"{type(exc).__name__}: {exc}",
                ))

        # Compare complete functions on the probes, not an output-wise mixture
        # of different operators. Two constants do not already implement identity.
        distinguishing = set()
        for operator in existing.values():
            differences = []
            for index, (probe, output) in enumerate(zip(probes, outputs, strict=True)):
                try:
                    if operator.fn(probe, _Budget(STEP_BUDGET)) != output:
                        differences.append(index)
                except (TypeError, ValueError, ArithmeticError, LookupError,
                        RecursionError, TimeoutError):
                    differences.append(index)
            if not differences:
                return self._record(Verdict(
                    candidate.name, False, Rejection.NOT_NOVEL,
                    detail=f"installed operator {operator.name!r} matches every probe",
                ))
            distinguishing.update(differences)
        novel_on = tuple(probes[index] for index in sorted(distinguishing)) if existing else tuple(probes)
        if not probes:
            return self._record(Verdict(
                candidate.name, False, Rejection.NOT_NOVEL, detail=(
                    "novelty cannot be measured without probes"
                ),
            ))

        # Reach: it must solve the family that was unreachable.
        if not solves(computes, family):
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
                computes(probe, _Budget(STEP_BUDGET))
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
            if self._operators != existing:
                return self._record(Verdict(candidate.name, False, Rejection.STATE_CHANGED,
                    detail="operator semantics changed while the candidate was being checked"))
            self._snapshots.append((candidate.name, dict(self._operators)))
            generation = 1 + max(
                (self._operators[n].generation for n in candidate.built_from),
                default=-1,
            )
            self._operators[candidate.name] = Operator(
                name=candidate.name, fn=computes, body=candidate.body,
                built_from=candidate.built_from, invented=True, generation=generation,
                term=candidate.term,
            )
        return self._record(Verdict(
            candidate.name, True, None, novel_on=tuple(novel_on), reach_gained=(family,),
            compression=compression, adversarial_passed=passed,
            adversarial_total=len(adversarial),
        ))

    def withdraw(self, name: str) -> tuple[str, ...]:
        """Remove an invented dependency closure, not unrelated later installs.

        Prune every rollback snapshot too: a later undo must not resurrect an
        intentionally withdrawn operator. A surrounding trial owns the rescue.
        """
        with self._lock:
            operator = self._operators.get(name)
            if operator is None or not operator.invented:
                return ()
            removed = {name}
            while True:
                dependents = {key for key, value in self._operators.items()
                              if value.invented and removed.intersection(value.built_from)}
                if dependents <= removed:
                    break
                removed.update(dependents)
            ordered = tuple(key for key in self._operators if key in removed)
            self._operators = {key: value for key, value in self._operators.items() if key not in removed}
            self._snapshots = [
                (key, {part: value for part, value in before.items() if part not in removed})
                for key, before in self._snapshots if key not in removed
            ]
            return ordered

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
            _, snapshot = self._snapshots[index]
            del self._snapshots[index:]
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
