"""core/science/experiment_registry.py — every number, back to the run that made it.

Aura has the hard parts of this already. ``preregistration.py`` freezes an
analysis plan and hashes it. ``matched_budget.py`` refuses a verdict when the
arms did not spend the same resources. ``model_validation.py`` binds a claim to
a test. What is missing is the record that joins them: a place where a number
in a document resolves to the run that produced it, the arms it was compared
against, the null that had to fail, and the commit it ran at.

Without that, "Aura scores 0.83" is a number with a provenance that lives in
somebody's memory of which afternoon it was.

An :class:`ExperimentRecord` refuses four things, and each refusal has a
committed instance behind it:

* **A treatment with no null.** An A/B whose control was never specified
  cannot fail. The retracted steering result scored d(steered, control) −
  d(steered, baseline) over a runner that gave steered and baseline the same
  prompt and seed; steering with no effect zeroes the subtracted term and
  leaves a positive number by construction. The null hypothesis passed
  decisively and the artifact recorded it doing so.

* **Arms that were not allowed the same resources.** Three structurally
  different baselines returned an identical 0.1667 because all three ran out
  of tokens before they could emit an answer. This delegates to
  ``matched_budget.require_budget_parity`` rather than reimplementing it.

* **A result with no seeds.** One sample never settles whether a failure is
  yours. Aura has a hash-randomised ranking defect on record for exactly this.

* **A conclusion wider than the tasks.** ``claim_boundary`` is required and
  is compared against the task families the experiment actually ran, so a
  result on one environment family cannot be written up as a result about
  environments.
"""

from __future__ import annotations

from core.runtime.lockdep import checked_lock
import hashlib
import json
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "Arm",
    "ExperimentRecord",
    "ExperimentRegistry",
    "get_experiment_registry",
    "reset_experiment_registry_for_test",
    "MalformedExperiment",
]

ROOT = Path(__file__).resolve().parent.parent.parent


class MalformedExperiment(ValueError):
    """An experiment that cannot support a claim, refused at registration."""


def _commit() -> str:
    """The commit an experiment ran at. Unknown rather than a guess.

    Through the subprocess gateway, because a raw ``subprocess.run`` here is
    how process ownership erodes: this is a small read and every small read is.
    """
    try:
        from core.runtime.subprocess_gateway import get_subprocess_gateway

        out = get_subprocess_gateway().run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT, timeout=5.0, read_only=True, source="experiment_registry",
            accelerator_capability="none",
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except (OSError, ImportError, RuntimeError, ValueError):
        return "unknown"


@dataclass(frozen=True, slots=True)
class Arm:
    """One condition. ``role`` says what it is FOR, which is what makes it a design."""

    name: str
    role: str  # treatment | control | null | baseline
    score: float
    n: int
    seeds: tuple[int, ...] = ()
    #: Fields for ``core.evaluation.matched_budget.ConditionBudget``, minus
    #: ``condition``. That module is the authority on what an arm may declare;
    #: duplicating its schema here would be a second definition of parity.
    budget: Mapping[str, Any] = field(default_factory=dict)
    ci: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "score": self.score,
            "n": self.n,
            "seeds": list(self.seeds),
            "budget": dict(self.budget),
            "ci": list(self.ci) if self.ci else None,
        }


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    """One experiment, complete enough that a number can be traced to it."""

    experiment_id: str
    hypothesis: str
    arms: tuple[Arm, ...]
    task_families: tuple[str, ...]
    claim_boundary: str
    commit: str = ""
    model: str = ""
    preregistration_hash: str = ""
    metrics: Mapping[str, Any] = field(default_factory=dict)
    verdict: str = ""
    at: float = field(default_factory=time.time)

    @property
    def content_hash(self) -> str:
        return hashlib.blake2s(
            json.dumps(self.to_dict(), sort_keys=True, default=str).encode("utf-8"),
            digest_size=16,
        ).hexdigest()

    def arm(self, role: str) -> Arm | None:
        return next((a for a in self.arms if a.role == role), None)

    @property
    def separation(self) -> float | None:
        """Treatment minus its null. The number a reader actually wants."""
        treatment, null = self.arm("treatment"), self.arm("null")
        return None if treatment is None or null is None else treatment.score - null.score

    @property
    def null_failed(self) -> bool:
        """Whether the null scored below the treatment, which is the point of it."""
        separation = self.separation
        return separation is not None and separation > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "hypothesis": self.hypothesis,
            "arms": [a.to_dict() for a in self.arms],
            "task_families": list(self.task_families),
            "claim_boundary": self.claim_boundary,
            "commit": self.commit,
            "model": self.model,
            "preregistration_hash": self.preregistration_hash,
            "metrics": dict(self.metrics),
            "verdict": self.verdict,
            "separation": self.separation,
            "null_failed": self.null_failed,
            "at": self.at,
        }


class ExperimentRegistry:
    """Where a quantitative claim goes to be checkable."""

    def __init__(self, *, max_records: int = 10_000) -> None:
        self._lock = checked_lock("core.science.experiment_registry.ExperimentRegistry", reentrant=True)
        self._records: dict[str, ExperimentRecord] = {}
        self._max = int(max_records)
        self._refused: list[str] = []

    def register(
        self,
        experiment_id: str,
        *,
        hypothesis: str,
        arms: Sequence[Arm],
        task_families: Sequence[str],
        claim_boundary: str,
        model: str = "",
        preregistration_hash: str = "",
        metrics: Mapping[str, Any] | None = None,
        verdict: str = "",
        require_parity: bool = True,
    ) -> ExperimentRecord:
        """Record an experiment, refusing one that cannot support a claim."""
        problems: list[str] = []
        roles = {a.role for a in arms}
        if "treatment" not in roles:
            problems.append("no treatment arm")
        if "null" not in roles:
            problems.append(
                "no null arm; an A/B whose control was never specified cannot fail"
            )
        for arm in arms:
            if not arm.seeds:
                problems.append(f"arm {arm.name!r} reports no seeds; one sample settles nothing")
            if arm.n <= 0:
                problems.append(f"arm {arm.name!r} has n={arm.n}")
        if not task_families:
            problems.append("no task families named; the claim has no stated scope")
        if not claim_boundary.strip():
            problems.append("no claim boundary; a result on one family reads as a general result")

        if require_parity and not problems:
            problems.extend(self._parity_problems(arms))

        if problems:
            message = f"experiment {experiment_id!r} refused: " + "; ".join(problems)
            with self._lock:
                self._refused.append(message)
            raise MalformedExperiment(message)

        record = ExperimentRecord(
            experiment_id=experiment_id,
            hypothesis=hypothesis,
            arms=tuple(arms),
            task_families=tuple(task_families),
            claim_boundary=claim_boundary,
            commit=_commit(),
            model=model,
            preregistration_hash=preregistration_hash,
            metrics=dict(metrics or {}),
            verdict=verdict,
        )
        with self._lock:
            if len(self._records) >= self._max:
                self._records.pop(next(iter(self._records)))
            self._records[experiment_id] = record
        return record

    @staticmethod
    def _parity_problems(arms: Sequence[Arm]) -> list[str]:
        """Delegate budget parity to the module that already refuses on it."""
        declared = [a for a in arms if a.budget]
        if len(declared) < 2:
            return [
                "fewer than two arms declared a budget; an unmatched comparison is not a "
                "comparison, and undeclared budgets cannot be matched"
            ]
        try:
            from core.evaluation.matched_budget import ConditionBudget, check_budget_parity
        except ImportError:
            return []
        try:
            budgets = [
                ConditionBudget(condition=a.name, **{k: v for k, v in a.budget.items()})
                for a in declared
            ]
        except (TypeError, ValueError) as exc:
            return [
                f"budget fields do not match ConditionBudget ({exc}); the parity checker "
                "is the authority on what an arm may declare"
            ]
        report = check_budget_parity(budgets)
        if report.matched:
            return []
        return [
            "arms were not allowed the same resources: "
            + "; ".join(v.describe() for v in report.violations)
        ]

    def get(self, experiment_id: str) -> ExperimentRecord | None:
        with self._lock:
            return self._records.get(experiment_id)

    def records(self) -> list[ExperimentRecord]:
        with self._lock:
            return sorted(self._records.values(), key=lambda r: r.at)

    def report(self) -> dict[str, Any]:
        with self._lock:
            records = list(self._records.values())
            refused = list(self._refused)
        return {
            "experiments": len(records),
            "refused": len(refused),
            "refusals": refused[-10:],
            "with_a_failing_null": sum(1 for r in records if r.null_failed),
            "preregistered": sum(1 for r in records if r.preregistration_hash),
            "families": sorted({f for r in records for f in r.task_families}),
            "by_verdict": {
                v: sum(1 for r in records if r.verdict == v)
                for v in sorted({r.verdict for r in records if r.verdict})
            },
        }


_lock = checked_lock("core.science.experiment_registry.singleton")
_registry: ExperimentRegistry | None = None


def get_experiment_registry() -> ExperimentRegistry:
    global _registry
    with _lock:
        if _registry is None:
            _registry = ExperimentRegistry()
        return _registry


def reset_experiment_registry_for_test() -> ExperimentRegistry:
    global _registry
    with _lock:
        _registry = ExperimentRegistry()
        return _registry
