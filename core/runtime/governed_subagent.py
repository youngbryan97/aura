"""core/runtime/governed_subagent.py — a temporary specialist that is not a second Aura.

Fanning work out to parallel workers is how the frontier agent stacks get their
throughput, and the naive adaptation would give Aura several minds. She is one
continuing individual; several of her is a different system.

A subagent here is a **temporary specialist with an isolated context and an
explicit budget**, and three properties keep it from becoming a second Aura:

* **It cannot commit.** A subagent returns a finding. Anything consequential
  goes back to the individual, who decides. The type system says so: the return
  is a :class:`Finding`, not an action.
* **Its context is a subset, declared.** It sees what it was given and cannot
  reach the rest. A worker that can read everything is not isolated, and
  isolation is what makes a finding attributable.
* **Its budget is spent, not shared.** Overspending ends the subagent rather
  than borrowing, so one worker cannot starve the turn.

The measurement that decides whether this is worth having
---------------------------------------------------------
:meth:`Conductor.fanout_report` compares total cost across workers against the
same work done by one, at the same budget. Parallel workers are only a win when
the work partitions; when it does not, fanout multiplies cost and produces
several partial answers that then have to be reconciled, which costs again. The
report says which happened rather than assuming the first.
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
    "Finding",
    "SubagentSpec",
    "Subagent",
    "Conductor",
    "BudgetExhausted",
]


class BudgetExhausted(RuntimeError):
    """A subagent spent what it was given. It ends rather than borrowing."""


@dataclass(frozen=True, slots=True)
class Finding:
    """What a subagent returns. Never an action; the individual decides."""

    subagent: str
    content: Any
    confidence: float = 0.0
    cost: float = 0.0
    context_keys: tuple[str, ...] = ()
    failed: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "subagent": self.subagent, "confidence": self.confidence, "cost": self.cost,
            "context_keys": list(self.context_keys), "failed": self.failed,
        }


@dataclass(frozen=True, slots=True)
class SubagentSpec:
    """What a specialist is for, what it may see, and what it may spend."""

    name: str
    purpose: str
    #: Keys of the parent context this worker may read. Declared, and enforced.
    context_keys: frozenset[str]
    budget: float
    tools: frozenset[str] = frozenset()


@dataclass
class Subagent:
    """One running specialist and its spend."""

    spec: SubagentSpec
    spent: float = 0.0
    started_at: float = field(default_factory=time.monotonic)

    def spend(self, amount: float) -> None:
        self.spent += max(0.0, amount)
        if self.spent > self.spec.budget:
            raise BudgetExhausted(
                f"{self.spec.name!r} spent {self.spent:.4g} of {self.spec.budget:.4g}; it "
                "ends rather than borrowing from the turn"
            )

    @property
    def remaining(self) -> float:
        return max(0.0, self.spec.budget - self.spent)


class Conductor:
    """Spawns specialists, isolates them, and reconciles what they return."""

    def __init__(self) -> None:
        self._lock = checked_lock("core.runtime.governed_subagent.Conductor", reentrant=True)
        self._findings: list[Finding] = []
        self._fanouts: list[dict[str, Any]] = []

    def run(
        self,
        spec: SubagentSpec,
        work: Callable[[Subagent, Mapping[str, Any]], Any],
        *,
        context: Mapping[str, Any],
        permitted_tools: frozenset[str] = frozenset(),
    ) -> Finding:
        """Run one specialist against a subset of the context it declared."""
        overreach = spec.tools - permitted_tools
        if overreach:
            return self._record(Finding(
                spec.name, None, failed=(
                    f"declared tools {sorted(overreach)} the caller may not use"
                ),
            ))
        missing = spec.context_keys - set(context)
        isolated = {k: v for k, v in context.items() if k in spec.context_keys}
        subagent = Subagent(spec=spec)
        try:
            content = work(subagent, isolated)
        except BudgetExhausted as exc:
            return self._record(Finding(
                spec.name, None, cost=subagent.spent,
                context_keys=tuple(sorted(isolated)), failed=str(exc),
            ))
        except Exception as exc:  # noqa: BLE001 - a failing worker is a finding
            return self._record(Finding(
                spec.name, None, cost=subagent.spent,
                context_keys=tuple(sorted(isolated)),
                failed=f"{type(exc).__name__}: {exc}",
            ))
        return self._record(Finding(
            spec.name, content, confidence=0.0 if missing else 1.0,
            cost=subagent.spent, context_keys=tuple(sorted(isolated)),
        ))

    def fanout(
        self,
        specs: Sequence[SubagentSpec],
        work: Callable[[Subagent, Mapping[str, Any]], Any],
        *,
        context: Mapping[str, Any],
        permitted_tools: frozenset[str] = frozenset(),
        single_worker_cost: float | None = None,
    ) -> dict[str, Any]:
        """Run several and report whether fanning out was worth it."""
        findings = [
            self.run(spec, work, context=context, permitted_tools=permitted_tools)
            for spec in specs
        ]
        total = sum(f.cost for f in findings)
        record = {
            "workers": len(specs),
            "findings": [f.to_dict() for f in findings],
            "total_cost": total,
            "single_worker_cost": single_worker_cost,
            "worth_it": (
                None if single_worker_cost is None else total < single_worker_cost
            ),
            "reconciliation_needed": sum(1 for f in findings if not f.failed) > 1,
            "failed": [f.subagent for f in findings if f.failed],
        }
        with self._lock:
            self._fanouts.append(record)
        return record

    def _record(self, finding: Finding) -> Finding:
        with self._lock:
            self._findings.append(finding)
        return finding

    def fanout_report(self) -> dict[str, Any]:
        with self._lock:
            fanouts = list(self._fanouts)
            findings = list(self._findings)
        judged = [f for f in fanouts if f["worth_it"] is not None]
        return {
            "subagents_run": len(findings),
            "fanouts": len(fanouts),
            "fanouts_judged": len(judged),
            "fanouts_worth_it": sum(1 for f in judged if f["worth_it"]),
            "failures": [f.to_dict() for f in findings if f.failed],
            "nothing_committed": True,
        }
