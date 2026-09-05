"""core/cognition/trace_compiler.py — a thing done twice becomes a thing known.

The mechanisms for turning experience into skill all exist and each compiles
from its own kind of episode: impasse.py from a deadlock resolution,
procedural_generalization.py from decision episodes, procedure_induction from
tool sequences. Nothing compiles from the thing all three are really watching -
a successful stretch of cognition, whatever produced it.

The compiler here reads the event DAG. When the same task succeeds repeatedly
it takes the causal chain, computes the minimal support, generalises what
varied, and emits a :class:`~core.cognition.procedure.Procedure` into the one
registry where every learner's output competes.

Three things it will not do
---------------------------
* **Compile from one success.** One run is an anecdote and its incidental
  context is indistinguishable from its causal support. ``min_occurrences``
  defaults to three, and the generalisation is over what CHANGED between them.
* **Keep a condition that never varied.** A field identical in every observed
  run may be a precondition or may be a constant of the environment, and the
  compiler cannot tell. It keeps it and marks it ``unvaried``, so a later run
  that varies it either confirms the condition or removes it - which is the
  only way to find out, and better than guessing in either direction.
* **Rest on something nobody looked at.** ``minimal_support`` already drops
  UNOBSERVED reads; the compiler refuses outright if that leaves no support at
  all, because a procedure with no preconditions fires everywhere.

Cost is measured, not estimated
-------------------------------
``cost_saved_per_use`` comes from the deliberation the compiled runs actually
spent, taken from the event durations. A compiler that estimates its own
saving will always find that it saved something.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.cognition.cognitive_event import CognitiveEvent, EventGraph, ReadDependency
from core.cognition.procedure import (
    Backend,
    Effect,
    Origin,
    Precondition,
    Procedure,
    ProceduralValue,
    ProcedureRegistry,
    Signature,
)
from core.runtime.lockdep import checked_lock

__all__ = ["Trace", "CompilationResult", "TraceCompiler", "MIN_OCCURRENCES"]

#: Successful runs of the same task before anything is compiled. One run is an
#: anecdote; two cannot separate what varied from what happened to differ.
MIN_OCCURRENCES = 3


@dataclass(frozen=True, slots=True)
class Trace:
    """One successful stretch of cognition, as the event graph recorded it."""

    task: str
    terminal_event: int
    events: tuple[CognitiveEvent, ...]
    support: tuple[ReadDependency, ...]
    seconds: float
    outcome_value: float = 1.0

    @property
    def support_keys(self) -> frozenset[str]:
        return frozenset(d.key for d in self.support)


@dataclass(frozen=True, slots=True)
class CompilationResult:
    """What the compiler decided about a task, and why."""

    task: str
    compiled: Procedure | None
    occurrences: int
    shared_support: tuple[str, ...]
    unvaried: tuple[str, ...]
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "compiled": self.compiled.procedure_id if self.compiled else None,
            "occurrences": self.occurrences,
            "shared_support": list(self.shared_support),
            "unvaried_conditions": list(self.unvaried),
            "reason": self.reason,
        }


class TraceCompiler:
    """Watches successful traces; compiles the ones that recur."""

    def __init__(
        self,
        registry: ProcedureRegistry,
        *,
        min_occurrences: int = MIN_OCCURRENCES,
    ) -> None:
        self._lock = checked_lock("core.cognition.trace_compiler.TraceCompiler", reentrant=True)
        self._registry = registry
        self._traces: dict[str, list[Trace]] = {}
        self._min = int(min_occurrences)
        self._compiled: dict[str, str] = {}
        self._refusals: list[dict[str, Any]] = []

    def observe(
        self, graph: EventGraph, task: str, terminal_event: int, *, outcome_value: float = 1.0
    ) -> Trace | None:
        """Record one successful run of a task from the event graph."""
        bundle = graph.bundle(terminal_event)
        if not bundle["found"]:
            return None
        event = graph.get(terminal_event)
        chain = graph.ancestors(terminal_event)
        seconds = sum(e.duration_s for e in [event, *chain] if e is not None)
        trace = Trace(
            task=task,
            terminal_event=terminal_event,
            events=tuple(e for e in [event, *chain] if e is not None),
            support=tuple(graph.minimal_support(terminal_event)),
            seconds=seconds,
            outcome_value=outcome_value,
        )
        with self._lock:
            self._traces.setdefault(task, []).append(trace)
        return trace

    def compile(self, task: str) -> CompilationResult:
        """Generalise over the recorded runs and emit a procedure, or say why not."""
        with self._lock:
            traces = list(self._traces.get(task, ()))
        if len(traces) < self._min:
            return CompilationResult(
                task, None, len(traces), (), (),
                reason=f"{len(traces)} run(s); one is an anecdote and two cannot separate "
                       f"what varied from what happened to differ ({self._min} needed)",
            )
        if task in self._compiled:
            return CompilationResult(
                task, self._registry.get(self._compiled[task]), len(traces), (), (),
                reason="already compiled",
            )

        shared = frozenset.intersection(*(t.support_keys for t in traces))
        if not shared:
            self._refusals.append({"task": task, "reason": "no support survives every run"})
            return CompilationResult(
                task, None, len(traces), (), (),
                reason="no read survived every run; a procedure with no preconditions "
                       "fires everywhere",
            )

        # A key whose observed value never changed may be a precondition or a
        # constant of the environment. Keep it and say which is which.
        values: dict[str, set[str]] = {}
        for trace in traces:
            for dependency in trace.support:
                if dependency.key in shared:
                    values.setdefault(dependency.key, set()).add(dependency.value_digest)
        unvaried = tuple(sorted(k for k, v in values.items() if len(v) <= 1))
        varied = tuple(sorted(k for k, v in values.items() if len(v) > 1))

        effects = tuple(
            Effect(key=f"task:{task}:done")
            for _ in (None,)
        )
        seconds = sorted(t.seconds for t in traces)
        median = seconds[len(seconds) // 2]

        procedure = self._registry.register(
            f"compiled:{task}",
            Backend.CHUNK,
            Signature(
                preconditions=tuple(Precondition(key=k) for k in sorted(shared)),
                effects=effects,
            ),
            program={"task": task, "traces": [t.terminal_event for t in traces]},
            value=ProceduralValue(
                p_success=1.0,
                # Measured from the deliberation these runs actually spent.
                value_when_it_works=median,
                # A chunk that fires and is wrong costs the same deliberation
                # over again, because the work it displaced still has to
                # happen. Without this term a rule that misses four times in
                # five still shows a positive net and is never retired, which
                # is the utility problem by another name.
                cost_when_it_fails=median,
                match_cost=0.001 * len(shared),
                uses=len(traces),
                successes=len(traces),
            ),
            origin=Origin(
                learner="core.cognition.trace_compiler",
                support_keys=tuple(sorted(shared)),
                causal_events=tuple(t.terminal_event for t in traces),
                rejected_conditions=tuple(
                    sorted(
                        {d.key for t in traces for d in t.support} - set(shared)
                    )
                ),
                # Kept as conditions but never seen to matter: present in every
                # run and different in every run. Success traces cannot tell
                # these from real preconditions, so they are named here for the
                # first run that succeeds without one to drop through
                # ProcedureRegistry.generalise.
                provisional_conditions=varied,
            ),
        )
        with self._lock:
            self._compiled[task] = procedure.procedure_id
        return CompilationResult(
            task, procedure, len(traces), tuple(sorted(shared)), unvaried,
            reason=(
                f"compiled from {len(traces)} runs; {len(varied)} condition(s) varied and "
                f"{len(unvaried)} did not and are kept until a run varies them"
            ),
        )

    def report(self) -> dict[str, Any]:
        with self._lock:
            traces = {task: len(rows) for task, rows in self._traces.items()}
            compiled = dict(self._compiled)
            refusals = list(self._refusals)
        return {
            "tasks_observed": len(traces),
            "runs_by_task": dict(sorted(traces.items())),
            "compiled": dict(sorted(compiled.items())),
            "awaiting_repetition": sorted(
                task for task, count in traces.items()
                if count < self._min and task not in compiled
            ),
            "refusals": refusals,
        }
