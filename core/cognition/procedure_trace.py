"""Observed procedure reads and writes in the existing cognitive event graph.

This records dependencies at the mapping boundary, not semantic necessity.
Copying an entire mapping conservatively records the entire mapping as read.
Raw values and backend evidence payloads are not retained in this graph.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
import logging
from typing import Any

from core.cognition.cognitive_event import (
    Epistemic, EventGraph, Phase, ReadDependency, reads,
)
from core.runtime.errors import record_degradation
from core.verify.invariants import invariant

#: What a trace may survive. A digest that will not hash, a store that will
#: not take the event, a degradation sink that is down: each is recorded on
#: the trace by name and the answer goes out. A bug in the trace itself
#: (an assertion, a name that does not exist) is not one of these.
_TRACE_RECOVERABLE = (
    AttributeError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)

logger = logging.getLogger(__name__)


class ObservedProcedureInputs(Mapping[str, Any]):
    """Read-only top-level view that records actual key lookups and enumeration."""

    def __init__(self, values: Mapping[str, Any], *, namespace: str, owner: str) -> None:
        self._values = values
        self._namespace = namespace
        self._owner = owner
        self.dependencies: dict[str, ReadDependency] = {}
        self.keys_read: set[str] = set()
        self.errors: set[str] = set()

    def _dependency(self, key: str, value: Any) -> ReadDependency:
        try:
            return reads(((key, value),), owner=self._owner)[0]
        except _TRACE_RECOVERABLE as exc:
            self.errors.add("read_digest:" + type(exc).__name__)
            return ReadDependency(key, Epistemic.INACCESSIBLE, owner=self._owner)

    def _record(self, key: str, value: Any, *, present: bool) -> None:
        if self._namespace + key in self.dependencies:
            return
        dependency = self._dependency(self._namespace + key, value)
        # False is an observed local value. The absence of a binding is separate.
        if present and dependency.status is not Epistemic.INACCESSIBLE:
            dependency = replace(dependency, status=Epistemic.OBSERVED)
        self.dependencies[dependency.key] = dependency

    def __getitem__(self, key: str) -> Any:
        self.keys_read.add(key)
        if key not in self._values:
            self._record(key, None, present=False)
            raise KeyError(key)
        value = self._values[key]
        self._record(key, value, present=True)
        return value

    def __iter__(self) -> Iterator[str]:
        self._record_structure()
        return iter(self._values)

    def __len__(self) -> int:
        self._record_structure()
        return len(self._values)

    def _record_structure(self) -> None:
        key = "procedure:keys:" + self._namespace
        if key not in self.dependencies:
            dependency = self._dependency(key, tuple(self._values))
            self.dependencies[key] = dependency
        self.keys_read.update(self._values)


@dataclass(frozen=True, slots=True)
class ProcedureTraceResult:
    event_id: int
    errors: tuple[str, ...] = ()


class ProcedureTrace:
    """Per-execution data provenance; no cross-request writer state."""

    def __init__(self, graph: EventGraph, parents: Sequence[int] = ()) -> None:
        self.graph = graph
        self.parents = tuple(parents)
        self.writers: dict[str, int] = {}

    def record(
        self,
        procedure_id: str,
        backend: str,
        state: ObservedProcedureInputs,
        context: ObservedProcedureInputs,
        *,
        outputs: Sequence[str],
        duration_s: float,
        error: str = "",
    ) -> ProcedureTraceResult:
        missing = tuple(sorted(key for key in state.keys_read if self.writers.get(key) == 0))
        parents = tuple(sorted({
            *self.parents,
            *(self.writers[key] for key in state.keys_read if self.writers.get(key)),
        }))
        errors = {*state.errors, *context.errors}
        if missing:
            errors.add("upstream_event_unavailable")
        event_id = 0
        try:
            event = self.graph.record(
                Phase.APPLY, "procedure_execution", procedure_id, loop="procedure",
                parents=parents,
                reads=(*state.dependencies.values(), *context.dependencies.values()),
                produced=tuple("procedure:state:" + key for key in outputs),
                duration_s=duration_s, outcome="failed" if error else "executed",
                detail={
                    "backend": backend, "procedure_id": procedure_id,
                    "correctness_measured": False, "dependency_basis": "observed_mapping_reads",
                    "error_type": error.partition(":")[0] if error else "",
                    "trace_errors": sorted(errors), "unrecorded_input_writers": missing,
                },
            )
            event_id = event.seq
            if set(parents) - set(event.parents):
                errors.add("parent_event_not_retained")
        except _TRACE_RECOVERABLE as exc:
            errors.add("event_record:" + type(exc).__name__)
        if not error:
            for key in outputs:
                # A missing new event must not leave an older writer attached.
                self.writers[key] = event_id
        if errors:
            try:
                record_degradation(
                    "procedure_trace", RuntimeError(",".join(sorted(errors))),
                    action="retain execution result with incomplete dependency evidence",
                    enforce_failure_policy=False,
                )
            except _TRACE_RECOVERABLE as exc:
                errors.add("degradation_record:" + type(exc).__name__)
                logger.warning("Procedure trace reporting failed: %s", type(exc).__name__)
        return ProcedureTraceResult(event_id, tuple(sorted(errors)))


@invariant(
    "procedure.local_false_is_observed", scope="procedure",
    owner="core/cognition/procedure_trace.py", observational=False,
)
def _false_binding_is_observed() -> tuple:
    values = ObservedProcedureInputs({"flag": False}, namespace="state:", owner="probe")
    assert values["flag"] is False
    assert values.get("missing") is None
    assert values.dependencies["state:flag"].status is Epistemic.OBSERVED
    assert values.dependencies["state:missing"].status is Epistemic.OBSERVED_ABSENT
    return ()
