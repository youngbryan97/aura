"""The causal record of a tick, written by the runtime rather than narrated.

"Why did you do that?" has three different answers and they are usually
conflated:

    architectural  why did routing choose DELIBERATE?
    decision       why this initiative and not that one?
    neural         why did the hidden state represent X and emit token Y?

The third is an open interpretability problem and this module does not touch
it. The first two are answerable exactly, from thresholds, score vectors and
receipts — and Aura already computed all of it. What she could not do was hand
it to anyone as one object, because the pieces lived in nine shapes across as
many modules: PhaseSpec, ExecutiveClosureSnapshot, ScoredInitiative,
WillDecision, initiative metadata, governance receipts, action receipts,
LifeTrace, state transition causes.

A :class:`CognitiveProvenanceGraph` is one tick's worth of those, in order,
each arrow carrying the contract it ran under, the contract's hash, the state
digest before and after, the branch taken, the criteria evaluated, and which
declared fields actually changed.

The distinction that makes this worth building: **Aura does not write this.**
The runtime measures it around her. Asking the model to explain its own tick
produces a plausible story with no privileged access to the mechanism; querying
this produces the mechanism. Those are not the same epistemic object and the
difference is the entire point.

Per-tick state lives in a ContextVar, so a foreground turn and a background
tick running against the same kernel cannot write into each other's graph —
which they would, being the two rates the pipeline actually runs at.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import time
import uuid
from collections import deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from core.runtime.cognitive_contract import (
    CognitiveTransformContract,
    TransformationReceipt,
    contract_for,
    diff_digests,
    observed_field_digest,
    state_fingerprint,
)
from core.runtime.lockdep import checked_lock

__all__ = [
    "CognitiveProvenanceGraph",
    "begin_transformation",
    "close_tick",
    "current_graph",
    "note_branch",
    "open_tick",
    "recent_graphs",
    "recording_tick",
    "reset_provenance_for_test",
    "why_field_changed",
]

logger = logging.getLogger(__name__)

#: How many ticks of provenance to keep. Bounded because this runs forever on
#: a desktop: the ring is for "why did that just happen", not for an archive.
_RING_SIZE = 64


@dataclass
class CognitiveProvenanceGraph:
    """One tick, as an ordered causal record."""

    tick_id: str
    objective: str = ""
    priority: bool = False
    started_at: float = field(default_factory=time.time)
    ended_at: float = 0.0
    receipts: list[TransformationReceipt] = field(default_factory=list)

    def add(self, receipt: TransformationReceipt) -> None:
        self.receipts.append(receipt)

    @property
    def contract_violations(self) -> list[TransformationReceipt]:
        return [r for r in self.receipts if not r.honoured_contract and not r.skipped]

    def last_writer(self, path: str) -> TransformationReceipt | None:
        """Which transformation last changed a given state field.

        The decision-level "why" for a value: not what the model would say
        about it, but which phase actually moved it and on which branch.
        """

        for receipt in reversed(self.receipts):
            if path in receipt.observed_writes:
                return receipt
        return None

    def narrate(self) -> str:
        """The tick as a human-readable trace.

        Written from the receipts, so it cannot describe a phase that did not
        run or a branch that was not taken.
        """

        lines = [f"Tick {self.tick_id}" + (" (foreground)" if self.priority else "")]
        if self.objective:
            lines.append(f"  objective: {self.objective[:120]}")
        for receipt in self.receipts:
            lines.append("")
            lines.append(f"{receipt.transform}")
            if receipt.skipped:
                lines.append(f"  skipped — {receipt.skip_reason or 'suppressed on this tick'}")
                continue
            if receipt.branch:
                lines.append(f"  branch: {receipt.branch}")
            for key, value in list(receipt.criteria.items())[:8]:
                lines.append(f"  {key}: {value}")
            if receipt.observed_writes:
                lines.append(f"  changed: {', '.join(receipt.observed_writes[:8])}")
            elif receipt.changed_state:
                lines.append("  changed: state version advanced (no watched field moved)")
            if receipt.undeclared_writes:
                lines.append(
                    f"  CONTRACT VIOLATION — undeclared writes: "
                    f"{', '.join(receipt.undeclared_writes)}"
                )
            if receipt.error:
                lines.append(f"  error: {receipt.error}")
            lines.append(f"  {receipt.duration_ms:.1f}ms  contract {receipt.contract_hash}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick_id": self.tick_id,
            "objective": self.objective,
            "priority": self.priority,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_s": round(max(0.0, (self.ended_at or time.time()) - self.started_at), 4),
            "transformations": [receipt.to_dict() for receipt in self.receipts],
            "contract_violations": [r.transform for r in self.contract_violations],
        }


_CURRENT: contextvars.ContextVar[CognitiveProvenanceGraph | None] = contextvars.ContextVar(
    "aura_cognitive_provenance", default=None
)
_RING: deque[CognitiveProvenanceGraph] = deque(maxlen=_RING_SIZE)
_RING_LOCK = checked_lock("cognitive_provenance_ring")


def current_graph() -> CognitiveProvenanceGraph | None:
    return _CURRENT.get()


def open_tick(
    *, objective: str = "", priority: bool = False, tick_id: str | None = None
) -> CognitiveProvenanceGraph:
    """Begin a tick's provenance record without a block.

    The kernel's phase loop is not written as a ``with`` body and re-indenting
    it to make one would be a large diff across the most timing-sensitive code
    in the runtime. This mirrors ``_begin_pass_run`` beside it, which has the
    same shape and the same reason: each tick opens a fresh record, and the
    ContextVar is per-task, so a foreground turn and a background tick on the
    same kernel keep separate graphs rather than interleaving into one.

    The graph joins the ring at open time, so a tick that dies mid-phase still
    leaves the record of how far it got — which is the tick whose record is
    worth the most.
    """

    graph = CognitiveProvenanceGraph(
        tick_id=tick_id or uuid.uuid4().hex[:12],
        objective=str(objective or "")[:400],
        priority=bool(priority),
    )
    _CURRENT.set(graph)
    with _RING_LOCK:
        _RING.append(graph)
    return graph


def close_tick(graph: CognitiveProvenanceGraph | None = None) -> None:
    """Stamp the end of a tick's record. Safe to call more than once."""

    target = graph if graph is not None else _CURRENT.get()
    if target is not None:
        target.ended_at = time.time()


@contextlib.contextmanager
def recording_tick(
    *, objective: str = "", priority: bool = False, tick_id: str | None = None
) -> Iterator[CognitiveProvenanceGraph]:
    """Bind a provenance graph for the duration of one tick."""

    graph = CognitiveProvenanceGraph(
        tick_id=tick_id or uuid.uuid4().hex[:12],
        objective=str(objective or "")[:400],
        priority=bool(priority),
    )
    token = _CURRENT.set(graph)
    try:
        yield graph
    finally:
        graph.ended_at = time.time()
        _CURRENT.reset(token)
        with _RING_LOCK:
            _RING.append(graph)


def recent_graphs(limit: int = 8) -> list[CognitiveProvenanceGraph]:
    with _RING_LOCK:
        return list(_RING)[-max(1, int(limit)) :]


def write_profile(*, within: int = 64) -> dict[str, dict[str, Any]]:
    """What each phase has actually been seen to write, measured not declared.

    The productive end of the ratchet. Writing 28 contracts by reading 28
    phases and their delegates is how a declaration ends up describing what the
    author believed rather than what the code does — this codebase has a
    standing finding for exactly that shape. Running the system and reading
    this instead produces contracts grounded in observation, and it keeps
    working for a phase that delegates through three layers of engine.

    A contract is still a human judgement about what a phase SHOULD write.
    This says what it DID, which is the other half of the comparison.
    """

    profile: dict[str, dict[str, Any]] = {}
    for graph in recent_graphs(within):
        for receipt in graph.receipts:
            entry = profile.setdefault(
                receipt.transform,
                {"runs": 0, "skipped": 0, "observed_writes": set(), "contracted": False},
            )
            entry["contracted"] = entry["contracted"] or bool(receipt.contract_hash)
            if receipt.skipped:
                entry["skipped"] += 1
                continue
            entry["runs"] += 1
            entry["observed_writes"].update(receipt.observed_writes)
    return {
        name: {
            "runs": entry["runs"],
            "skipped": entry["skipped"],
            "contracted": entry["contracted"],
            "observed_writes": sorted(entry["observed_writes"]),
        }
        for name, entry in profile.items()
    }


def why_field_changed(path: str, *, within: int = 8) -> dict[str, Any]:
    """Query the record: which transformation last moved this field, and why.

    This is the query the whole module exists for. The answer comes from
    receipts, so it is the same answer whether or not the model is available,
    and it is wrong only if the measurement was wrong.
    """

    graphs = recent_graphs(within)
    for graph in reversed(graphs):
        receipt = graph.last_writer(path)
        if receipt is not None:
            return {
                "found": True,
                "field": path,
                "tick_id": graph.tick_id,
                "transform": receipt.transform,
                "branch": receipt.branch,
                "criteria": dict(receipt.criteria),
                "contract_hash": receipt.contract_hash,
                "at": receipt.at,
            }
    return {"found": False, "field": path, "searched_ticks": len(graphs)}


class _Transformation:
    """One in-flight phase execution, measured from outside it."""

    __slots__ = (
        "_before",
        "_branch",
        "_contract",
        "_criteria",
        "_name",
        "_started",
        "_state_before_hash",
    )

    def __init__(
        self,
        name: str,
        contract: CognitiveTransformContract | None,
        state: Any,
    ) -> None:
        self._name = name
        self._contract = contract
        self._started = time.perf_counter()
        self._branch = ""
        self._criteria: dict[str, Any] = {}
        self._state_before_hash = state_fingerprint(state)
        # Digested for EVERY phase, contracted or not. An uncontracted phase
        # still reports which fields it moved, which is how the next contract
        # gets written from measurement instead of from reading the code and
        # hoping — see ``write_profile``. The cost is a few short hashes per
        # phase, which is why it can be unconditional.
        #
        # `discover` is on precisely when there is no contract. Without it the
        # watch set is derived from the contracts alone, so a phase that
        # declares nothing is compared only against fields OTHER phases
        # declared — which reported nothing for almost every uncontracted
        # phase and left the ratchet with no productive end at all.
        self._before = observed_field_digest(state, discover=contract is None)

    def note_branch(self, branch: str, **criteria: Any) -> None:
        self._branch = str(branch or "")
        for key, value in criteria.items():
            self._criteria[str(key)] = value

    def complete(
        self,
        state: Any,
        *,
        error: str = "",
        skipped: bool = False,
        skip_reason: str = "",
        inputs: Mapping[str, Any] | None = None,
        publish_violation: bool = True,
    ) -> TransformationReceipt | None:
        graph = _CURRENT.get()
        contract = self._contract
        duration_ms = (time.perf_counter() - self._started) * 1000.0
        declared = tuple(contract.writes) if contract else ()
        observed: tuple[str, ...] = ()
        undeclared: tuple[str, ...] = ()
        if not skipped:
            # Same watch set as `self._before`, or the diff compares two
            # different surfaces and every path present in one but not the
            # other reads as a change.
            after = observed_field_digest(state, discover=contract is None)
            observed = diff_digests(self._before, after)
            if contract is not None:
                declared_set = set(declared)
                undeclared = tuple(
                    path
                    for path in observed
                    if path not in declared_set and not _is_derived_field(path)
                )
        receipt = TransformationReceipt(
            transform=self._name,
            contract_hash=contract.content_hash if contract else "",
            contract_version=contract.version if contract else "",
            state_before_hash=self._state_before_hash,
            state_after_hash=state_fingerprint(state),
            inputs=dict(inputs or {}),
            branch=self._branch,
            criteria=dict(self._criteria),
            declared_writes=declared,
            observed_writes=observed,
            undeclared_writes=undeclared,
            duration_ms=duration_ms,
            skipped=skipped,
            skip_reason=skip_reason,
            error=error,
        )
        if graph is not None:
            graph.add(receipt)
        if undeclared and publish_violation:
            _report_violation(self._name, undeclared)
        return receipt


#: Transformations announce their branch through this rather than through a
#: changed signature, so a phase opts in with one call and phases that have not
#: opted in still get measured hashes and writes.
_ACTIVE: contextvars.ContextVar[_Transformation | None] = contextvars.ContextVar(
    "aura_active_transformation", default=None
)


def begin_transformation(phase_name: str, state: Any) -> _Transformation:
    """Start measuring one phase execution."""

    transformation = _Transformation(phase_name, contract_for(phase_name), state)
    _ACTIVE.set(transformation)
    return transformation


def note_branch(branch: str, **criteria: Any) -> None:
    """Record which path this phase took and what decided it.

    Called from inside a phase. A no-op when nothing is measuring, so a phase
    can announce itself unconditionally and stay usable in a unit test.
    """

    active = _ACTIVE.get()
    if active is None:
        return
    active.note_branch(branch, **criteria)


#: State that the STATE maintains about itself, which no transform declares
#: because no transform decides it.
#:
#: `health` is refreshed by AuraState._refresh_cognitive_health, which runs
#: from working-memory compaction — housekeeping that fires inside whichever
#: phase happened to push memory over the threshold. So it surfaced as an
#: undeclared write by ConsciousnessPhase, CognitiveIntegrationPhase and
#: ProprioceptiveLoop alike, several hundred times, attributing to a phase a
#: field it never touched.
#:
#: Declaring it in each of those contracts would have silenced the warning by
#: recording something false: they do not write it, and a later reader would
#: have concluded these phases decide her health. Excluding derived state is
#: the honest version — the diff measures what a TRANSFORM did, and this is
#: not a transform's doing.
_DERIVED_STATE_PREFIXES = ("health",)


def _is_derived_field(path: str) -> bool:
    """Whether a changed field is state housekeeping rather than a phase's work."""
    text = str(path or "")
    return any(
        text == prefix or text.startswith(f"{prefix}.")
        for prefix in _DERIVED_STATE_PREFIXES
    )


def _report_violation(transform: str, undeclared: tuple[str, ...]) -> None:
    """An undeclared write is a defect, and it is reported as one.

    Not raised. A contract violation means the declaration and the code
    disagree, which is worth a loud record and is never worth killing a live
    tick over — the phase's actual work already happened by the time this runs.
    """

    try:
        from core.runtime.errors import record_degradation

        record_degradation(
            "cognitive_contract",
            RuntimeError(
                f"{transform} wrote undeclared state fields: {', '.join(undeclared)}"
            ),
            action="recorded contract violation and continued the tick",
            severity="warning",
        )
    except (ImportError, AttributeError, RuntimeError) as exc:
        logger.warning(
            "contract violation in %s (%s) could not be recorded: %s",
            transform,
            ", ".join(undeclared),
            exc,
        )


def reset_provenance_for_test() -> None:
    with _RING_LOCK:
        _RING.clear()
    _CURRENT.set(None)
    _ACTIVE.set(None)
