"""What Aura is allowed to learn permanently, and on whose evidence.

A one-turn misclassification is inconvenient. A misclassification that
becomes persistent learning data changes future behaviour, and Aura's
learning updates feed routing, confidence, verifier reliability,
curriculum selection, memory, adaptation and self-assessment. Getting this
wrong does not produce a wrong answer; it produces a system that is
gradually, confidently wrong because its own machinery kept congratulating
itself.

The failures that motivated this were all the same shape — a component's
report about itself, accepted as a checked result:

* Mycelium strengthened a route after a tool call because nothing threw,
  even when the tool returned a failure. The route became more trusted for
  producing a failed outcome.
* Worker self-reports were treated as graded receipts.
* The string ``"false"`` became ``True`` through boolean coercion, so a
  policy that was off read as on.
* Malformed inputs reached calibration stores.
* Successful execution was read as correct reasoning.

So durable learning gets an admission gate with one rule: **the strength
of the evidence decides the scope of the change.** An unverified success
may steer this session and dies with it. Only an independently checked
outcome may alter what Aura believes tomorrow.

Asymmetry is deliberate. A verified FAILURE weakens a route immediately
and durably, while an unverified SUCCESS does not strengthen one durably.
That is not inconsistency: acting on a route that did not work is the
risk, and requiring proof before believing bad news is how a broken route
stays alive.

Every durable update carries the identity of the verifier that justified
it and an inverse. When evidence is later invalidated, everything that
rested on it is rolled back — the property that makes this a governance
system rather than a logging convention.
"""
from __future__ import annotations

import enum
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from core.runtime.errors import record_degradation
from core.runtime.turn_outcome import VerificationGrade
from core.runtime.lockdep import checked_lock

__all__ = [
    "LearningScope",
    "VerificationGrade",
    "LearningUpdate",
    "Admission",
    "DurableLearningGate",
    "get_durable_learning_gate",
    "admit_learning_update",
]


class LearningScope(str, enum.Enum):
    """How far a proposed update is allowed to reach."""

    #: Applies now, affects this session's behaviour, never persisted.
    #: The honest home for "it seemed to work" — useful, not believed.
    SESSION = "session"
    #: Persisted, and allowed to change future behaviour.
    DURABLE = "durable"
    #: Recorded but NOT applied. Calibration and verifier-training data
    #: land here first: data that teaches a judge must be reviewed before
    #: it teaches the judge, or a bad verifier certifies its own successors.
    QUARANTINE = "quarantine"
    #: Refused outright. Malformed, unattributed, or contradicted.
    REJECTED = "rejected"


#: Grades that may alter durable behaviour on a SUCCESS. Below this line a
#: success is a claim, not a finding.
_DURABLE_SUCCESS_FLOOR = VerificationGrade.POSTCONDITION_VERIFIED

#: Grades that may alter durable behaviour on a FAILURE. Lower on purpose:
#: an observed failure is worth acting on, and demanding counterfactual
#: proof before weakening a broken route is how broken routes survive.
_DURABLE_FAILURE_FLOOR = VerificationGrade.OBSERVED

#: Subsystems whose data trains a judge. Quarantined regardless of grade:
#: the risk is not that the datum is wrong, it is that a wrong datum
#: becomes the standard by which later data is judged.
_QUARANTINED_SUBSYSTEMS = frozenset(
    {
        "calibration",
        "calibration_gate",
        "verifier_training",
        "verifier_foundry",
        "confidence_calibration",
        "reward_model",
    }
)

#: Ledger bound. Rollback needs history, but an unbounded durable ledger is
#: a disk leak on a machine that runs for weeks.
_MAX_LEDGER_ENTRIES = 20_000


@dataclass(frozen=True)
class LearningUpdate:
    """A proposed change to something Aura will still believe tomorrow."""

    subsystem: str
    key: str
    #: What the update does, for the audit trail: "reinforce", "weaken",
    #: "set_confidence". Free-form but recorded.
    operation: str
    #: True when the update encodes "this worked". Drives which floor
    #: applies — success and failure are held to different bars on purpose.
    success: bool
    grade: VerificationGrade
    #: WHO checked. A durable update with no named verifier is refused:
    #: unattributed evidence cannot be invalidated later, so it cannot be
    #: rolled back, so it must never have been believed.
    verifier: str | None = None
    #: Identity of the evidence itself, so invalidating it can find every
    #: update that rested on it.
    evidence_id: str | None = None
    #: Applied to reverse this update. Absent means irreversible, which is
    #: refused durably — see the class docstring.
    inverse: Mapping[str, Any] | None = None
    #: A signed verdict from the independent evidence service. Required for
    #: the two strongest grades: below them a caller's word is checkable
    #: against a receipt, but "counterfactually verified" and "externally
    #: verified" are claims about work a caller cannot have done alone, and
    #: accepting them on assertion would put the strongest evidence tier on
    #: the weakest footing in the system.
    verdict: Any = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        return f"{self.subsystem}:{self.key}:{self.operation}"


@dataclass(frozen=True)
class Admission:
    """The gate's verdict on one proposed update."""

    scope: LearningScope
    reason: str
    update: LearningUpdate
    record_id: str | None = None

    @property
    def is_durable(self) -> bool:
        return self.scope is LearningScope.DURABLE

    @property
    def applies_now(self) -> bool:
        """Whether the caller should apply the change at all.

        Session updates apply; quarantined and rejected ones do not.
        """
        return self.scope in (LearningScope.DURABLE, LearningScope.SESSION)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope.value,
            "reason": self.reason,
            "subsystem": self.update.subsystem,
            "key": self.update.key,
            "operation": self.update.operation,
            "success": self.update.success,
            "grade": self.update.grade.value,
            "verifier": self.update.verifier,
            "evidence_id": self.update.evidence_id,
            "record_id": self.record_id,
        }


def _verdict_supports(update: LearningUpdate) -> tuple[bool, str]:
    """Whether a signed, certified verdict actually backs the claimed grade.

    The seam between "what evidence exists" (core.brain.verification) and
    "what Aura is allowed to keep" (here). A caller may state any grade it
    likes; at the top two tiers the gate goes and looks.
    """
    verdict = update.verdict
    if verdict is None:
        return False, "top_grade_claimed_without_an_independent_verdict"
    try:
        from core.brain.verification.independent_evidence import (
            VerdictStatus,
            verdict_signature_valid,
        )
    except ImportError as exc:
        record_degradation(
            "durable_learning",
            exc,
            severity="degraded",
            action=(
                "kept a top-grade update session-local because the independent "
                "evidence service could not be loaded to check its verdict"
            ),
        )
        return False, "evidence_service_unavailable"
    if not verdict_signature_valid(verdict):
        return False, "verdict_signature_invalid"
    if getattr(verdict, "status", None) is not VerdictStatus.CERTIFIED:
        return False, "verdict_did_not_certify"
    if getattr(verdict, "grade", None) < update.grade:
        # The verdict is real but establishes less than the caller claimed.
        return False, "verdict_grade_below_the_claimed_grade"
    return True, "verdict_supports_the_claimed_grade"


class DurableLearningGate:
    """Decides scope, records justification, and can undo what it admitted.

    Thread-safe. Learning updates arrive from the response lane, background
    cognition and maintenance threads at once.
    """

    def __init__(self, *, ledger_path: Path | str | None = None) -> None:
        self._lock = checked_lock("durable_learning.instance", reentrant=True)
        self._ledger: list[dict[str, Any]] = []
        self._by_evidence: dict[str, list[str]] = {}
        self._invalidated: set[str] = set()
        self._quarantine: list[dict[str, Any]] = []
        self._counts: dict[str, int] = {scope.value: 0 for scope in LearningScope}
        self._ledger_path = Path(ledger_path) if ledger_path else None

    # ------------------------------------------------------------------ admission

    def admit(self, update: LearningUpdate) -> Admission:
        """Decide how far this update may reach. The only entry point."""
        verdict = self._classify(update)
        if verdict is not None:
            scope, reason = verdict
        else:
            scope, reason = self._scope_for_grade(update)

        record_id: str | None = None
        with self._lock:
            self._counts[scope.value] = self._counts.get(scope.value, 0) + 1
            if scope is LearningScope.DURABLE:
                record_id = self._append_ledger(update, reason)
            elif scope is LearningScope.QUARANTINE:
                record_id = self._append_quarantine(update, reason)

        return Admission(scope=scope, reason=reason, update=update, record_id=record_id)

    def _classify(self, update: LearningUpdate) -> tuple[LearningScope, str] | None:
        """Structural refusals, checked before any grade is trusted.

        Returns None when nothing structural applies and the grade decides.
        """
        if not str(update.subsystem or "").strip():
            return LearningScope.REJECTED, "no_subsystem"
        if not str(update.key or "").strip():
            return LearningScope.REJECTED, "no_key"
        if not isinstance(update.success, bool):
            # The "false" -> True coercion bug, refused at the type level
            # rather than trusted through bool().
            return LearningScope.REJECTED, "success_is_not_a_bool"
        if not isinstance(update.grade, VerificationGrade):
            return LearningScope.REJECTED, "grade_is_not_a_verification_grade"
        if update.subsystem.strip().lower() in _QUARANTINED_SUBSYSTEMS:
            return (
                LearningScope.QUARANTINE,
                "trains_a_judge_and_must_be_reviewed_before_it_teaches_one",
            )
        if update.evidence_id and update.evidence_id in self._invalidated:
            return LearningScope.REJECTED, "evidence_already_invalidated"
        return None

    def _scope_for_grade(self, update: LearningUpdate) -> tuple[LearningScope, str]:
        floor = _DURABLE_FAILURE_FLOOR if not update.success else _DURABLE_SUCCESS_FLOOR
        if update.grade < floor:
            return (
                LearningScope.SESSION,
                f"grade_{update.grade.value}_below_durable_floor_{floor.value}",
            )
        # Above the floor. Durability additionally requires attribution and
        # reversibility, because a durable belief that cannot be traced or
        # undone is one Aura is stuck with.
        if not str(update.verifier or "").strip():
            return LearningScope.SESSION, "durable_grade_but_no_named_verifier"
        if update.inverse is None:
            return LearningScope.SESSION, "durable_grade_but_no_inverse_to_roll_back"
        if update.grade >= VerificationGrade.COUNTERFACTUALLY_VERIFIED:
            supported, why = _verdict_supports(update)
            if not supported:
                return LearningScope.SESSION, why
        return LearningScope.DURABLE, f"grade_{update.grade.value}_meets_{floor.value}"

    # ------------------------------------------------------------------- rollback

    def invalidate_evidence(self, evidence_id: str) -> list[dict[str, Any]]:
        """Withdraw evidence and return the updates that rested on it.

        Returns each affected ledger entry with its ``inverse`` so the
        owning subsystem can undo it. The gate does not reach into other
        subsystems' state — it knows what to undo, not how, and a gate that
        mutated foreign state would be the coupling it exists to prevent.

        Idempotent: invalidating twice returns the same set and does not
        double-undo.
        """
        key = str(evidence_id or "").strip()
        if not key:
            return []
        with self._lock:
            already = key in self._invalidated
            self._invalidated.add(key)
            record_ids = list(self._by_evidence.get(key, ()))
            affected = [
                entry
                for entry in self._ledger
                if entry["record_id"] in record_ids and not entry.get("rolled_back")
            ]
            if not already:
                for entry in affected:
                    entry["rolled_back"] = True
                    entry["rolled_back_at"] = time.time()
        if affected:
            record_degradation(
                "durable_learning",
                RuntimeError(f"evidence {key} invalidated"),
                severity="warning",
                action=(
                    f"withdrew {len(affected)} durable learning update(s) that rested "
                    "on invalidated evidence; owners must apply the returned inverses"
                ),
                extra={"evidence_id": key, "affected": len(affected)},
            )
        return affected

    def is_invalidated(self, evidence_id: str) -> bool:
        with self._lock:
            return str(evidence_id or "") in self._invalidated

    # -------------------------------------------------------------------- reading

    def durable_updates(self, *, include_rolled_back: bool = False) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return tuple(
                entry
                for entry in self._ledger
                if include_rolled_back or not entry.get("rolled_back")
            )

    def quarantined(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return tuple(self._quarantine)

    def release_from_quarantine(
        self, record_id: str, *, reviewer: str
    ) -> dict[str, Any] | None:
        """Promote a quarantined datum after review. Requires a named reviewer.

        Anonymous release would make quarantine a delay rather than a check.
        """
        if not str(reviewer or "").strip():
            raise ValueError("release_from_quarantine requires a named reviewer")
        with self._lock:
            for entry in self._quarantine:
                if entry["record_id"] == record_id:
                    entry["released_by"] = str(reviewer)
                    entry["released_at"] = time.time()
                    return dict(entry)
        return None

    def report(self) -> dict[str, Any]:
        """Health-surface view: what has Aura been allowed to learn."""
        with self._lock:
            durable = [e for e in self._ledger if not e.get("rolled_back")]
            return {
                "admissions": dict(self._counts),
                "durable_updates": len(durable),
                "rolled_back": sum(1 for e in self._ledger if e.get("rolled_back")),
                "quarantined": len(self._quarantine),
                "quarantine_awaiting_review": sum(
                    1 for e in self._quarantine if not e.get("released_by")
                ),
                "invalidated_evidence": len(self._invalidated),
                "verifiers": sorted(
                    {str(e.get("verifier") or "") for e in durable if e.get("verifier")}
                ),
            }

    # ------------------------------------------------------------------ internals

    def _append_ledger(self, update: LearningUpdate, reason: str) -> str:
        record_id = uuid.uuid4().hex
        entry = {
            "record_id": record_id,
            "at": time.time(),
            "subsystem": update.subsystem,
            "key": update.key,
            "operation": update.operation,
            "success": update.success,
            "grade": update.grade.value,
            "verifier": update.verifier,
            "evidence_id": update.evidence_id,
            "inverse": dict(update.inverse or {}),
            "reason": reason,
            "rolled_back": False,
        }
        self._ledger.append(entry)
        if update.evidence_id:
            self._by_evidence.setdefault(update.evidence_id, []).append(record_id)
        while len(self._ledger) > _MAX_LEDGER_ENTRIES:
            dropped = self._ledger.pop(0)
            ids = self._by_evidence.get(dropped.get("evidence_id") or "")
            if ids and dropped["record_id"] in ids:
                ids.remove(dropped["record_id"])
        return record_id

    def _append_quarantine(self, update: LearningUpdate, reason: str) -> str:
        record_id = uuid.uuid4().hex
        self._quarantine.append(
            {
                "record_id": record_id,
                "at": time.time(),
                "subsystem": update.subsystem,
                "key": update.key,
                "operation": update.operation,
                "success": update.success,
                "grade": update.grade.value,
                "verifier": update.verifier,
                "evidence_id": update.evidence_id,
                "reason": reason,
                "released_by": None,
            }
        )
        while len(self._quarantine) > _MAX_LEDGER_ENTRIES:
            self._quarantine.pop(0)
        return record_id

    async def persist_async(self, *, source: str = "durable_learning") -> str | None:
        """Write the durable ledger through the governed write gateway.

        Async-only: an on-loop fsync once froze the live event loop for
        twenty minutes, and this runs from the response lane.
        """
        if self._ledger_path is None:
            return None
        with self._lock:
            payload = {
                "version": 1,
                "written_at": time.time(),
                "ledger": list(self._ledger),
                "quarantine": list(self._quarantine),
                "invalidated": sorted(self._invalidated),
            }
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            with local_internal_governed_scope(source, domain="state_mutation"):
                return await get_file_write_gateway().write_json_async(
                    self._ledger_path, payload, source=source
                )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "durable_learning",
                exc,
                severity="degraded",
                action=(
                    "kept the durable learning ledger in memory after the governed "
                    "write failed; rollback still works this process, not across a restart"
                ),
            )
            return None

    def load(self) -> bool:
        """Restore a persisted ledger. Returns whether anything was read."""
        if self._ledger_path is None or not self._ledger_path.exists():
            return False
        try:
            payload = json.loads(self._ledger_path.read_text("utf-8"))
        except (OSError, ValueError) as exc:
            record_degradation(
                "durable_learning",
                exc,
                severity="degraded",
                action="started with an empty durable learning ledger after a read failure",
            )
            return False
        if not isinstance(payload, dict):
            return False
        with self._lock:
            self._ledger = [e for e in payload.get("ledger", []) if isinstance(e, dict)]
            self._quarantine = [
                e for e in payload.get("quarantine", []) if isinstance(e, dict)
            ]
            self._invalidated = {
                str(e) for e in payload.get("invalidated", []) if isinstance(e, str)
            }
            self._by_evidence = {}
            for entry in self._ledger:
                evidence = entry.get("evidence_id")
                if evidence:
                    self._by_evidence.setdefault(str(evidence), []).append(
                        str(entry.get("record_id") or "")
                    )
        return True


_GATE: DurableLearningGate | None = None
_GATE_LOCK = checked_lock("durable_learning.gate")


def get_durable_learning_gate() -> DurableLearningGate:
    """The process-wide gate.

    Its ledger path comes from ``AURA_STATE_ROOT`` when set, so test and
    bench runs cannot write into the live learning ledger.
    """
    global _GATE
    with _GATE_LOCK:
        if _GATE is None:
            root = os.environ.get("AURA_STATE_ROOT") or ""
            path = Path(root).expanduser() / "durable_learning.json" if root else None
            _GATE = DurableLearningGate(ledger_path=path)
            _GATE.load()
        return _GATE


def admit_learning_update(
    subsystem: str,
    key: str,
    *,
    operation: str,
    success: bool,
    grade: VerificationGrade,
    verifier: str | None = None,
    evidence_id: str | None = None,
    inverse: Mapping[str, Any] | None = None,
    verdict: Any = None,
    payload: Mapping[str, Any] | None = None,
) -> Admission:
    """Ask the gate how far one proposed learning update may reach.

    The call every learning site makes before it changes anything it keeps.
    """
    return get_durable_learning_gate().admit(
        LearningUpdate(
            subsystem=subsystem,
            key=key,
            operation=operation,
            success=success,
            grade=grade,
            verifier=verifier,
            evidence_id=evidence_id,
            inverse=inverse,
            verdict=verdict,
            payload=dict(payload or {}),
        )
    )


def grade_from_evidence(evidence: Any, *, success: bool) -> VerificationGrade:
    """Read an honest grade off a caller's evidence object.

    Deliberately stingy. Anything that does not carry an explicit outcome
    AGREEING with the claim grades ``ASSERTED`` at best — which is below
    every durable floor. This is the function that stops "nothing threw"
    from becoming a durable belief.
    """
    if evidence is None:
        return VerificationGrade.ASSERTED
    if isinstance(evidence, bool):
        # A second boolean is not corroboration.
        return VerificationGrade.ASSERTED

    declared = None
    outcome: Any = None
    if isinstance(evidence, Mapping):
        declared = evidence.get("verification_grade") or evidence.get("grade")
        for candidate in ("verified_success", "ok", "success"):
            if candidate in evidence:
                outcome = evidence[candidate]
                break
    else:
        declared = getattr(evidence, "verification_grade", None) or getattr(
            evidence, "grade", None
        )
        for candidate in ("verified_success", "ok", "success"):
            if hasattr(evidence, candidate):
                outcome = getattr(evidence, candidate)
                break

    if not isinstance(outcome, bool) or outcome is not bool(success):
        # No outcome field, or one that contradicts the caller. Evidence
        # that disagrees with the claim is not evidence for the claim.
        return VerificationGrade.ASSERTED

    if isinstance(declared, VerificationGrade):
        return declared
    if isinstance(declared, str):
        try:
            return VerificationGrade(declared)
        except ValueError:
            return VerificationGrade.OBSERVED
    return VerificationGrade.OBSERVED


def summarize_admissions(admissions: Iterable[Admission]) -> dict[str, int]:
    counts: dict[str, int] = {scope.value: 0 for scope in LearningScope}
    for admission in admissions:
        counts[admission.scope.value] = counts.get(admission.scope.value, 0) + 1
    return counts
