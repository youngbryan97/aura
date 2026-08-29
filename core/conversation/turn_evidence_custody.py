"""Exact ownership for evidence produced while serving one conversation turn.

``ContextVar`` answers *which turn is this code running under?* It does not
answer *may this task write evidence for that turn?* Async tasks inherit a
copy of their parent's context, so putting a mutable collector in a ContextVar
lets every background task spawned during a request mutate the foreground
reply's evidence. Making the value immutable fixes contamination but also
makes legitimate child-task writes invisible to the parent.

This module separates the two questions. A custody object is shared by the
turn, but only the owner execution may use it. A deliberate child receives a
one-use lease from the owner and joins explicitly. Incidental background tasks,
thread-pool hops, and other turns inherit no authority merely because Python
copied their context.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import math
import secrets
import threading
import time
from collections.abc import Awaitable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from core.conversation.session_scope import (
    conversation_session_var,
    conversation_turn_var,
    current_conversation_session,
    current_conversation_turn,
    normalize_conversation_id,
    normalize_conversation_turn_id,
)
from core.runtime.lockdep import checked_lock

__all__ = [
    "EvidenceParticipantLease",
    "TurnEvidenceCustody",
    "bind_turn_evidence_custody",
    "current_turn_evidence_custody",
    "join_turn_evidence_custody",
    "record_turn_capability_availability",
    "record_turn_grounding",
    "record_turn_sensory_evidence",
    "run_as_turn_evidence_participant",
    "turn_capability_availability",
    "turn_grounding_evidence",
    "turn_sensory_evidence",
]


def _execution_identity() -> tuple[int, int]:
    """Stable identity for one thread/task execution locus."""

    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return (threading.get_ident(), id(task) if task is not None else 0)


_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EvidenceParticipantLease:
    """One-use capability allowing an intentional child to join a turn."""

    token: str
    session_id: str
    turn_id: str
    purpose: str
    issued_at: float


class TurnEvidenceCustody:
    """Synchronized evidence owned by one exact session/turn/task tree."""

    def __init__(self, *, session_id: str, turn_id: str) -> None:
        session = normalize_conversation_id(session_id)
        turn = normalize_conversation_turn_id(turn_id)
        if not session or not turn:
            raise ValueError("turn evidence custody requires exact session and turn identities")
        self.session_id = session
        self.turn_id = turn
        self.started_at = time.time()
        self._lock = checked_lock("core.conversation.turn_evidence_custody", reentrant=True)
        self._owner = _execution_identity()
        self._participants: set[tuple[int, int]] = {self._owner}
        self._leases: dict[str, EvidenceParticipantLease] = {}
        self._receipts: list[dict[str, Any]] = []
        self._grounding: list[str] = []
        self._sensory_evidence: dict[str, dict[str, Any]] = {}
        self._capability_availability: dict[str, dict[str, Any]] = {}
        self._closed = False

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def _identity_matches(self) -> bool:
        return (
            current_conversation_session() == self.session_id
            and current_conversation_turn() == self.turn_id
        )

    def admits_current_execution(self) -> bool:
        with self._lock:
            return (
                not self._closed
                and self._identity_matches()
                and _execution_identity() in self._participants
            )

    def issue_child_lease(self, purpose: str) -> EvidenceParticipantLease:
        """Issue a one-use lease; ambient child tasks receive no authority."""

        if not self.admits_current_execution():
            raise PermissionError("only an admitted turn participant may issue an evidence lease")
        lease = EvidenceParticipantLease(
            token=secrets.token_urlsafe(24),
            session_id=self.session_id,
            turn_id=self.turn_id,
            purpose=" ".join(str(purpose or "turn child").split())[:120],
            issued_at=time.time(),
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("turn evidence custody is closed")
            self._leases[lease.token] = lease
        return lease

    @contextmanager
    def join(self, lease: EvidenceParticipantLease) -> Iterator[TurnEvidenceCustody]:
        """Consume ``lease`` and admit this exact child for the block."""

        participant = _execution_identity()
        with self._lock:
            issued = self._leases.pop(str(getattr(lease, "token", "") or ""), None)
            if (
                self._closed
                or issued is None
                or issued != lease
                or lease.session_id != self.session_id
                or lease.turn_id != self.turn_id
                or not self._identity_matches()
            ):
                raise PermissionError("invalid, stale, or cross-turn evidence participant lease")
            self._participants.add(participant)
        try:
            yield self
        finally:
            with self._lock:
                if participant != self._owner:
                    self._participants.discard(participant)

    def clear_receipts(self) -> bool:
        if not self.admits_current_execution():
            return False
        with self._lock:
            self._receipts.clear()
        return True

    def append_receipt(self, receipt: dict[str, Any]) -> bool:
        if not self.admits_current_execution():
            return False
        row = dict(receipt)
        row["session_id"] = self.session_id
        row["turn_id"] = self.turn_id
        with self._lock:
            if self._closed:
                return False
            if len(self._receipts) < 64:
                self._receipts.append(row)
                return True
        return False

    def receipts(self) -> tuple[dict[str, Any], ...]:
        if not self.admits_current_execution():
            return ()
        with self._lock:
            return tuple(dict(item) for item in self._receipts)

    def append_grounding(self, evidence: Any) -> bool:
        """Attach source text that this exact turn was entitled to use."""

        if not self.admits_current_execution():
            return False
        text = str(evidence or "").strip()
        if not text:
            return False
        with self._lock:
            if self._closed:
                return False
            if text not in self._grounding and len(self._grounding) < 32:
                self._grounding.append(text[:16_000])
            return True

    def grounding(self) -> tuple[str, ...]:
        """Authenticated recall/evidence text admitted to the current turn."""

        if not self.admits_current_execution():
            return ()
        with self._lock:
            return tuple(self._grounding)

    def record_sensory_evidence(self, evidence: Any) -> bool:
        """Attach one typed sensor result to this exact turn.

        Prose grounding reaches the model, but a separate MLX worker cannot
        recover provenance from prose. Keeping the bounded typed receipt here
        lets every gate distinguish a real observation from model-authored
        text without treating a sense as a tool execution.
        """

        if not self.admits_current_execution() or not isinstance(evidence, dict):
            return False
        channel = str(evidence.get("channel") or "").strip().casefold()
        if channel not in {"camera", "microphone", "screen"}:
            return False
        if not isinstance(evidence.get("ok"), bool):
            return False
        try:
            observed_at = float(evidence.get("observed_at") or 0.0)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(observed_at) or observed_at <= 0.0:
            return False
        if bool(evidence.get("ok")) and not str(evidence.get("observation") or "").strip():
            return False
        row = dict(evidence)
        row["session_id"] = self.session_id
        row["turn_id"] = self.turn_id
        with self._lock:
            if self._closed:
                return False
            self._sensory_evidence[channel] = row
        return True

    def sensory_evidence(self) -> tuple[dict[str, Any], ...]:
        """Typed sensor results admitted to this exact turn."""

        if not self.admits_current_execution():
            return ()
        with self._lock:
            return tuple(
                dict(self._sensory_evidence[channel])
                for channel in sorted(self._sensory_evidence)
            )

    def record_capability_availability(
        self,
        capability: Any,
        *,
        available: bool,
        reason: Any = "",
        observed_at: float | None = None,
    ) -> bool:
        """Record a turn-bound observation, never a durable capability claim."""

        if not self.admits_current_execution():
            return False
        name = str(capability or "").strip().casefold()
        if not name:
            return False
        row = {
            "capability": name[:64],
            "available": bool(available),
            "reason": " ".join(str(reason or "").split())[:240],
            "observed_at": float(observed_at if observed_at is not None else time.time()),
            "session_id": self.session_id,
            "turn_id": self.turn_id,
        }
        with self._lock:
            if self._closed:
                return False
            self._capability_availability[name] = row
            return True

    def capability_availability(self) -> tuple[dict[str, Any], ...]:
        """Freshness-checkable availability observations for this exact turn."""

        if not self.admits_current_execution():
            return ()
        with self._lock:
            return tuple(
                dict(self._capability_availability[name])
                for name in sorted(self._capability_availability)
            )

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._leases.clear()
            self._participants.clear()


_ACTIVE_CUSTODY: contextvars.ContextVar[TurnEvidenceCustody | None] = contextvars.ContextVar(
    "aura_turn_evidence_custody",
    default=None,
)


def current_turn_evidence_custody() -> TurnEvidenceCustody | None:
    return _ACTIVE_CUSTODY.get()


def record_turn_grounding(evidence: Any) -> bool:
    """Attach recall evidence to the active turn, if this task owns it."""

    custody = current_turn_evidence_custody()
    return bool(custody and custody.append_grounding(evidence))


def turn_grounding_evidence() -> tuple[str, ...]:
    """Return only evidence owned by the current exact turn execution."""

    custody = current_turn_evidence_custody()
    return custody.grounding() if custody is not None else ()


def record_turn_sensory_evidence(evidence: Any) -> bool:
    """Attach typed sensory evidence under the active turn's custody."""

    custody = current_turn_evidence_custody()
    return bool(custody and custody.record_sensory_evidence(evidence))


def turn_sensory_evidence() -> tuple[dict[str, Any], ...]:
    """Return only typed sensory evidence owned by this execution."""

    custody = current_turn_evidence_custody()
    return custody.sensory_evidence() if custody is not None else ()


def record_turn_capability_availability(
    capability: Any,
    *,
    available: bool,
    reason: Any = "",
    observed_at: float | None = None,
) -> bool:
    """Record one current availability observation under turn custody."""

    custody = current_turn_evidence_custody()
    return bool(
        custody
        and custody.record_capability_availability(
            capability,
            available=available,
            reason=reason,
            observed_at=observed_at,
        )
    )


def turn_capability_availability() -> tuple[dict[str, Any], ...]:
    """Return exact-turn capability observations, never inherited ambient state."""

    custody = current_turn_evidence_custody()
    return custody.capability_availability() if custody is not None else ()


@contextmanager
def bind_turn_evidence_custody(
    *,
    session_id: str,
    turn_id: str,
) -> Iterator[TurnEvidenceCustody]:
    """Own evidence for exactly one conversation turn and close it on exit."""

    custody = TurnEvidenceCustody(session_id=session_id, turn_id=turn_id)
    session_token = conversation_session_var.set(custody.session_id)
    turn_token = conversation_turn_var.set(custody.turn_id)
    custody_token = _ACTIVE_CUSTODY.set(custody)
    try:
        yield custody
    finally:
        custody.close()
        _ACTIVE_CUSTODY.reset(custody_token)
        conversation_turn_var.reset(turn_token)
        conversation_session_var.reset(session_token)


@contextmanager
def join_turn_evidence_custody(
    lease: EvidenceParticipantLease,
) -> Iterator[TurnEvidenceCustody]:
    """Join the inherited custody only with an explicit parent-issued lease."""

    custody = current_turn_evidence_custody()
    if custody is None:
        raise RuntimeError("no turn evidence custody is active")
    with custody.join(lease):
        yield custody


def run_as_turn_evidence_participant(
    awaitable: Awaitable[Any],
    *,
    purpose: str,
) -> Awaitable[Any]:
    """Wrap a deliberate child coroutine with a one-use turn evidence lease."""

    custody = current_turn_evidence_custody()
    if custody is None:
        return awaitable
    if not custody.admits_current_execution():
        # Only an admitted participant may issue a lease, so a caller that is
        # itself unadmitted cannot pass one down and this returns what it was
        # given. It used to do that silently, and silence here looks exactly
        # like success: the child runs, writes receipts nobody accepts, and the
        # turn reports that its tools found nothing.
        #
        # LIVE, 2026-08-28: the tool loop was wrapped here and five file reads
        # were still refused, because the wrap itself was a no-op — the gate
        # calling it was not an admitted participant either.
        _logger.info(
            "🧾 no lease for %s: this execution is not itself an admitted "
            "participant, so the child inherits nothing",
            " ".join(str(purpose or "child").split())[:60],
        )
        return awaitable
    lease = custody.issue_child_lease(purpose)

    async def _run() -> Any:
        with join_turn_evidence_custody(lease):
            return await awaitable

    return _run()
