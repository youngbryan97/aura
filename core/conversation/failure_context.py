"""core/conversation/failure_context.py — failures she can talk about.

A capability fails and the user gets a sentence written months earlier by
whoever touched that module last. "I'm unable to browse the web right now."
It is accurate, it is useless, and it is unmistakably not her — the register
changes mid-conversation, which is precisely the tell that makes assistants
feel like switchboards. The recurring complaint about every prior generation
of voice assistant is some variant of "I can't help with that", and the
reason it grates is not that the assistant failed. It is that it stopped
being a participant in the conversation at the moment the failure happened.

The fix is a division of labour, and it is the whole idea here:

    **The runtime supplies the facts. She supplies the words.**

A module that fails does not get to write dialogue. It records what it
tried, what stopped it, how it knows, and what is still possible — and then
her next turn is told those facts and says them the way she says anything
else. "I can't get out to the network right now — the DNS probe's been
failing for the last four minutes, so search is out, but I still have
everything we've talked about locally" is not a string in any file. It is
what she says when she is handed four true things.

Three rules keep this from becoming a nicer flavour of canned:

1. **Facts, never phrasing.** Nothing in this module is written to be read
   aloud. If a field starts to look like a sentence she should say, it is a
   canned response with extra steps.

2. **The evidence travels with the claim.** ``detail`` carries the actual
   reading — the probe that failed, the exit code, the missing binary. She
   cannot honestly say why something failed unless she is told why, and
   without it she will do what any language model does with a gap, which is
   fill it plausibly.

3. **What remains is part of the failure.** A failure report that lists only
   what broke invites her to over-generalise from it ("I'm offline" when one
   host is unreachable). ``still_possible`` is what keeps a bounded failure
   bounded.

The ledger is turn-scoped through a ContextVar, so records raised while
serving one person cannot surface in someone else's reply, and a probe or a
background task cannot inject context into a live user's turn.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from core.runtime.errors import record_degradation
from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Aura.Conversation.FailureContext")

# More than this in one turn and the prompt block stops being context and
# starts being a log dump. The most recent are the ones that explain what
# just happened, so the oldest are dropped.
_MAX_RECORDS_PER_TURN = 6

# Why something did not work. The taxonomy is deliberately about *what the
# user can do next*, not about which exception fired — she is explaining a
# situation, not reading a stack trace.
CAUSES = (
    "offline",  # no usable network path
    "unauthorized",  # exists, but this principal may not
    "unavailable",  # the service or device is down or absent
    "not_installed",  # the tool genuinely is not on this machine
    "timeout",  # it was reachable and did not finish in time
    "refused",  # her own governance declined it
    "empty_result",  # it worked and there was nothing there
    "failed",  # ran and errored; detail carries the specifics
)


@dataclass(slots=True)
class CapabilityFailure:
    """One thing that did not work, in facts rather than phrasing."""

    capability: str
    intent: str
    cause: str
    detail: str = ""
    still_possible: tuple[str, ...] = ()
    at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        cause = str(self.cause or "").strip().lower()
        self.cause = cause if cause in CAUSES else "failed"
        self.capability = " ".join(str(self.capability or "unknown").split())[:64]
        self.intent = " ".join(str(self.intent or "").split())[:200]
        self.detail = " ".join(str(self.detail or "").split())[:400]
        self.still_possible = tuple(
            " ".join(str(item).split())[:120]
            for item in (self.still_possible or ())
            if str(item).strip()
        )[:5]

    def as_facts(self) -> str:
        """One line of ground truth. Not a sentence to be spoken."""
        parts = [f"tried: {self.intent or self.capability}", f"stopped by: {self.cause}"]
        if self.detail:
            parts.append(f"reading: {self.detail}")
        if self.still_possible:
            parts.append("still works: " + "; ".join(self.still_possible))
        return " | ".join(parts)

    def to_dict(self) -> dict[str, object]:
        return {
            "capability": self.capability,
            "intent": self.intent,
            "cause": self.cause,
            "detail": self.detail,
            "still_possible": list(self.still_possible),
            "at": self.at,
        }


class FailureLedger:
    """Everything that failed while serving one turn."""

    __slots__ = ("_closed", "_lock", "_owner", "_records")

    def __init__(self) -> None:
        self._records: list[CapabilityFailure] = []
        self._lock = checked_lock("core.conversation.failure_context", reentrant=True)
        self._owner = _execution_identity()
        self._closed = False

    def admits_current_execution(self) -> bool:
        try:
            from core.conversation.turn_evidence_custody import (
                current_turn_evidence_custody,
            )

            custody = current_turn_evidence_custody()
        except (ImportError, RuntimeError):
            custody = None
        if custody is not None:
            return custody.admits_current_execution()
        return _execution_identity() == self._owner

    def record(self, failure: CapabilityFailure) -> bool:
        if not self.admits_current_execution():
            return False
        with self._lock:
            if self._closed:
                return False
            self._records.append(failure)
            del self._records[:-_MAX_RECORDS_PER_TURN]
            return True

    def drain(self) -> list[CapabilityFailure]:
        if not self.admits_current_execution():
            return []
        with self._lock:
            records = list(self._records)
            self._records.clear()
            return records

    @property
    def records(self) -> tuple[CapabilityFailure, ...]:
        if not self.admits_current_execution():
            return ()
        with self._lock:
            return tuple(self._records)

    def __bool__(self) -> bool:
        return bool(self.records)

    def close(self) -> None:
        with self._lock:
            self._closed = True


def _execution_identity() -> tuple[int, int]:
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return (threading.get_ident(), id(task) if task is not None else 0)


_ACTIVE_LEDGER: ContextVar[FailureLedger | None] = ContextVar(
    "aura_active_failure_ledger", default=None
)


@contextmanager
def bind_failure_ledger(ledger: FailureLedger | None = None) -> Iterator[FailureLedger]:
    """Collect capability failures for the duration of one turn."""
    active = ledger if ledger is not None else FailureLedger()
    token = _ACTIVE_LEDGER.set(active)
    try:
        yield active
    finally:
        active.close()
        _ACTIVE_LEDGER.reset(token)


def active_failure_ledger() -> FailureLedger | None:
    return _ACTIVE_LEDGER.get()


def record_capability_failure(
    capability: str,
    *,
    intent: str,
    cause: str,
    detail: str = "",
    still_possible: Sequence[str] = (),
) -> CapabilityFailure | None:
    """Record a failure as facts for her to narrate.

    Returns the record, or None when nothing is collecting — a background
    task's failure is not part of anybody's turn, and injecting it into the
    next unrelated reply would be worse than dropping it. Callers that must
    not lose the failure should still call ``record_degradation`` as usual;
    the two are complementary, not alternatives. This one is for what she
    *says*, that one is for what the runtime *knows*.
    """
    ledger = _ACTIVE_LEDGER.get()
    if ledger is None:
        return None
    try:
        failure = CapabilityFailure(
            capability=capability,
            intent=intent,
            cause=cause,
            detail=detail,
            still_possible=tuple(still_possible or ()),
        )
    except (TypeError, ValueError) as exc:
        record_degradation(
            "conversation.failure_context",
            exc,
            action="dropped a malformed capability-failure record",
            severity="debug",
        )
        return None
    if not ledger.record(failure):
        return None
    logger.debug("capability failure recorded: %s", failure.as_facts())
    return failure


def render_failure_block(failures: Sequence[CapabilityFailure]) -> str:
    """Turn recorded facts into the context block her turn reasons over.

    The instruction is about *stance*, not wording. It tells her the facts are
    hers to explain and that she should not pretend the thing worked; it does
    not hand her a sentence, because a sentence handed over is a canned
    response no matter how good it is.
    """
    if not failures:
        return ""
    lines = [f"  - {failure.as_facts()}" for failure in failures]
    return (
        "[what just failed, as facts rather than phrasing:\n"
        + "\n".join(lines)
        + "\nThese are real readings from this machine, taken during this turn. "
        "Say what happened in your own words, as part of the answer rather than "
        "as an error notice — the same way you would mention any other thing you "
        "found out. Be specific about the cause; you know it, so vagueness here "
        "reads as evasion. Do not claim the attempt succeeded, do not invent a "
        "result you did not get, and do not generalise past what actually failed "
        "— if something is still available, it is still available.]"
    )


def pending_failure_context() -> str:
    """The failure block for the current turn, if anything failed."""
    ledger = _ACTIVE_LEDGER.get()
    if ledger is None or not ledger:
        return ""
    return render_failure_block(ledger.records)


# ── the common case: the network is not there ────────────────────────────


def record_offline_failure(
    capability: str,
    *,
    intent: str,
    still_possible: Sequence[str] = (),
) -> CapabilityFailure | None:
    """Record "this needed the network and there is none" with real evidence.

    The evidence is the probe's own last reading rather than an assumption.
    "I think I'm offline" and "the DNS probe to 1.1.1.1:53 has been failing
    for four minutes" are different claims, and only one of them is something
    she actually knows.
    """
    detail = "no connectivity reading available"
    try:
        from core.runtime.connectivity import get_connectivity_status

        status = get_connectivity_status()
        age_s = max(0.0, time.time() - float(status.checked_at))
        detail = (
            f"connectivity probe to {status.target} reports "
            f"{'online' if status.online else 'offline'}"
            f" (mode {status.mode}, checked {age_s:.0f}s ago)"
        )
        if status.reason:
            detail += f"; {status.reason}"
    except (RuntimeError, AttributeError, TypeError, ValueError, ImportError, OSError) as exc:
        record_degradation(
            "conversation.failure_context",
            exc,
            action="recorded the offline failure without a connectivity reading",
            severity="debug",
        )
    return record_capability_failure(
        capability,
        intent=intent,
        cause="offline",
        detail=detail,
        still_possible=still_possible,
    )


__all__ = [
    "CAUSES",
    "CapabilityFailure",
    "FailureLedger",
    "active_failure_ledger",
    "bind_failure_ledger",
    "pending_failure_context",
    "record_capability_failure",
    "record_offline_failure",
    "render_failure_block",
]
