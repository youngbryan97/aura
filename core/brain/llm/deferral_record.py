"""Why the last generation came back empty.

The router defers background inference by returning ``""``. That is the right
behaviour — the local substrate holds one 32B model and a foreground turn must
win — but the empty string carries no reason, and every caller downstream has
to guess at one. The guesses are wrong in a specific, expensive way:

    RuntimeError: LLM returned no Python source; the model returned nothing at
    all.

The model was never asked. The reconstruction lane then reported "0/14 held-out
positions reproduced", blaming verification for a generation that never ran, and
the user was told the build had failed on quality when it had failed on
admission. That is the same failure class as a good answer discarded by a gate
and then reported as an infrastructure fault: the true cause exists, briefly, in
one function, and is thrown away before anyone who could report it sees it.

So the router writes the reason down here on its way out, and the same async
task holding an unexplained empty generation can ask. Deliberately tiny: one
context-local value, no history and no lock contention on the hot path. Most
callers use it only to explain an empty result. A caller that must distinguish
admission from failure consumes it with an exact origin and call-start bound,
so unrelated concurrent or stale deferrals cannot alter control flow.
"""
from __future__ import annotations

import time
from contextvars import ContextVar
from dataclasses import dataclass

# Older than this and the deferral almost certainly belongs to some other call.
# An empty generation is reported within milliseconds of the deferral that
# caused it; a stale reason attached to an unrelated emptiness would be a
# confident wrong answer, which is worse than no answer.
_FRESHNESS_S = 20.0


@dataclass(frozen=True)
class Deferral:
    """One refusal to run inference, with the reason the refuser gave."""

    origin: str
    reason: str
    at: float

    def describe(self) -> str:
        return (
            f"inference was deferred, not run: {self.reason}"
            + (f" (origin {self.origin})" if self.origin else "")
        )


_last: ContextVar[Deferral | None] = ContextVar("aura_last_llm_deferral", default=None)

#: The same deferral, outside the context tree.
#:
#: LIVE, 2026-08-22: the router recorded "no_endpoint_available_for_tier" and
#: the task engine, reading a moment later, found nothing — so it raised "LLM
#: returned empty or None response", reported planning as a FAILURE, and drove
#: frustration to 1.00 and the resilience state to strain. Every layer of that
#: cascade was working from an absence.
#:
#: A ContextVar set inside a child task does not propagate back: asyncio gives
#: children a COPY of the context. The router runs beneath the caller, so its
#: write landed in a copy the caller could never read. The same trap is
#: documented in core/conversation/session_scope.py, which solves it by
#: sharing a mutable container.
#:
#: This holder is process-wide, and the freshness, origin and not_before
#: filters below are what keep one call from reading another's refusal.
_shared: dict[str, Deferral | None] = {"entry": None}


def record_deferral(*, origin: str, reason: str) -> None:
    """Called by whoever returned the empty string, at the moment it did."""
    cleaned = " ".join(str(reason or "").split())[:200]
    if not cleaned:
        return
    entry = Deferral(
        origin=" ".join(str(origin or "").split())[:80],
        at=time.time(),
        reason=cleaned,
    )
    _last.set(entry)
    _shared["entry"] = entry


def last_deferral(
    *,
    now: float | None = None,
    origin: str | None = None,
    not_before: float | None = None,
) -> Deferral | None:
    """Return the current task's matching fresh deferral, when one exists."""
    entry = _last.get()
    if entry is None:
        # The router may have recorded it beneath this task, where a
        # ContextVar write cannot be seen from here.
        entry = _shared.get("entry")
    if entry is None:
        return None
    stamp = float(now if now is not None else time.time())
    if stamp - entry.at > _FRESHNESS_S:
        return None
    if not_before is not None and entry.at < float(not_before):
        return None
    if (
        origin is not None
        and entry.origin.strip().lower() != str(origin).strip().lower()
    ):
        return None
    return entry


def take_deferral(
    *,
    origin: str,
    not_before: float,
    now: float | None = None,
) -> Deferral | None:
    """Consume the exact deferral produced by one just-finished routed call."""
    entry = last_deferral(now=now, origin=origin, not_before=not_before)
    if entry is not None:
        _last.set(None)
        if _shared.get("entry") is entry:
            _shared["entry"] = None
    return entry


def explain_empty_generation(*, now: float | None = None) -> str:
    """A cause to append to an empty-generation error, or "" if unknown.

    Returns a clause, not a sentence, so callers keep their own phrasing and
    this only ever adds the part they could not know.
    """
    entry = last_deferral(now=now)
    return entry.describe() if entry else ""


def reset_for_test() -> None:
    _last.set(None)
    _shared["entry"] = None


__all__ = [
    "Deferral",
    "explain_empty_generation",
    "last_deferral",
    "record_deferral",
    "reset_for_test",
    "take_deferral",
]
