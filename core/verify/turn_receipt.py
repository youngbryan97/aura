"""core/verify/turn_receipt.py

Which path actually produced this reply, and which phases actually ran.

A working demo does not establish that the claimed architecture caused the
behaviour. Aura can answer a user through at least four routes that look
identical from outside:

    the full sequential phase pipeline;

    ``_direct_desktop_quick_reply``, which assembles context, calls the model
    once, and returns before a single phase executes;

    a canonical structured floor, which returns pre-rendered text and never
    calls the model at all;

    and reactive recovery after a timeout or a crash.

All four produce a fluent reply with a ``response_path`` string the code writes
about itself. None of them, on their own, tell a reader whether affect, qualia,
Φ, the global workspace or planning had anything to do with the answer.

This records it. The phases are marked off as they execute, the response path is
marked when the turn commits to one, and ``full_pipeline_ran`` is derived from
comparing what ran against what was registered — there is no way to set it. A
turn that skipped the pipeline says so, and says which phases it skipped.

Per-turn state lives in a ContextVar so concurrent turns cannot write into each
other's receipts.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from core.runtime.lockdep import checked_lock

__all__ = [
    "TurnReceipt",
    "recording_turn",
    "record_phase",
    "record_response_path",
    "record_latency",
    "record_model_generation",
    "current_receipt",
    "recent_receipts",
    "reset_turn_receipts_for_test",
]


@dataclass
class TurnReceipt:
    """What one turn actually did."""

    turn_id: str
    #: Every phase registered on the engine for this turn.
    phases_available: tuple[str, ...] = ()
    #: Phases that actually executed, in execution order.
    phases_executed: list[str] = field(default_factory=list)
    #: The lane that produced the reply. Written once, when the turn commits.
    response_path: str = "unresolved"
    #: Whether the foreground model was called at all. False for the canonical
    #: floors, which return text assembled by the code — a reply that never
    #: passed through the model is not evidence of anything the model does.
    model_generation: bool = False
    started_at: float = field(default_factory=lambda: time.time())
    finished_at: float | None = None
    notes: list[str] = field(default_factory=list)
    #: Where the turn's time went, in seconds, each from the thing that
    #: measured it: queue (admission to the first worker activity), retrieval
    #: (the memory retrieval phase), prefill and decode (MLX's own timing in
    #: the worker), tool (time inside tool execution), delivery. A component
    #: nobody measured is absent, not zero — a zero would read as "instant".
    #: A component measured more than once on a turn (a repair retry, a
    #: second generation) accumulates, and `latency_samples` says how often.
    latency_s: dict[str, float] = field(default_factory=dict)
    latency_samples: dict[str, int] = field(default_factory=dict)

    @property
    def full_pipeline_ran(self) -> bool:
        """Derived, never set. Every registered phase executed for this turn."""

        if not self.phases_available:
            return False
        return set(self.phases_executed) >= set(self.phases_available)

    @property
    def phases_skipped(self) -> tuple[str, ...]:
        executed = set(self.phases_executed)
        return tuple(p for p in self.phases_available if p not in executed)

    @property
    def coverage(self) -> float:
        if not self.phases_available:
            return 0.0
        return len(set(self.phases_executed)) / len(set(self.phases_available))

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "response_path": self.response_path,
            "full_pipeline_ran": self.full_pipeline_ran,
            "model_generation": self.model_generation,
            "phase_coverage": round(self.coverage, 4),
            "phases_available": list(self.phases_available),
            "phases_executed": list(self.phases_executed),
            "phases_skipped": list(self.phases_skipped),
            "duration_s": (
                round(self.finished_at - self.started_at, 4)
                if self.finished_at is not None
                else None
            ),
            "latency_s": {name: round(seconds, 4) for name, seconds in self.latency_s.items()},
            "latency_samples": dict(self.latency_samples),
            "unattributed_s": self.unattributed_s,
            "notes": list(self.notes),
        }

    @property
    def unattributed_s(self) -> float | None:
        """The turn's wall time nothing above accounts for. Derived, never set."""
        if self.finished_at is None:
            return None
        total = self.finished_at - self.started_at
        return round(max(0.0, total - sum(self.latency_s.values())), 4)


_CURRENT: ContextVar[TurnReceipt | None] = ContextVar("aura_turn_receipt", default=None)
_RECENT: list[TurnReceipt] = []
_RECENT_LOCK = checked_lock("turn_receipt.recent")
_RECENT_LIMIT = 64


@contextmanager
def recording_turn(
    turn_id: str,
    *,
    phases_available: Sequence[str],
) -> Iterator[TurnReceipt]:
    """Record what one turn does, for the duration of the turn."""

    receipt = TurnReceipt(
        turn_id=str(turn_id),
        phases_available=tuple(str(p) for p in phases_available),
    )
    token = _CURRENT.set(receipt)
    try:
        yield receipt
    finally:
        receipt.finished_at = time.time()
        try:
            _CURRENT.reset(token)
        except ValueError:
            _CURRENT.set(None)
        with _RECENT_LOCK:
            _RECENT.append(receipt)
            if len(_RECENT) > _RECENT_LIMIT:
                del _RECENT[: len(_RECENT) - _RECENT_LIMIT]


def record_phase(name: str) -> None:
    """Mark one phase as executed. Called from the phase loop, not asserted."""

    receipt = _CURRENT.get()
    if receipt is not None:
        receipt.phases_executed.append(str(name))


def record_response_path(path: str, *, model_generation: bool) -> None:
    """Name the lane that produced the reply, and whether the model was called."""

    receipt = _CURRENT.get()
    if receipt is not None:
        receipt.response_path = str(path)
        receipt.model_generation = bool(model_generation)


#: The components a turn's latency is split into. A name outside this set is
#: refused, so the receipt cannot grow a vocabulary nobody reads.
LATENCY_COMPONENTS = frozenset({"queue", "retrieval", "prefill", "decode", "tool", "delivery"})


def record_latency(component: str, seconds: float) -> None:
    """Add a measured span to this turn's latency, by the thing that measured it.

    R11 asked for prefill, decode, tool, retrieval and queue measured
    separately. Prefill and decode were, in the worker; retrieval was, as a
    phase duration; tool time and queue time were not written down anywhere a
    turn could be read back from. One place now, and a component that was
    never measured stays absent rather than reading as zero.
    """
    name = str(component or "").strip().lower()
    if name not in LATENCY_COMPONENTS:
        raise ValueError(f"{component!r} is not a latency component: {sorted(LATENCY_COMPONENTS)}")
    try:
        span = float(seconds)
    except (TypeError, ValueError):
        return
    if not (span >= 0.0) or span != span:
        return
    receipt = _CURRENT.get()
    if receipt is None:
        return
    receipt.latency_s[name] = receipt.latency_s.get(name, 0.0) + span
    receipt.latency_samples[name] = receipt.latency_samples.get(name, 0) + 1


def record_model_generation() -> None:
    """Mark that a foreground model call happened on this turn."""

    receipt = _CURRENT.get()
    if receipt is not None:
        receipt.model_generation = True


def current_receipt() -> TurnReceipt | None:
    return _CURRENT.get()


def recent_receipts(limit: int = 16) -> list[dict[str, Any]]:
    """The last few turns, for health reporting and for answering honestly.

    This is what makes "did the full mind run?" a question with an answer
    rather than a claim.
    """

    with _RECENT_LOCK:
        return [r.as_dict() for r in _RECENT[-max(0, limit) :]]


def reset_turn_receipts_for_test() -> None:
    _CURRENT.set(None)
    with _RECENT_LOCK:
        _RECENT.clear()
