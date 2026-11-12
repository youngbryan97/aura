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

import logging
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from core.runtime.lockdep import checked_lock

__all__ = [
    "latency_by_component",
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
    #: When the thinking that joined this turn returned its answer; what follows is delivery.
    answered_at: float | None = None
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
    #: Each recorded span as (component, began, ended) on the wall clock, so that work inside a wait is not counted
    #: twice: see `_net_of_work`.
    spans: list[tuple[str, float, float]] = field(default_factory=list)

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


logger = logging.getLogger("Aura.TurnReceipt")
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
    """Record what one turn does, for the duration of the turn.

    A turn already being recorded is THE turn, and a recording opened inside it joins it, as her thinking joins the
    turn the chat route opened when the person's message arrived: LIVE 10 November 2026 the receipt began when her
    thinking did, so what the route read before it (her web check among it) and everything after it to delivery was
    outside the split, and the person's wait was longer than the turn the receipt timed. The joined scope's end is when
    the answer existed; the owner's end, the turn's, and the stretch between them is delivery.
    """

    existing = _CURRENT.get()
    if existing is not None and existing.finished_at is None:
        if not existing.phases_available:
            existing.phases_available = tuple(str(p) for p in phases_available)
        try:
            yield existing
        finally:
            existing.answered_at = time.time()
        return
    receipt = TurnReceipt(
        turn_id=str(turn_id),
        phases_available=tuple(str(p) for p in phases_available),
    )
    token = _CURRENT.set(receipt)
    try:
        yield receipt
    finally:
        receipt.finished_at = time.time()
        if receipt.answered_at is not None and "delivery" not in receipt.latency_s:
            receipt.latency_s["delivery"] = max(0.0, receipt.finished_at - receipt.answered_at)
            receipt.latency_samples["delivery"] = 1
            receipt.spans.append(("delivery", receipt.answered_at, receipt.finished_at))
        _net_of_work(receipt)
        try:
            _CURRENT.reset(token)
        except ValueError:
            _CURRENT.set(None)
        with _RECENT_LOCK:
            _RECENT.append(receipt)
            if len(_RECENT) > _RECENT_LIMIT:
                del _RECENT[: len(_RECENT) - _RECENT_LIMIT]
        _say_where_the_time_went(receipt)


#: Spans of work, each measured by what did it, against spans of waiting around work: queue and delivery.
_WORK = frozenset({"prefill", "decode", "tool", "retrieval"})
_AROUND_WORK = frozenset({"queue", "delivery"})


def _net_of_work(receipt: TurnReceipt) -> None:
    """Queue and delivery without the work done inside them, which is already counted under its own name.

    LIVE 11 November 2026, two typed turns: her gate's queue ran from its entry to her cortex's call and held a read
    of her local corpus and a one-token generation, and delivery held the generation that repaired her reply; each
    was also recorded as prefill and decode, so the second turn's parts added up past its 52 s.
    """
    work = sorted((began, ended) for name, began, ended in receipt.spans if name in _WORK and ended > began)
    merged: list[list[float]] = []
    for began, ended in work:
        if merged and began <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], ended)
        else:
            merged.append([began, ended])
    for name in _AROUND_WORK:
        around = [(began, ended) for kind, began, ended in receipt.spans if kind == name and ended > began]
        if not around or name not in receipt.latency_s:
            continue
        inside = sum(max(0.0, min(ended, w_end) - max(began, w_began))
                     for began, ended in around for w_began, w_end in merged)
        receipt.latency_s[name] = max(0.0, receipt.latency_s[name] - inside)


def _say_where_the_time_went(receipt: TurnReceipt) -> None:
    """One line per turn in her log: each component's seconds as its recorder measured them, and the rest.

    R11 asks for the five read off a live turn. They were recorded, and the health route's read model served them
    as an empty block, so on 10 November a 62-second turn typed into her chat could not be split from outside. A
    component nobody measured is said as unmeasured, not as zero.
    """
    total = (receipt.finished_at or receipt.started_at) - receipt.started_at
    if not receipt.latency_s and not receipt.phases_executed:
        # Opened and closed with no phase run and nothing measured: four at once on 10 November, each "took 0.00s".
        logger.debug("⏱️ turn %s closed in %.2fs without running a phase (%s)", receipt.turn_id[:12], total,
                     receipt.response_path)
        return
    measured = " ".join(f"{name}={receipt.latency_s[name]:.2f}s" for name in sorted(receipt.latency_s))
    unmeasured = ",".join(sorted(LATENCY_COMPONENTS - set(receipt.latency_s))) or "none"
    logger.info("⏱️ turn %s (%s) took %.2fs: %s; unattributed %.2fs; unmeasured: %s", receipt.turn_id[:12],
                receipt.response_path, total, measured or "nothing measured", receipt.unattributed_s or 0.0,
                unmeasured)


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


def record_latency(component: str, seconds: float, *, ended_at: float | None = None) -> None:
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
    # not a failure: a value that is not a number is not one this can read.
    except (TypeError, ValueError):
        return
    if not (span >= 0.0) or span != span:
        return
    receipt = _CURRENT.get()
    if receipt is None or receipt.finished_at is not None:
        return  # work that outlives its turn is not the turn's time
    receipt.latency_s[name] = receipt.latency_s.get(name, 0.0) + span
    receipt.latency_samples[name] = receipt.latency_samples.get(name, 0) + 1
    ended = time.time() if ended_at is None else float(ended_at)  # a span is recorded as it ends, unless told
    receipt.spans.append((name, ended - span, ended))


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


def latency_by_component(limit: int = 16) -> dict[str, Any]:
    """The last few turns' latency, split by the thing that measured it.

    R11 asks for prefill, decode, tool, retrieval and queue measured
    SEPARATELY. Each has a recorder and each writes into the turn's receipt;
    what nobody outside the process could do was read the split back — the
    health route publishes a curated payload and this was not in it, so a
    measurement that exists was unreadable from outside (2026-09-20).

    A component nobody measured on a turn is absent, not zero, so an unread
    lane cannot look instant. `unattributed_s` is the rest of the wall clock:
    what the turn spent outside every component that reported.
    """
    receipts = recent_receipts(limit=limit)
    per_component: dict[str, list[float]] = {}
    for receipt in receipts:
        for name, seconds in (receipt.get("latency_s") or {}).items():
            per_component.setdefault(str(name), []).append(float(seconds))
    summary: dict[str, Any] = {}
    for name, values in sorted(per_component.items()):
        ordered = sorted(values)
        middle = len(ordered) // 2
        median = (
            ordered[middle]
            if len(ordered) % 2
            else (ordered[middle - 1] + ordered[middle]) / 2.0
        )
        summary[name] = {
            "turns": len(ordered),
            "median_s": round(median, 4),
            "slowest_s": round(ordered[-1], 4),
            "total_s": round(sum(ordered), 4),
        }
    unattributed = [
        float(receipt["unattributed_s"])
        for receipt in receipts
        if receipt.get("unattributed_s") is not None
    ]
    return {
        "turns_read": len(receipts),
        "components": summary,
        "never_measured": sorted(LATENCY_COMPONENTS - set(summary)),
        "unattributed_s_total": round(sum(unattributed), 4) if unattributed else None,
    }


def reset_turn_receipts_for_test() -> None:
    _CURRENT.set(None)
    with _RECENT_LOCK:
        _RECENT.clear()
