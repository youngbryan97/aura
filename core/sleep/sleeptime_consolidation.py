"""Consolidation that runs while she is awake.

`SleepManager` is stop-the-world: it sets ``is_sleeping = True``, blocks active
actions, runs six stages, and restores energy. That is the right shape for
world-model training and nightly reporting, and the wrong shape for the thing
consolidation is mostly *for* — noticing, an hour into a conversation, that
what she believes about the person in front of her has drifted from what she
wrote down. By the time low energy triggers a sleep cycle, the conversation
that produced the correction is over.

This is the other half: a consolidator that runs *during* normal operation,
against the same memory blocks the live turn is editing. Concurrency is the
entire difficulty. Two writers on one block with last-write-wins loses whichever
finished first, silently, and the loser is usually the turn that had the user in
front of it — the user says "no, my sister's name is Tanya", the live turn
writes it, a background pass that read the block ten seconds earlier writes its
summary over the top, and the correction is gone with nothing logged.

So every write here is compare-and-swap, and a conflict is *re-derived against
the winner's value* rather than retried blindly or dropped. The background pass
is always the one that yields: it has no user waiting and can afford to redo
its work.

Three further positions:

* **Consolidation targets pressure, not age.** A block at 20% utilization does
  not need compacting however stale it looks; a block at 95% is about to start
  refusing writes. Pressure is the signal that predicts loss.
* **An overflowing summary is a failure, not a truncation.** If the rewrite
  does not fit, the block keeps what it had.
* **Background edits are attributed.** The block history distinguishes what a
  live turn learned from what a background pass inferred, because those two
  warrant different amounts of trust when they later disagree.
* **Some blocks are never summarised at all.** A block carrying
  ``derived_from`` is the display surface of a structured store, not prose.
  Summarising what she knows about a person is the operation that walks
  "seemed frustrated once, during a failing deploy" to "is easily frustrated" —
  compression drops adjectives before nouns, and in person-knowledge the
  qualifiers *are* the adjectives. No care in the summariser prevents that, so
  those blocks are skipped here and would refuse the write anyway.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from core.memory.memory_blocks import (
    BlockDerived,
    BlockImmutable,
    BlockOverflow,
    BlockVersionConflict,
    MemoryBlock,
    MemoryBlockSet,
    UnknownBlock,
)
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.SleepTime")

__all__ = [
    "ConsolidationOutcome",
    "ConsolidationResult",
    "SleepTimeConsolidator",
    "DEFAULT_PRESSURE_THRESHOLD",
]

#: Utilization above which a block is worth consolidating. Below this the
#: rewrite costs a model call to reclaim room nothing is asking for.
DEFAULT_PRESSURE_THRESHOLD = 0.75

#: How many times to re-derive against a concurrent winner before giving up.
#: Bounded because an unbounded retry against a busy block is a spin, and the
#: live turn's writes are the ones that must not be delayed.
DEFAULT_MAX_ATTEMPTS = 3


class ConsolidationOutcome(str):
    """Why a consolidation ended the way it did."""


REWRITTEN = "rewritten"
UNCHANGED = "unchanged"
SKIPPED_BELOW_PRESSURE = "skipped_below_pressure"
SKIPPED_IMMUTABLE = "skipped_immutable"
SKIPPED_DERIVED = "skipped_derived"
FAILED_CONFLICT = "failed_conflict"
FAILED_OVERFLOW = "failed_overflow"
FAILED_SUMMARIZER = "failed_summarizer"


@dataclass(frozen=True)
class ConsolidationResult:
    """What happened to one block, and why.

    ``attempts`` is kept because repeated conflicts are the observable signal
    that a block is contended — which is a tuning fact about the schedule, not
    a failure of the consolidator.
    """

    label: str
    outcome: str
    attempts: int = 0
    before_utilization: float = 0.0
    after_utilization: float = 0.0
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.outcome in {REWRITTEN, UNCHANGED, SKIPPED_BELOW_PRESSURE,
                                SKIPPED_IMMUTABLE, SKIPPED_DERIVED}

    @property
    def reclaimed(self) -> float:
        return max(0.0, self.before_utilization - self.after_utilization)


class SleepTimeConsolidator:
    """Rewrites memory blocks in the background without clobbering live writes.

    ``summarize`` receives the block's *current* value — re-read on every
    attempt, never cached — and returns its replacement. Injected rather than
    reached for so this is testable without a cortex, and so the caller decides
    which model pays: consolidation is exactly the work that belongs on a small
    fast model rather than the resident 32B.
    """

    def __init__(
        self,
        blocks: MemoryBlockSet,
        *,
        summarize: Callable[[MemoryBlock], str],
        author: str = "sleeptime",
        pressure_threshold: float = DEFAULT_PRESSURE_THRESHOLD,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        if not 0.0 <= pressure_threshold <= 1.0:
            raise ValueError("pressure_threshold is a utilization fraction, 0..1")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.blocks = blocks
        self._summarize = summarize
        self.author = author
        self.pressure_threshold = pressure_threshold
        self.max_attempts = max_attempts

    # -- selection ---------------------------------------------------------

    def under_pressure(self) -> list[str]:
        """Blocks worth consolidating, fullest first.

        Age is deliberately not a criterion. A stale block at 20% is costing
        nothing; a fresh one at 95% is about to start refusing writes.
        """
        candidates = [
            (block.utilization, block.label)
            for block in self.blocks
            if not block.immutable
            and not block.derived_from
            and block.utilization >= self.pressure_threshold
        ]
        return [label for _, label in sorted(candidates, reverse=True)]

    # -- the write loop ----------------------------------------------------

    def consolidate(self, label: str) -> ConsolidationResult:
        """Rewrite one block, yielding to any concurrent live write."""
        try:
            block = self.blocks.get(label)
        except UnknownBlock as exc:
            return ConsolidationResult(
                label=label, outcome=FAILED_SUMMARIZER, detail=str(exc)
            )

        if block.immutable:
            return ConsolidationResult(
                label=label, outcome=SKIPPED_IMMUTABLE,
                before_utilization=block.utilization,
                after_utilization=block.utilization,
            )

        if block.derived_from:
            # Never summarised, at any pressure. This block is the display
            # surface of a structured store, and summarising it is precisely
            # the operation that walks "seemed frustrated once, during a
            # failing deploy" to "is easily frustrated" — no step unreasonable,
            # nobody deciding. Returned before the summarizer is consulted so
            # the model is never even asked; the block would refuse the write
            # regardless, but paying for a call to be refused is a
            # standing cost for nothing.
            return ConsolidationResult(
                label=label, outcome=SKIPPED_DERIVED,
                before_utilization=block.utilization,
                after_utilization=block.utilization,
                detail=f"rendered from {block.derived_from}",
            )

        before = block.utilization
        if before < self.pressure_threshold:
            return ConsolidationResult(
                label=label, outcome=SKIPPED_BELOW_PRESSURE, attempts=0,
                before_utilization=before, after_utilization=before,
            )

        for attempt in range(1, self.max_attempts + 1):
            # Re-read every attempt. Summarizing a value the live turn has
            # already replaced is how a background pass reintroduces a fact the
            # user just corrected.
            block = self.blocks.get(label)
            observed_version = block.version

            try:
                proposed = self._summarize(block)
            except Exception as exc:  # noqa: BLE001 - degrade, never clobber
                record_degradation(
                    "sleep.sleeptime_consolidation",
                    exc,
                    action=f"left block {label!r} as it was",
                )
                return ConsolidationResult(
                    label=label, outcome=FAILED_SUMMARIZER, attempts=attempt,
                    before_utilization=before, after_utilization=block.utilization,
                    detail=str(exc)[:200],
                )

            if not proposed or not proposed.strip():
                return ConsolidationResult(
                    label=label, outcome=FAILED_SUMMARIZER, attempts=attempt,
                    before_utilization=before, after_utilization=block.utilization,
                    detail="summarizer returned empty text",
                )

            proposed = proposed.strip()
            if proposed == block.value:
                return ConsolidationResult(
                    label=label, outcome=UNCHANGED, attempts=attempt,
                    before_utilization=before, after_utilization=block.utilization,
                )

            try:
                updated = self.blocks.rewrite(
                    label, proposed, author=self.author,
                    reason="sleep-time consolidation",
                    expected_version=observed_version,
                )
            except BlockVersionConflict:
                # The live turn won. Redo the work against what it wrote —
                # never overwrite it. The background pass has no user waiting.
                logger.debug(
                    "block %r changed under consolidation; re-deriving (attempt %d)",
                    label, attempt,
                )
                continue
            except BlockOverflow as exc:
                # Keeping the old value is the conservative loss: it is stale,
                # not absent.
                return ConsolidationResult(
                    label=label, outcome=FAILED_OVERFLOW, attempts=attempt,
                    before_utilization=before, after_utilization=block.utilization,
                    detail=str(exc)[:200],
                )
            except BlockImmutable as exc:  # pragma: no cover - guarded above
                return ConsolidationResult(
                    label=label, outcome=SKIPPED_IMMUTABLE, attempts=attempt,
                    before_utilization=before, after_utilization=block.utilization,
                    detail=str(exc)[:200],
                )

            return ConsolidationResult(
                label=label, outcome=REWRITTEN, attempts=attempt,
                before_utilization=before, after_utilization=updated.utilization,
            )

        # Every attempt lost the race. Not an error — the block is contended,
        # which is information about the schedule.
        logger.info(
            "block %r stayed contended through %d attempts; leaving it to the "
            "live writer", label, self.max_attempts,
        )
        return ConsolidationResult(
            label=label, outcome=FAILED_CONFLICT, attempts=self.max_attempts,
            before_utilization=before,
            after_utilization=self.blocks.get(label).utilization,
            detail="block was rewritten by a live turn on every attempt",
        )

    def run_once(self, labels: Iterable[str] | None = None) -> list[ConsolidationResult]:
        """One pass over the blocks that need it.

        Safe to call on a timer during normal operation: every write is
        compare-and-swap, so the worst case is wasted model calls, never a lost
        live edit.
        """
        targets: Sequence[str] = (
            list(labels) if labels is not None else self.under_pressure()
        )
        return [self.consolidate(label) for label in targets]

    def scheduled_task(self, *, interval_s: float = 900.0, name: str = "sleeptime_consolidation"):
        """A ``TaskSpec`` the Scheduler can register, for periodic passes.

        Not registered anywhere by default, and deliberately so: this makes a
        summarizer call per pressured block on a timer, which is a standing
        cost against the live host rather than a free background nicety. Whoever
        turns it on should be choosing that.

        The work runs in a thread — ``run_once`` is synchronous and the
        summarizer will usually be a model call, which is precisely the kind of
        thing that must not sit on the event loop.
        """
        import asyncio

        from core.scheduler import TaskSpec

        async def _tick() -> None:
            await asyncio.to_thread(self.run_once)

        return TaskSpec(
            name=name,
            coro=_tick,
            tick_interval=interval_s,
            critical=False,
            metadata={"owner": "sleep.sleeptime_consolidation"},
        )
