"""A batch of actions decided together, and the ledger of what happened to them.

OpenHands separates the batch from its execution: an immutable list of actions
with ids, a mutable ledger keyed by those ids, and no executor allowed to touch
the plan. Results come back joined by id and emitted in the order the actions
were decided, whatever order they finished in.

The three things that separation buys, and each of them is a defect you get
without it:

* **the plan cannot change under the executor.** A batch that is mutated while
  running produces a receipt describing something nobody decided, and the
  reasoning that produced it can never be reconstructed.
* **blocked is not failed.** An action refused by governance did not run; an
  action that ran and raised did. Collapsing them loses the only difference
  that decides whether asking again is sensible.
* **order survives concurrency.** Parallel actions finish in whatever order the
  machine gives, and a reply assembled in completion order tells a different
  story every run from identical inputs.

Concurrency is bounded because parallel tools share one filesystem, one
working directory and one conversation. Two actions that both write the same
path are a race however carefully each was decided, so the limit is a property
of the batch rather than a tuning knob.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger("Aura.WhatSheDecidedToDoAtOnce")

__all__ = [
    "AnAction",
    "ABatch",
    "HowItWent",
    "AnOutcome",
    "TheLedger",
    "run_the_batch",
    "how_the_batches_have_gone",
    "forget_everything",
]


class HowItWent(StrEnum):
    """What happened to one action. Blocked and failed are not the same."""

    #: Decided, not yet started.
    WAITING = "waiting"
    RUNNING = "running"
    DID_IT = "did it"
    #: It ran and raised. Asking again may work.
    FAILED = "failed"
    #: It never ran. Governance, a missing capability, a refusal. Asking again
    #: gets the same answer unless something else changes first.
    BLOCKED = "blocked"
    #: Everything after a finishing action, which is not a failure of theirs.
    NOT_REACHED = "not reached"


@dataclass(frozen=True, slots=True)
class AnAction:
    """One thing to do. Immutable, and carries the id everything joins on."""

    name: str
    #: What running it needs. Never mutated by an executor.
    takes: dict[str, Any] = field(default_factory=dict)
    #: Why this was decided, so a receipt can be read back to a reason.
    because: str = ""
    #: True where this ends the batch: nothing after it is reached.
    finishes: bool = False
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "takes": dict(self.takes),
            "because": self.because,
            "finishes": self.finishes,
        }


@dataclass(frozen=True, slots=True)
class ABatch:
    """What she decided to do, in the order she decided it. Never modified."""

    actions: tuple[AnAction, ...]
    #: The most that may run together. One means in order, one at a time.
    at_once: int = 4
    decided_at: float = field(default_factory=time.time)
    because: str = ""

    def __post_init__(self) -> None:
        if self.at_once < 1:
            raise ValueError("a batch that may run nothing is not a batch")
        seen = [one.id for one in self.actions]
        if len(seen) != len(set(seen)):
            raise ValueError(
                "two actions share an id; results are joined by id and would "
                "be attributed to whichever arrived second"
            )

    def up_to_the_finish(self) -> tuple[AnAction, ...]:
        """The actions that can be reached, stopping after the first finish."""
        out: list[AnAction] = []
        for one in self.actions:
            out.append(one)
            if one.finishes:
                break
        return tuple(out)

    def to_dict(self) -> dict[str, Any]:
        return {
            "actions": [one.to_dict() for one in self.actions],
            "at_once": self.at_once,
            "decided_at": self.decided_at,
            "because": self.because,
        }


@dataclass
class AnOutcome:
    """What happened to one action. Mutable, and never part of the batch."""

    action_id: str
    went: HowItWent = HowItWent.WAITING
    value: Any = None
    said: str = ""
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "went": str(self.went),
            "said": self.said,
            "seconds": round(self.seconds, 4),
        }


class TheLedger:
    """What happened, keyed by action id. Separate from the plan on purpose."""

    def __init__(self, batch: ABatch) -> None:
        self._batch = batch
        self._by_id: dict[str, AnOutcome] = {
            one.id: AnOutcome(action_id=one.id) for one in batch.actions
        }
        self._lock = threading.Lock()

    def note(
        self,
        action_id: str,
        went: HowItWent,
        *,
        value: Any = None,
        said: str = "",
        seconds: float = 0.0,
    ) -> AnOutcome:
        with self._lock:
            one = self._by_id.get(action_id)
            if one is None:
                raise KeyError(f"{action_id} is not in this batch")
            one.went = went
            one.value = value
            one.said = said
            one.seconds = seconds
            return one

    def outcome(self, action_id: str) -> AnOutcome:
        with self._lock:
            return self._by_id[action_id]

    def in_the_order_decided(self) -> tuple[AnOutcome, ...]:
        """Every outcome, in the order the actions were decided.

        Not the order they finished. A reply assembled in completion order
        tells a different story every run from identical inputs.
        """
        with self._lock:
            return tuple(self._by_id[one.id] for one in self._batch.actions)

    def what_ran(self) -> tuple[AnOutcome, ...]:
        return tuple(
            one
            for one in self.in_the_order_decided()
            if one.went in (HowItWent.DID_IT, HowItWent.FAILED)
        )

    def what_was_blocked(self) -> tuple[AnOutcome, ...]:
        return tuple(
            one for one in self.in_the_order_decided() if one.went is HowItWent.BLOCKED
        )

    def report(self) -> dict[str, Any]:
        rows = self.in_the_order_decided()
        by_name = {one.id: one.name for one in self._batch.actions}
        counted: dict[str, int] = {}
        for one in rows:
            counted[str(one.went)] = counted.get(str(one.went), 0) + 1
        return {
            "actions": len(rows),
            "at_once": self._batch.at_once,
            "counted": counted,
            "in_order": [
                {"name": by_name[one.action_id], **one.to_dict()} for one in rows
            ],
        }


_HISTORY: list[dict[str, Any]] = []
_LOCK = threading.Lock()
_KEEP = 100


async def run_the_batch(
    batch: ABatch,
    do: Callable[[AnAction], Awaitable[Any]],
    *,
    may_it_run: Callable[[AnAction], str] | None = None,
) -> TheLedger:
    """Run a batch and return the ledger. The batch itself is never touched.

    ``may_it_run`` returns an empty string to allow, or a sentence saying why
    not — which records BLOCKED rather than FAILED, because an action that
    never ran did not fail.
    """
    ledger = TheLedger(batch)
    reachable = {one.id for one in batch.up_to_the_finish()}
    for one in batch.actions:
        if one.id not in reachable:
            ledger.note(
                one.id,
                HowItWent.NOT_REACHED,
                said="a finishing action came first",
            )

    room = asyncio.Semaphore(batch.at_once)

    async def one_of_them(action: AnAction) -> None:
        refused = may_it_run(action) if may_it_run else ""
        if refused:
            ledger.note(action.id, HowItWent.BLOCKED, said=refused)
            return
        async with room:
            ledger.note(action.id, HowItWent.RUNNING)
            started = time.monotonic()
            try:
                value = await do(action)
            except asyncio.CancelledError:
                ledger.note(
                    action.id,
                    HowItWent.BLOCKED,
                    said="the batch was cancelled",
                    seconds=time.monotonic() - started,
                )
                raise
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                ledger.note(
                    action.id,
                    HowItWent.FAILED,
                    said=f"{type(exc).__name__}: {exc}",
                    seconds=time.monotonic() - started,
                )
                return
            ledger.note(
                action.id,
                HowItWent.DID_IT,
                value=value,
                seconds=time.monotonic() - started,
            )

    try:
        await asyncio.gather(
            *(one_of_them(one) for one in batch.up_to_the_finish())
        )
    finally:
        with _LOCK:
            _HISTORY.append(ledger.report())
            del _HISTORY[:-_KEEP]
    return ledger


def how_the_batches_have_gone() -> dict[str, Any]:
    """For the health report: what ran, what was blocked, what never got there."""
    with _LOCK:
        rows = list(_HISTORY)
    counted: dict[str, int] = {}
    for row in rows:
        for went, many in row["counted"].items():
            counted[went] = counted.get(went, 0) + many
    return {
        "batches": len(rows),
        "actions": sum(row["actions"] for row in rows),
        "counted": dict(sorted(counted.items())),
        "widest": max((row["at_once"] for row in rows), default=0),
        "recent": rows[-3:],
    }


def forget_everything() -> None:
    with _LOCK:
        _HISTORY.clear()


def a_batch_of(
    actions: Iterable[AnAction], *, at_once: int = 4, because: str = ""
) -> ABatch:
    """A batch from actions, with the ids already assigned."""
    return ABatch(actions=tuple(actions), at_once=at_once, because=because)
