"""core/cognition/transaction.py — a learning update that cannot land half-done.

One experience updates several stores. The episode goes to memory, the rule to
the procedure store, the truth value to the AtomSpace, the attention value to
the graph, the outcome to the ledger. Nothing coordinates them. A crash, a
refusal or an exception between the second and the third leaves Aura believing
a rule whose supporting episode was never written, or holding an episode that
no rule points at — and neither is detectable afterwards, because the stores
agree with themselves.

A :class:`CognitiveTransaction` makes the group atomic in the only sense
available to a process with heterogeneous stores: every participant declares
how to prepare, how to commit and how to undo, and the transaction refuses to
commit any of them until all of them have prepared. If a commit fails
part-way, the ones that committed are undone in reverse order.

Two refusals, both deliberate:

* **A participant with no rollback cannot join.** "I cannot undo this" is a
  fine property for a store to have, and it means the store must be the last
  thing in the transaction or outside it. Silently accepting it would make the
  transaction a comment.
* **A prepare that fails aborts before anything commits.** Prepare is where a
  store says "I can do this" — it validates, reserves and returns. Work that
  can fail belongs in prepare, and a commit that raises is a defect the
  transaction reports rather than absorbs.

The compensation log is the honest part. When rollback itself fails the
transaction does not pretend to have unwound; it records which participants
are stranded and raises :class:`InconsistentRollback`, because a store that
could not be undone is exactly the thing an operator has to know about.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from core.runtime.lockdep import checked_lock

__all__ = [
    "TransactionState",
    "Participant",
    "CognitiveTransaction",
    "TransactionAborted",
    "InconsistentRollback",
    "transaction",
]


class TransactionState(StrEnum):
    OPEN = "open"
    PREPARED = "prepared"
    COMMITTED = "committed"
    ABORTED = "aborted"
    INCONSISTENT = "inconsistent"


class TransactionAborted(RuntimeError):
    """Nothing was committed. Every store is where it was."""


class InconsistentRollback(RuntimeError):
    """A rollback failed. Named stores are stranded and need an operator."""

    def __init__(self, message: str, stranded: Sequence[str]) -> None:
        super().__init__(message)
        self.stranded = tuple(stranded)


@dataclass
class Participant:
    """One store's part in a multi-store update."""

    name: str
    prepare: Callable[[], Any]
    commit: Callable[[Any], Any]
    rollback: Callable[[Any], Any]
    prepared_value: Any = None
    committed: bool = False


@dataclass
class CognitiveTransaction:
    """Prepare everything, then commit everything, or leave nothing changed."""

    evidence_id: str
    state: TransactionState = TransactionState.OPEN
    participants: list[Participant] = field(default_factory=list)
    opened_at: float = field(default_factory=time.time)
    log: list[str] = field(default_factory=list)

    def join(
        self,
        name: str,
        prepare: Callable[[], Any],
        commit: Callable[[Any], Any],
        rollback: Callable[[Any], Any] | None = None,
    ) -> None:
        """Add a store. A store with no rollback is refused."""
        if self.state is not TransactionState.OPEN:
            raise RuntimeError(f"transaction is {self.state.value}; nothing more can join")
        if rollback is None:
            raise ValueError(
                f"{name!r} joined with no rollback; a participant that cannot be undone "
                "makes the transaction a comment. Commit it outside, last, and say so."
            )
        self.participants.append(Participant(name, prepare, commit, rollback))

    def run(self) -> dict[str, Any]:
        """Prepare all, commit all, or undo what committed."""
        for participant in self.participants:
            try:
                participant.prepared_value = participant.prepare()
            except Exception as exc:  # noqa: BLE001 - the abort path is the point
                self.state = TransactionState.ABORTED
                self.log.append(f"{participant.name}: prepare failed: {type(exc).__name__}: {exc}")
                raise TransactionAborted(
                    f"{participant.name} could not prepare ({exc}); nothing was committed"
                ) from exc
        self.state = TransactionState.PREPARED

        for participant in self.participants:
            try:
                participant.commit(participant.prepared_value)
                participant.committed = True
                self.log.append(f"{participant.name}: committed")
            except Exception as exc:  # noqa: BLE001
                self.log.append(f"{participant.name}: commit failed: {type(exc).__name__}: {exc}")
                self._unwind()
                raise TransactionAborted(
                    f"{participant.name} failed to commit ({exc}); earlier commits were undone"
                ) from exc

        self.state = TransactionState.COMMITTED
        return {
            "evidence_id": self.evidence_id,
            "state": self.state.value,
            "participants": [p.name for p in self.participants],
            "log": list(self.log),
        }

    def _unwind(self) -> None:
        stranded: list[str] = []
        for participant in reversed(self.participants):
            if not participant.committed:
                continue
            try:
                participant.rollback(participant.prepared_value)
                self.log.append(f"{participant.name}: rolled back")
            except Exception as exc:  # noqa: BLE001
                stranded.append(participant.name)
                self.log.append(
                    f"{participant.name}: ROLLBACK FAILED: {type(exc).__name__}: {exc}"
                )
        if stranded:
            self.state = TransactionState.INCONSISTENT
            raise InconsistentRollback(
                "rollback failed for " + ", ".join(stranded) + "; these stores are stranded",
                stranded,
            )
        self.state = TransactionState.ABORTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "state": self.state.value,
            "participants": [
                {"name": p.name, "committed": p.committed} for p in self.participants
            ],
            "log": list(self.log),
            "opened_at": self.opened_at,
        }


class _TransactionScope:
    def __init__(self, evidence_id: str) -> None:
        self.txn = CognitiveTransaction(evidence_id=evidence_id)
        self.result: dict[str, Any] | None = None

    def __enter__(self) -> CognitiveTransaction:
        return self.txn

    def __exit__(self, exc_type, exc, _tb) -> bool:
        if exc_type is not None:
            self.txn.state = TransactionState.ABORTED
            self.txn.log.append(f"body raised {exc_type.__name__}; nothing committed")
            return False
        self.result = self.txn.run()
        return False


def transaction(evidence_id: str) -> _TransactionScope:
    """Open a transaction. Joining inside the block; commit on clean exit."""
    return _TransactionScope(evidence_id)


_ledger_lock = checked_lock("core.cognition.transaction.singleton")
_counts: dict[str, int] = {}


def record_outcome(state: TransactionState) -> None:
    with _ledger_lock:
        _counts[state.value] = _counts.get(state.value, 0) + 1


def transaction_report() -> dict[str, int]:
    with _ledger_lock:
        return dict(sorted(_counts.items()))
