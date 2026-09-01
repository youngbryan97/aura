"""core/cognition/action_receipt.py — proof the world changed, not proof the call returned.

An action succeeded is three different claims and they are routinely confused:

1. the call did not raise;
2. the actuator reports it did something;
3. the world is now different in the way the action intended.

Only the third is evidence, and a learner fed the first two learns from
fiction. Aura has been here: thirty-five moves typed into a terminal while the
game was one window back, six guards passing on empty readings; a skill scored
successful because its error field was empty. Screen pursuit has since grown
real before/after checking, and that fix lives in one path while every other
environment action still reports success the first way.

An :class:`ActionTransitionReceipt` is the third claim, made checkable. It
carries the state before, the state after, the identity of the thing acted on,
who authorised it, and — the part that does the work — a **verdict** computed
from those, not supplied by the caller:

* ``CONFIRMED`` — the observed change matches what the action predicted.
* ``NO_CHANGE`` — the action ran and nothing moved. Soar would call this an
  impasse, and :mod:`core.cognition.impasse` is where it goes.
* ``WRONG_TARGET`` — something changed, but not the thing addressed. The
  terminal-window failure, made visible.
* ``UNVERIFIED`` — nothing was observed after. Not success, not failure:
  the action may well have worked and nobody looked. A learner must treat
  this as absent evidence rather than as a weak positive.
* ``CONTRADICTED`` — the observed change is incompatible with the prediction.

:func:`qualified` is the gate. A learner that calls it before recording an
outcome cannot learn from an unverified action, and card 195's bar — every
live environment learner rejects transitions without a qualified receipt — is
that call being present rather than a promise in a docstring.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "SCHEMA_VERSION",
    "TransitionVerdict",
    "ActionTransitionReceipt",
    "verify_transition",
    "qualified",
    "ReceiptLedger",
    "get_receipt_ledger",
    "reset_receipt_ledger_for_test",
]


#: Schema version for the receipt as a cross-organ contract. A learner that
#: gates on this shape has to know when the shape moved.
SCHEMA_VERSION = "aura.cognition.action_receipt.v1"


class TransitionVerdict(StrEnum):
    CONFIRMED = "confirmed"
    NO_CHANGE = "no_change"
    WRONG_TARGET = "wrong_target"
    UNVERIFIED = "unverified"
    CONTRADICTED = "contradicted"


#: Verdicts a learner may treat as evidence about the world.
_QUALIFIED = frozenset({TransitionVerdict.CONFIRMED, TransitionVerdict.CONTRADICTED})


@dataclass(frozen=True, slots=True)
class ActionTransitionReceipt:
    """One action, and what the world did about it."""

    action: str
    target: str
    authority: str
    before: Mapping[str, Any] | None
    after: Mapping[str, Any] | None
    predicted: Mapping[str, Any] = field(default_factory=dict)
    verdict: TransitionVerdict = TransitionVerdict.UNVERIFIED
    observed_change: tuple[str, ...] = ()
    unexpected_change: tuple[str, ...] = ()
    #: Whether the after-state was read more than once and agreed with itself.
    #: A single read of a settling interface is a guess about the future.
    stable: bool = False
    at: float = field(default_factory=time.time)
    note: str = ""

    @property
    def is_qualified(self) -> bool:
        """Whether a learner may use this as evidence about the world."""
        return self.verdict in _QUALIFIED and self.stable

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "target": self.target,
            "authority": self.authority,
            "verdict": self.verdict.value,
            "observed_change": list(self.observed_change),
            "unexpected_change": list(self.unexpected_change),
            "stable": self.stable,
            "qualified": self.is_qualified,
            "at": self.at,
            "note": self.note,
        }


def _changed_keys(before: Mapping[str, Any], after: Mapping[str, Any]) -> tuple[str, ...]:
    keys = set(before) | set(after)
    return tuple(sorted(k for k in keys if before.get(k) != after.get(k)))


def verify_transition(
    *,
    action: str,
    target: str,
    authority: str,
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any] | None,
    predicted: Mapping[str, Any] | None = None,
    stable: bool = False,
    target_key: str = "",
) -> ActionTransitionReceipt:
    """Compute the verdict from the two observations. The caller does not get a vote.

    ``predicted`` is what the action said it would make true. ``target_key``
    names the field that belongs to the thing being acted on, which is what
    separates "the screen changed" from "the thing I clicked changed".
    """
    predicted = dict(predicted or {})
    if before is None or after is None:
        return ActionTransitionReceipt(
            action=action, target=target, authority=authority,
            before=before, after=after, predicted=predicted,
            verdict=TransitionVerdict.UNVERIFIED, stable=False,
            note="no observation on one side of the action",
        )

    changed = _changed_keys(before, after)
    if not changed:
        return ActionTransitionReceipt(
            action=action, target=target, authority=authority,
            before=before, after=after, predicted=predicted,
            verdict=TransitionVerdict.NO_CHANGE, stable=stable,
            note="the action ran and nothing moved",
        )

    if not predicted:
        # Something happened and nothing said what should. That is an
        # observation of change, not evidence that this action caused it.
        return ActionTransitionReceipt(
            action=action, target=target, authority=authority,
            before=before, after=after, predicted=predicted,
            verdict=TransitionVerdict.UNVERIFIED, observed_change=changed, stable=stable,
            note="the action predicted nothing, so nothing can be confirmed",
        )

    matched = tuple(sorted(k for k, v in predicted.items() if after.get(k) == v))
    missed = tuple(sorted(k for k, v in predicted.items() if after.get(k) != v))
    unexpected = tuple(k for k in changed if k not in predicted)

    if target_key and target_key not in changed and missed:
        return ActionTransitionReceipt(
            action=action, target=target, authority=authority,
            before=before, after=after, predicted=predicted,
            verdict=TransitionVerdict.WRONG_TARGET,
            observed_change=changed, unexpected_change=unexpected, stable=stable,
            note=f"{target_key!r} did not change; {list(unexpected)} did",
        )

    if missed and not matched:
        return ActionTransitionReceipt(
            action=action, target=target, authority=authority,
            before=before, after=after, predicted=predicted,
            verdict=TransitionVerdict.CONTRADICTED,
            observed_change=changed, unexpected_change=unexpected, stable=stable,
            note=f"predicted {list(predicted)}, none of it happened",
        )
    if missed:
        return ActionTransitionReceipt(
            action=action, target=target, authority=authority,
            before=before, after=after, predicted=predicted,
            verdict=TransitionVerdict.CONTRADICTED,
            observed_change=changed, unexpected_change=unexpected, stable=stable,
            note=f"{list(matched)} happened, {list(missed)} did not",
        )
    return ActionTransitionReceipt(
        action=action, target=target, authority=authority,
        before=before, after=after, predicted=predicted,
        verdict=TransitionVerdict.CONFIRMED,
        observed_change=changed, unexpected_change=unexpected, stable=stable,
    )


class UnqualifiedTransition(RuntimeError):
    """A learner was handed an action whose effect was never established."""


def qualified(receipt: ActionTransitionReceipt, *, learner: str) -> ActionTransitionReceipt:
    """Gate. Raise unless this receipt is evidence a learner may use.

    Call this at the top of any update that learns from an environment action.
    The exception is the point: an unverified action must not silently become
    a training example.
    """
    ledger = get_receipt_ledger()
    ledger.record(receipt, learner=learner)
    if not receipt.is_qualified:
        raise UnqualifiedTransition(
            f"{learner!r} cannot learn from {receipt.action!r} on {receipt.target!r}: "
            f"verdict {receipt.verdict.value}"
            + ("" if receipt.stable else ", after-state never settled")
        )
    return receipt


class ReceiptLedger:
    """What the environment learners have been fed, and what was refused."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_verdict: dict[str, int] = {}
        self._by_learner: dict[str, dict[str, int]] = {}

    def record(self, receipt: ActionTransitionReceipt, *, learner: str = "") -> None:
        with self._lock:
            key = receipt.verdict.value
            self._by_verdict[key] = self._by_verdict.get(key, 0) + 1
            if learner:
                bucket = self._by_learner.setdefault(learner, {})
                bucket[key] = bucket.get(key, 0) + 1
                if receipt.is_qualified:
                    bucket["qualified"] = bucket.get("qualified", 0) + 1

    def report(self) -> dict[str, Any]:
        with self._lock:
            total = sum(self._by_verdict.values())
            confirmed = self._by_verdict.get(TransitionVerdict.CONFIRMED.value, 0)
            return {
                "transitions": total,
                "by_verdict": dict(sorted(self._by_verdict.items())),
                "by_learner": {k: dict(sorted(v.items())) for k, v in self._by_learner.items()},
                "confirmed_fraction": (confirmed / total) if total else None,
            }


_ledger_lock = threading.Lock()
_ledger: ReceiptLedger | None = None


def get_receipt_ledger() -> ReceiptLedger:
    global _ledger
    with _ledger_lock:
        if _ledger is None:
            _ledger = ReceiptLedger()
        return _ledger


def reset_receipt_ledger_for_test() -> ReceiptLedger:
    global _ledger
    with _ledger_lock:
        _ledger = ReceiptLedger()
        return _ledger


def observe_twice(read: Callable[[], Mapping[str, Any]], *, settle: float = 0.0) -> tuple[Mapping[str, Any], bool]:
    """Read the world twice and say whether it agreed with itself.

    A single read of a settling interface is a guess about the future. This
    returns the second reading and whether the two matched, which is what
    ``stable`` on the receipt means.
    """
    first = dict(read())
    if settle > 0:
        time.sleep(settle)
    second = dict(read())
    return second, first == second
