"""Which failures are worth another attempt, and which are a decision.

CrewAI keeps a central retry classifier that marks the classes nothing may
retry — cancel, denied, invariant, invalid input, unsafe — and every retry
loop consults it. The closure asked for the same, and the reason is specific:
a generic retry treats a refusal as a transient fault and asks again, so a
deliberate stop becomes a loop, and the loop looks from outside like the
system trying hard.

Aura has several retry loops and each decides for itself. That is how a
governance refusal gets asked three times, and how a cancelled turn gets
restarted by the thing that was cancelling it.

Three answers, and the middle one is why a boolean will not do:

* **again** — a fault that may not recur. A timeout, a busy lane.
* **not like this** — the attempt was wrong, and repeating it will be wrong
  the same way. Invalid input. Changing the input is a different attempt and
  a caller may make one.
* **never** — somebody said no, or it would be unsafe. Asking again is asking
  the same question of the same answer.

Classified by exception type and by the words a refusal used, in that order. A
type is a promise; the words are a fallback for the refusals that travel as
plain strings, and that fallback is why the words matter.
"""
from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any

logger = logging.getLogger("Aura.WhatMustNeverBeRetried")

__all__ = [
    "TryAgain",
    "how_to_treat",
    "may_be_retried",
    "why_not",
]


class TryAgain(StrEnum):
    """What to do about a failure."""

    AGAIN = "again"
    NOT_LIKE_THIS = "not like this"
    NEVER = "never"


#: Exception names that are always a decision, never a fault. Names rather
#: than classes: importing every one of these to classify a failure would make
#: the classifier depend on the whole tree, and a classifier that cannot be
#: imported early is one the early paths do not use.
_NEVER: frozenset[str] = frozenset(
    {
        "GovernanceViolationError",
        "PermissionError",
        "NotTheOwner",
        "OnTheWrongThread",
        "StateOwnershipViolation",
        "OwnershipViolation",
        "LineageBroken",
        "AnIllegalWrite",
        "CancelledError",
        "Stopped",
        "GaveUp",
        "KeyboardInterrupt",
        "SystemExit",
    }
)

#: Exceptions where the attempt itself was malformed. Another identical
#: attempt fails identically; a different one may not.
_NOT_LIKE_THIS: frozenset[str] = frozenset(
    {
        "ValueError",
        "TypeError",
        "KeyError",
        "AttributeError",
        "ValidationError",
        "AnIllegalWrite",
        "JSONDecodeError",
        "NotImplementedError",
    }
)

#: What a refusal says when it travels as words instead of a type. Checked
#: after the type, because a type is a promise and a phrase is a guess.
_WORDS_THAT_MEAN_NO: tuple[str, ...] = (
    "refused",
    "denied",
    "not permitted",
    "not allowed",
    "unsafe",
    "cancelled",
    "canceled",
    "governance",
    "invariant",
    "forbidden",
)


def how_to_treat(failure: Any) -> TryAgain:
    """What to do about this failure. Never guesses in the retrying direction.

    An unrecognised failure is AGAIN, which is the honest default: most
    faults are transient and a classifier that refused everything it did not
    recognise would stop the system doing ordinary work. The safety comes from
    the two lists being about refusals rather than about faults.
    """
    name = type(failure).__name__ if isinstance(failure, BaseException) else ""
    said = str(failure).lower()

    if name in _NEVER:
        return TryAgain.NEVER
    for kind in type(failure).__mro__ if isinstance(failure, BaseException) else ():
        if kind.__name__ in _NEVER:
            return TryAgain.NEVER
    if any(word in said for word in _WORDS_THAT_MEAN_NO):
        return TryAgain.NEVER
    if name in _NOT_LIKE_THIS:
        return TryAgain.NOT_LIKE_THIS
    return TryAgain.AGAIN


def may_be_retried(failure: Any) -> bool:
    """Whether an identical attempt is worth making.

    False for both NEVER and NOT_LIKE_THIS: repeating an attempt that was
    malformed is repeating the malformation.
    """
    return how_to_treat(failure) is TryAgain.AGAIN


def why_not(failure: Any) -> str:
    """A sentence a caller can act on, or empty where retrying is fine."""
    verdict = how_to_treat(failure)
    if verdict is TryAgain.AGAIN:
        return ""
    if verdict is TryAgain.NEVER:
        return (
            f"{type(failure).__name__} is a decision rather than a fault; "
            "asking again asks the same question of the same answer"
        )
    return (
        f"{type(failure).__name__} means the attempt was malformed; "
        "an identical one fails identically, a different one may not"
    )
