"""What computed the last answer, kept so the next turn can say.

LIVE 2026-08-19: "Can you reverse a string for me? If so, reverse 'stressed'
and tell me exactly how you did it — model or code." She returned "desserts",
which is right, and then said she had "a model capability for string
manipulation" and had "requested the reverse operation". A regex matched and a
Python slice ran. The answer travelled and its provenance did not, so the only
account of the method available to the reply was an invented one.

The question also arrives a turn later — "was that you or a calculator?",
"how did you get that?" — by which time the computation is over. A receipt
outlives the turn that produced it, which is the whole reason to write one.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from core.conversation.word_markers import names_any
from core.runtime.lockdep import checked_lock

__all__ = [
    "ComputationReceipt",
    "record_computation",
    "recent_computations",
    "last_computation",
    "asks_how_it_was_computed",
    "how_it_was_computed_block",
    "clear_computation_receipts",
]

#: Enough to answer about the last few, not a log. A person asking "how did
#: you get that" means the answer they just read.
_MAX_RECEIPTS = 8


@dataclass(frozen=True)
class ComputationReceipt:
    """One exact answer and the code object that produced it."""

    question: str
    value: str
    provenance: str
    at: float


_RECEIPTS: deque[ComputationReceipt] = deque(maxlen=_MAX_RECEIPTS)
_LOCK = checked_lock("core.conversation.computation_receipts")


def record_computation(question: str, value: object, provenance: str) -> None:
    """Keep what produced an answer, at the moment it is produced."""
    text = str(question or "").strip()
    detail = str(provenance or "").strip()
    if not text or not detail:
        return
    with _LOCK:
        _RECEIPTS.append(
            ComputationReceipt(
                question=text,
                value=str(value),
                provenance=detail,
                at=time.time(),
            )
        )


def recent_computations() -> tuple[ComputationReceipt, ...]:
    with _LOCK:
        return tuple(_RECEIPTS)


def last_computation() -> ComputationReceipt | None:
    with _LOCK:
        return _RECEIPTS[-1] if _RECEIPTS else None


def clear_computation_receipts() -> None:
    with _LOCK:
        _RECEIPTS.clear()


#: "how did you do that", "was that the model or code", "did you actually
#: calculate it". All of them ask about the METHOD of an answer already given.
_HOW_MARKERS = (
    "how did you do",
    "how did you get",
    "how did you work",
    "how did you compute",
    "how did you calculate",
    "how did you figure",
    "how do you know that",
    "how was that computed",
    "how was that calculated",
    "did you compute",
    "did you calculate",
    "did you actually compute",
    "was that computed",
    "was that calculated",
    "model or code",
    "you or a calculator",
    "did you just guess",
    "did you guess",
    "show your work",
    "what produced that",
    "where did that number come from",
)


def asks_how_it_was_computed(text: str) -> bool:
    """Whether the turn asks by what means the last answer was produced."""
    message = str(text or "").strip().lower()
    if not message:
        return False
    return names_any(message, _HOW_MARKERS)


def how_it_was_computed_block(text: str) -> str:
    """The recorded method for the last computed answer, or nothing."""
    if not asks_how_it_was_computed(text):
        return ""
    receipt = last_computation()
    if receipt is None:
        return ""
    return (
        "How the last exact answer was produced, from the record rather than "
        "from memory:\n"
        f"- question: {receipt.question}\n"
        f"- answer: {receipt.value}\n"
        f"- method: {receipt.provenance}\n"
        "It ran as Python in this process. It was not produced by the language "
        "model, and there is no model capability involved in it."
    )
