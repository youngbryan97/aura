"""What a counter counts over, and what resetting it is allowed to mean.

Soar makes statistics reset semantics explicit, and the closure asked for the
same: every metric declares a lifetime domain — turn, session, boot, lifetime
— and reset APIs operate by domain and never silently mix them.

Aura accumulated a lot of counters this year and almost none of them says what
it counts over. That is not tidiness. "Routes offered: 4,812" means one thing
since boot and another thing this turn, and a reader who guesses wrong draws
the opposite conclusion. Worse, a reset that clears everything takes the
lifetime counters with it, and a lifetime counter that can be reset is not a
lifetime counter.

So a counter is registered with its domain, and ``reset(domain)`` touches only
the counters that declared it. Resetting a domain nothing declared is refused
rather than doing nothing quietly, because a reset that silently matches
nothing looks exactly like a reset that worked.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable

logger = logging.getLogger("Aura.HowLongANumberLives")

__all__ = [
    "HowLong",
    "declare_a_number",
    "how_the_numbers_stand",
    "reset_the_numbers_for",
    "what_has_no_declared_lifetime",
]


class HowLong(StrEnum):
    """What a number counts over. Four, and they do not nest."""

    #: Cleared at the start of every turn. Answers "what happened just now".
    TURN = "turn"
    #: Cleared when a conversation ends.
    SESSION = "session"
    #: Cleared when the process restarts. Most counters are this and few say so.
    BOOT = "boot"
    #: Never cleared. A lifetime counter that can be reset is not one.
    LIFETIME = "lifetime"


@dataclass
class ANumber:
    """One counter, what it counts over, and how to clear it."""

    name: str
    lives: HowLong
    what_it_counts: str
    clear: Callable[[], None] | None = None
    read: Callable[[], Any] | None = None
    cleared: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "lives": str(self.lives),
            "counts": self.what_it_counts,
            "can_be_cleared": self.clear is not None,
            "cleared": self.cleared,
        }


_NUMBERS: dict[str, ANumber] = {}
_LOCK = threading.RLock()


def declare_a_number(
    name: str,
    lives: HowLong,
    what_it_counts: str,
    *,
    clear: Callable[[], None] | None = None,
    read: Callable[[], Any] | None = None,
) -> ANumber:
    """Say what this counter counts over. Declaring twice replaces.

    A LIFETIME counter may not be given a way to clear it. That is the whole
    content of the word, and accepting one would make the domain decorative.
    """
    if lives is HowLong.LIFETIME and clear is not None:
        raise ValueError(
            f"{name} is a lifetime counter and was given a way to clear it; "
            "a lifetime counter that can be reset is not a lifetime counter"
        )
    if not what_it_counts.strip():
        raise ValueError(f"{name} does not say what it counts")
    number = ANumber(
        name=str(name), lives=lives, what_it_counts=str(what_it_counts),
        clear=clear, read=read,
    )
    with _LOCK:
        _NUMBERS[number.name] = number
    return number


def reset_the_numbers_for(lives: HowLong) -> list[str]:
    """Clear every counter that declared this domain. Nothing else.

    Refuses a domain nothing declared: a reset that silently matches nothing
    looks exactly like a reset that worked.
    """
    with _LOCK:
        mine = [one for one in _NUMBERS.values() if one.lives is lives]
    if not mine:
        raise KeyError(
            f"nothing declared the {lives} domain; "
            f"declared: {sorted({str(one.lives) for one in _NUMBERS.values()})}"
        )
    cleared: list[str] = []
    for one in mine:
        if one.clear is None:
            continue
        try:
            one.clear()
        except Exception as exc:  # noqa: BLE001 — one stuck counter is not all of them
            logger.warning("%s would not clear: %s", one.name, exc)
            continue
        one.cleared += 1
        cleared.append(one.name)
    return sorted(cleared)


def how_the_numbers_stand() -> dict[str, Any]:
    """Every declared counter, by what it counts over."""
    with _LOCK:
        held = list(_NUMBERS.values())
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for one in held:
        by_domain.setdefault(str(one.lives), []).append(one.to_dict())
    return {
        "declared": len(held),
        "by_domain": {name: len(rows) for name, rows in sorted(by_domain.items())},
        "numbers": by_domain,
        "what_this_means": (
            "a count means one thing since boot and another this turn; a "
            "reader who guesses wrong draws the opposite conclusion"
        ),
    }


def what_has_no_declared_lifetime(names: Any) -> list[str]:
    """Which of these counters never said what they count over."""
    with _LOCK:
        known = set(_NUMBERS)
    return sorted(str(one) for one in (names or ()) if str(one) not in known)


def forget_everything() -> None:
    """For tests. The live runtime never calls this."""
    with _LOCK:
        _NUMBERS.clear()
