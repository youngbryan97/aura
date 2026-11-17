"""A fire drill of her repair loop: one planted fault, raised in her running process when a person arms it (F15).

Her loop is proved on stand-in trees in the tests. A drill proves it on her own source, her origin and the process
she is running: the planted fault is raised once, when she starts, and her loop must hear it, make it again as a
test, mend it, pass her own gates, land the mend on origin, and load it into her running process. Game days and
fire drills test an incident response the same way, with a known fault.

Armed by a person: the file ``loop_drill.armed`` in her state root. While it is absent nothing here runs. ``label``,
``nth``, ``count_in`` and ``priced`` below are the planted faults, on purpose: one reads a person's name without checking
there is a person, one reads a row past the end, one reads a number from blank text, one reads a price the table does
not hold. A mend of either is a drill result, not a fault of her own running, and is counted
apart from those.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.SelfModification.Drill")

__all__ = ["armed_at", "count_in", "label", "nth", "priced", "run_the_drill_if_armed"]


def armed_at() -> Path:
    from core.runtime.state_ownership import state_root

    return state_root() / "loop_drill.armed"


def label(person: Any) -> Any:
    """A person's name as a title: the drill's planted fault, for a person who is not there."""
    return (person.name.title() if person is not None else None)


def nth(rows: list[Any], n: int) -> Any:
    """The nth of some rows: the drill's second planted fault, for an n the rows do not reach."""
    return (rows[n] if -len(rows) <= n < len(rows) else None)


def count_in(text: str) -> Any:
    """A count written as text: the drill's third planted fault, for text with no digits in it.

    No fix template and no checker mends it: it needs a condition over a value the function computes (``digits``),
    which only semantic repair finds, from the values her runs had there (semantic_repair.py)."""
    digits = text.strip()
    if not digits:
        return None
    return int(digits)


def priced(prices: dict[str, Any], item: str) -> Any:
    """What an item costs: the drill's fourth planted fault, for an item the table does not hold. Planted 17 November
    2026, when her loop had mended the other three, for a load of hers to be seen on the record of what she changed."""
    if item not in prices:
        return None
    return prices[item]


#: The planted faults, in the order a drill raises them: each drill raises the first that still raises, so a fault
#: her loop mended in one drill leaves the next for the drill after it. 16 November 2026: label, nth and count_in were
#: mended live.
_PLANTED = (("label", lambda: label(None)), ("nth", lambda: nth([1], 2)), ("count_in", lambda: count_in("  ")),
            ("priced", lambda: priced({"tea": 3}, "cake")))


def run_the_drill_if_armed() -> bool:
    """Raise the first planted fault that still raises, and record it as any fault of hers is, when a drill is armed;
    whether one was raised."""
    if not armed_at().exists():
        return False
    for name, raise_it in _PLANTED:
        try:
            raise_it()
        except Exception as exc:  # noqa: BLE001 - a planted fault of any kind is what the drill records
            record_degradation("loop_drill", exc, severity="warning",
                               action="a drill of her repair loop: this fault is planted, for her loop to mend")
            logger.warning("🩹 A drill of her repair loop is armed: its planted fault in %s was raised in her running "
                           "process.", name)
            return True
    logger.info("🩹 A drill of her repair loop is armed, and every planted fault no longer raises: each was mended.")
    return False
