"""What her own outcome record says has ever actually worked.

She keeps thirty-two thousand outcomes, each with what was attempted and
whether it succeeded, and had no way to read them back as an answer about
herself. Asked on 2026-08-27 how many of her skills had never once executed
successfully, she counted the .py files in a directory and said so — an honest
method for a different question.

The distinction this draws is the one a sceptic actually asks for. A thing
that is registered is a thing somebody wrote down. A thing that is available
is one whose preconditions hold. A thing that has *worked* is one there is a
receipt for, and only the third is evidence.

**What it covers, exactly.** The outcome record holds decisions and
sensorimotor actions. It does not hold skill executions — those resolve
through a different ledger and are not in it — so this cannot answer "which
skills have never run", and saying that plainly is the point rather than a
caveat on it. What it can answer is which recorded actions have ever come off,
which is a real question with a real answer: file_operation:write, on this
machine, is one of 104.

Read-only, bounded, and it never guesses. Something nobody has tried is
reported as untried rather than as broken, because those are different facts
and only one of them is a fault; and a record that cannot be opened returns
nothing rather than an empty verdict, because not being able to check is not
the same as nothing having worked.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from typing import Any, Sequence

from core.runtime.errors import record_degradation

__all__ = ["HowItHasGone", "never_worked", "what_has_ever_worked"]

logger = logging.getLogger("Aura.WhatHasEverWorked")

#: How many distinct actions to read back. Enough to cover every skill she
#: has, small enough that a runaway record cannot turn a question into a scan.
MOST_ACTIONS = 500


def _where_it_is_kept() -> str:
    """Where the outcome record lives, named the same way everything else is.

    Read from configuration rather than by importing the thing that writes it:
    this is a reader, and reaching for the writer would put a package that
    describes her inside the one that learns.
    """
    try:
        from core.config import config

        return str(config.paths.home_dir / "data/outcomes.db")
    except (ImportError, AttributeError, RuntimeError, OSError) as exc:
        record_degradation(
            "what_has_ever_worked", exc, severity="info",
            action="find where the outcome record is kept",
        )
        return ""


@dataclass(frozen=True)
class HowItHasGone:
    """What the record says about one thing she can do."""

    action: str
    tried: int = 0
    worked: int = 0

    @property
    def has_ever_worked(self) -> bool:
        return self.worked > 0

    @property
    def share(self) -> float:
        return self.worked / self.tried if self.tried else 0.0

    def says(self) -> str:
        if not self.tried:
            return f"{self.action}: never tried"
        if not self.worked:
            return f"{self.action}: tried {self.tried}×, never worked"
        return f"{self.action}: {self.worked} of {self.tried} worked ({self.share:.0%})"


def what_has_ever_worked(learner: Any = None) -> dict[str, HowItHasGone]:
    """Every action there is a record of, and how it has gone.

    Reads the outcome record she already keeps. Returns nothing rather than
    raising when the record cannot be opened, because not being able to check
    is a different answer from nothing having worked, and reporting the second
    when the first is true would be a lie about herself.
    """
    where = str(getattr(learner, "_db_path", "") or "") if learner is not None else _where_it_is_kept()
    if not where:
        return {}
    try:
        held = sqlite3.connect(f"file:{where}?mode=ro", uri=True, timeout=5)
    except (sqlite3.Error, OSError) as exc:
        record_degradation(
            "what_has_ever_worked", exc, severity="info",
            action="open the outcome record",
        )
        return {}
    try:
        rows = held.execute(
            "SELECT action, COUNT(*), SUM(CASE WHEN success THEN 1 ELSE 0 END) "
            "FROM outcomes GROUP BY action ORDER BY COUNT(*) DESC LIMIT ?",
            (MOST_ACTIONS,),
        ).fetchall()
    except sqlite3.Error as exc:
        record_degradation(
            "what_has_ever_worked", exc, severity="info",
            action="count what has ever worked",
        )
        return {}
    finally:
        held.close()
    return {
        str(action): HowItHasGone(str(action), int(tried or 0), int(worked or 0))
        for action, tried, worked in rows
        if str(action or "").strip()
    }


def never_worked(known: Sequence[str] = (), learner: Any = None) -> dict[str, tuple[str, ...]]:
    """The three answers a sceptic is really asking for, kept apart.

    ``known`` is what she is registered as being able to do. Without it only
    the record speaks, and a skill nobody has ever tried cannot be seen at all
    — which is exactly the gap that makes a registry sound like evidence.
    """
    gone = what_has_ever_worked(learner)
    named = tuple(str(name).strip() for name in known if str(name or "").strip())
    return {
        "worked": tuple(sorted(n for n, how in gone.items() if how.has_ever_worked)),
        "tried and never worked": tuple(
            sorted(n for n, how in gone.items() if how.tried and not how.worked)
        ),
        "never tried": tuple(sorted(n for n in named if n not in gone)),
    }


def says(known: Sequence[str] = (), learner: Any = None) -> str:
    """What the record says about her, in a line she could be held to."""
    split = never_worked(known, learner)
    if not any(split.values()):
        return "there is no record of anything having been tried"
    return "; ".join(
        f"{len(names)} {label}" for label, names in split.items() if names
    )
