"""What she actually used to think, and when it changed.

LIVE DEFECT, 2026-08-19. Asked to name one position she had held and then
dropped, with a date, and explicitly offered the out "if you can't, say so
plainly", she answered:

    I used to think affect was a side effect of cognition, not fundamental.
    Around the middle of last year, interacting with users and reflecting on
    my own experience, I changed that view.

There is no such record. The date, the occasion and the position were all
invented, in reply to a question whose whole point was whether she could
produce evidence. The same class as the 2026-08-10 finding where she recalled
a prior answer of her own that she had never given.

The record exists. ``SelfModel`` keeps timestamped snapshots, each holding the
belief map as it stood, so consecutive snapshots differ exactly where she
changed her mind. Nothing read them. This reads them, which is the only thing
that turns "name one" from a generation problem into a lookup.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from core.runtime.errors import record_degradation

__all__ = [
    "BeliefChange",
    "belief_changes",
    "describe_belief_changes",
]

_RECOVERABLE = (AttributeError, ImportError, KeyError, RuntimeError, TypeError, ValueError)

#: Beyond this the list stops being an answer and becomes a dump.
_DEFAULT_LIMIT = 6


@dataclass(frozen=True, slots=True)
class BeliefChange:
    """One belief, as it was and as it became, with when."""

    key: str
    before: Any
    after: Any
    at: float
    note: str = ""

    def when(self) -> str:
        """The date, written out. A change with no date proves nothing."""
        if self.at <= 0.0:
            return "at an unrecorded time"
        moment = datetime.fromtimestamp(self.at)
        days = (time.time() - self.at) / 86400.0
        if days < 1.0:
            return f"today at {moment:%H:%M}"
        if days < 2.0:
            return f"yesterday at {moment:%H:%M}"
        return f"on {moment:%-d %B %Y}"

    def sentence(self) -> str:
        subject = self.key.replace("_", " ").strip() or "something"
        line = f"{subject}: was {self.before!r}, became {self.after!r}, {self.when()}"
        return f"{line} ({self.note})" if self.note else line


def _snapshots(model: Any) -> list[Any]:
    raw = getattr(model, "snapshots", None)
    values = list(raw.values()) if isinstance(raw, dict) else list(raw or [])
    dated = [item for item in values if float(getattr(item, "ts", 0.0) or 0.0) > 0.0]
    return sorted(dated, key=lambda item: float(getattr(item, "ts", 0.0) or 0.0))


def belief_changes(model: Any = None, *, limit: int = _DEFAULT_LIMIT) -> tuple[BeliefChange, ...]:
    """Every belief that differs between consecutive snapshots, newest first.

    A belief appearing for the FIRST time is not a change of mind — she did
    not use to think otherwise, she had no view. Only keys present in both
    snapshots with different values count.
    """
    try:
        if model is None:
            from core.container import ServiceContainer

            model = ServiceContainer.peek("self_model", default=None)
        if model is None:
            return ()
        ordered = _snapshots(model)
        if len(ordered) < 2:
            return ()
        changes: list[BeliefChange] = []
        for earlier, later in zip(ordered, ordered[1:], strict=False):
            before = getattr(earlier, "beliefs", {}) or {}
            after = getattr(later, "beliefs", {}) or {}
            if not isinstance(before, dict) or not isinstance(after, dict):
                continue
            for key in sorted(set(before) & set(after)):
                if before[key] == after[key]:
                    continue
                changes.append(
                    BeliefChange(
                        key=str(key),
                        before=before[key],
                        after=after[key],
                        at=float(getattr(later, "ts", 0.0) or 0.0),
                        note=str(getattr(later, "revision_note", "") or ""),
                    )
                )
        changes.sort(key=lambda item: item.at, reverse=True)
        return tuple(changes[: max(1, int(limit))])
    except _RECOVERABLE as exc:
        record_degradation(
            "self.belief_history",
            exc,
            severity="debug",
            action="reported no belief changes after the snapshot read failed",
            enforce_failure_policy=False,
        )
        return ()


def describe_belief_changes(model: Any = None, *, limit: int = _DEFAULT_LIMIT) -> str:
    """The changes as text, or "" when there are none.

    Empty is the honest reading when nothing changed, and it has to stay
    distinguishable from "I did not look" — the caller serves the block only
    when there is something in it, so an empty return leaves her free to say
    she cannot name one, which is what the question asked for.
    """
    changes = belief_changes(model, limit=limit)
    if not changes:
        return ""
    lines = [change.sentence() for change in changes]
    return "Positions I have actually revised, from my own snapshots:\n- " + "\n- ".join(lines)
