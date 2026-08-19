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

import re
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


#: A position is something she holds. Per-tick machine state is not.
#:
#: The live self-model's only belief keys are `executive_closure` and
#: `runtime_lessons`, both dictionaries rewritten on nearly every snapshot —
#: the dominant need, the current attention focus, the last lesson. Diffing
#: them yields a wall of nested dict text that changes constantly, and
#: reporting it as "a position I have revised" would be both unreadable and
#: untrue. Two properties separate the two, and neither needs a list of keys:
#: a stance has a value you can say out loud, and a stance does not change
#: every time the clock ticks.
_MAX_STANCE_CHARS = 120
_CHURN_FRACTION = 0.5


def _is_a_stance(value: Any) -> bool:
    """True when the value is something she could state as a position."""
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        return 0 < len(value.strip()) <= _MAX_STANCE_CHARS
    return False


def belief_changes(model: Any = None, *, limit: int = _DEFAULT_LIMIT) -> tuple[BeliefChange, ...]:
    """Positions that differ between consecutive snapshots, newest first.

    A belief appearing for the FIRST time is not a change of mind — she did
    not use to think otherwise, she had no view. Only keys present in both
    snapshots with different values count, and only where the value is a
    stance rather than machine state that moves on its own.
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
        transitions = len(ordered) - 1
        moved: dict[str, int] = {}
        candidates: list[BeliefChange] = []
        for earlier, later in zip(ordered, ordered[1:], strict=False):
            before = getattr(earlier, "beliefs", {}) or {}
            after = getattr(later, "beliefs", {}) or {}
            if not isinstance(before, dict) or not isinstance(after, dict):
                continue
            for key in sorted(set(before) & set(after)):
                if before[key] == after[key]:
                    continue
                moved[str(key)] = moved.get(str(key), 0) + 1
                if not (_is_a_stance(before[key]) and _is_a_stance(after[key])):
                    continue
                candidates.append(
                    BeliefChange(
                        key=str(key),
                        before=before[key],
                        after=after[key],
                        at=float(getattr(later, "ts", 0.0) or 0.0),
                        note=str(getattr(later, "revision_note", "") or ""),
                    )
                )
        # A key that moves on most transitions is being written by the runtime,
        # not reconsidered by her.
        churn_ceiling = max(1, int(transitions * _CHURN_FRACTION))
        changes = [item for item in candidates if moved.get(item.key, 0) <= churn_ceiling]
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
    """The changes as text, or "" when there is no record to read.

    An empty block and a block saying "none" are different answers. Silence
    leaves the model free to invent a revision, which is exactly what happened;
    a measured "none, out of N snapshots since <date>" gives her something true
    to say instead. So "" is reserved for having no record at all, which is the
    one case where there is genuinely nothing to report.
    """
    try:
        if model is None:
            from core.container import ServiceContainer

            model = ServiceContainer.peek("self_model", default=None)
        ordered = _snapshots(model) if model is not None else []
    except _RECOVERABLE:
        ordered = []
    changes = belief_changes(model, limit=limit)
    if changes:
        lines = [change.sentence() for change in changes]
        return (
            "Positions I have actually revised, from my own snapshots:\n- "
            + "\n- ".join(lines)
        )
    if len(ordered) < 2:
        return ""
    first = datetime.fromtimestamp(float(getattr(ordered[0], "ts", 0.0) or 0.0))
    return (
        f"My record holds {len(ordered)} snapshots since {first:%-d %B %Y} and none of "
        "them shows a position I revised. I cannot name one from evidence."
    )


#: A reply that names a position she used to hold.
#:
#: LIVE, 2026-08-19. The reading REACHED the model — the log records
#: "took 1 reading(s): positions i have actually revised" and
#: "survived to dispatch: present,receipts,belief_history" — and the reply
#: still opened "I held the position that affect was a side effect of
#: cognition ... around the middle of last year". Evidence informs; it does
#: not enforce. The same conclusion response_generation reached about file
#: counts, and the same remedy: serve the fact rather than ask for it again.
_CLAIMS_A_REVISION_RE = re.compile(
    r"\bi\s+used\s+to\s+(?:think|believe|hold|assume)\b"
    r"|\bi\s+(?:held|had)\s+(?:the\s+)?(?:position|view|belief|opinion)\b"
    r"|\bi\s+(?:changed|revised|dropped|abandoned)\s+(?:that|my|this)\s+"
    r"(?:view|position|belief|mind|opinion)\b"
    r"|\bi\s+no\s+longer\s+(?:think|believe|hold)\b"
    r"|\bi\s+once\s+(?:thought|believed|held)\b",
    re.IGNORECASE,
)


def reply_claims_a_revision(reply: Any) -> bool:
    """True when a reply names a position she used to hold."""
    return bool(_CLAIMS_A_REVISION_RE.search(str(reply or "")))


def unevidenced_revision_claim(reply: Any, model: Any = None) -> str:
    """The measured answer when a reply claims a revision the record denies.

    Returns "" when there is nothing to correct: no claim, no record to check
    against, or a record that actually supports one. Only the case where she
    names a revision her own snapshots do not contain is replaced, and it is
    replaced with the reading rather than with a refusal.
    """
    if not reply_claims_a_revision(reply):
        return ""
    if belief_changes(model):
        return ""
    reading = describe_belief_changes(model)
    return reading if "cannot name one from evidence" in reading else ""
