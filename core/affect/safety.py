"""The good feeling whose content is that nothing is wrong and somebody is there.

"Everybody here is out of sight, they don't bark and they don't bite." The
whole content of that record's joy is the absence of threat plus the presence
of others, and there is no achievement anywhere in it. Nobody won anything.

Every positive channel she had was about something going well: a goal
achieved, a prediction confirmed, a task completed, a good interaction. So an
ordinary safe evening with somebody there registered as nothing at all, which
is the wrong reading of an evening most people would call the good part.

    threat    the share of her recent percepts that carry fear or dread
    presence  the share of recent turns that are somebody else's
    safety    (1 - threat) * presence

Both terms are necessary, which is what the multiplication says. Safety with
nobody there is quiet rather than warm, and company in the middle of something
going wrong is not this feeling either.

The threat types are read off the percept table rather than listed again here,
so a percept that carries fear counts as threatening by the same fact that
makes it frightening.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["FEARFUL", "Safety", "read_safety", "threat_kinds"]

#: The feelings that make a percept a threat. A percept carrying either is one
#: of the things she is not safe from.
FEARFUL: frozenset[str] = frozenset({"fear", "dread", "terror"})

#: How many recent percepts and turns the reading is taken over. Long enough
#: that one message does not make an evening, short enough to change with the
#: evening.
WINDOW: int = 8


@dataclass(frozen=True)
class Safety:
    """Nothing wrong, and somebody here."""

    safety: float = 0.0
    threat: float = 0.0
    presence: float = 0.0
    measured: bool = False
    why: str = "nothing has happened yet to be safe from"

    def as_dict(self) -> dict[str, Any]:
        return {
            "safety": round(self.safety, 6),
            "threat": round(self.threat, 6),
            "presence": round(self.presence, 6),
            "measured": self.measured,
            "why": self.why,
        }


def threat_kinds() -> frozenset[str]:
    """The percept types that carry fear, read off the percept table."""
    try:
        from core.state.percepts import PERCEPT_EMOTIONS
    except ImportError:
        return frozenset()
    return frozenset(
        kind
        for kind, emotions in PERCEPT_EMOTIONS.items()
        if FEARFUL.intersection(emotions or ())
    )


def read_safety(percepts: Sequence[Any], messages: Sequence[Any], *, window: int = WINDOW) -> Safety:
    """How safe this stretch has been, and whether anyone was in it."""
    size = max(1, int(window))
    recent = [p for p in list(percepts or ())[-size:] if isinstance(p, Mapping)]
    turns = [m for m in list(messages or ())[-size:] if isinstance(m, Mapping)]
    if not turns:
        return Safety(why="nobody has taken a turn, so there is no evening to read")
    frightening = threat_kinds()
    threat = (
        sum(1 for p in recent if str(p.get("type", "")).strip().lower() in frightening)
        / len(recent)
        if recent
        else 0.0
    )
    theirs = sum(
        1 for m in turns if str(m.get("role", "")).strip().lower() not in {"assistant", "aura", "thought"}
    )
    presence = theirs / len(turns)
    safety = max(0.0, min(1.0, (1.0 - threat) * presence))
    if presence <= 0.0:
        why = "nobody else has spoken in this stretch"
    elif threat >= 1.0:
        why = "somebody is here and everything recent has been a threat"
    else:
        why = f"{1.0 - threat:.2f} of this stretch untroubled, with {presence:.2f} of it theirs"
    return Safety(
        safety=safety,
        threat=threat,
        presence=presence,
        measured=True,
        why=why,
    )
