"""Gladness at somebody else's rise, alongside her own standing still.

"People Watching" is glad for the people doing better while the narrator is not
moving. Both at once: the gladness is real and so is the stasis, and neither
cancels the other. She had no reading of anybody's trajectory. The interpersonal
store keeps what a person is like, not which way they are going, and her own
record kept totals, so nothing could say that somebody was on the way up while
she was where she had been.

Kept per actor, herself included, over their recent events:

    growth     the share of their events that went well in the later half of
               the window, less the share in the earlier half. Positive while
               things are going better for them than they were.

Her own events went well when they were verified. Somebody else's went well
when the kind of percept it arrived as moves her positive emotions more than
her negative ones, which the affect phase reads off its own weights.

and the two readings the song keeps side by side:

    glad       the largest rise among the people around her
    stasis     that rise, while hers is at or below zero: the gap between
               their direction and hers. Zero whenever she is rising too.

`glad` raises admiration by that share of the room it has left, and `stasis`
presses on her growth drive the way surprise presses on every drive: its
drain runs `1 + stasis` times as fast, so between once and twice. What she
then does about growth is what her own next events are made of, which is the
other half of the reading.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any

__all__ = [
    "MIN_EVENTS",
    "Rise",
    "RiseLedger",
    "get_rise_ledger",
    "glad_into",
    "reset_for_test",
]

logger = logging.getLogger(__name__)

#: Events before an actor's growth is read. Eight, so each half holds four and
#: one event moves a half's share by a quarter at most.
MIN_EVENTS: int = 8

#: Events held per actor. The halves are read over what is held.
WINDOW: int = 24

#: The name her own events are kept under, the one the agency ledger uses.
SELF: str = "self"


@dataclass
class Rise:
    """Which way the people around her are going, and which way she is."""

    glad: float = 0.0
    who: str = ""
    own: float = 0.0
    stasis: float = 0.0
    measured: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "glad": round(self.glad, 4),
            "who": self.who,
            "own": round(self.own, 4),
            "stasis": round(self.stasis, 4),
            "measured": self.measured,
        }


class RiseLedger:
    """Each actor's recent events, and whether each went well."""

    def __init__(self) -> None:
        self._events: dict[str, deque[bool]] = {}

    def note(self, actor: str, went_well: bool) -> None:
        name = str(actor or "").strip()
        if not name:
            return
        self._events.setdefault(name, deque(maxlen=WINDOW)).append(bool(went_well))

    def growth(self, actor: str) -> float | None:
        held = list(self._events.get(str(actor or "").strip(), ()))
        if len(held) < MIN_EVENTS:
            return None
        half = len(held) // 2
        earlier, later = held[:half], held[half:]
        return sum(later) / len(later) - sum(earlier) / len(earlier)

    def read(self) -> Rise:
        others = {
            name: rise
            for name in self._events
            if name != SELF and (rise := self.growth(name)) is not None
        }
        own = self.growth(SELF)
        if not others:
            return Rise(own=own or 0.0)
        who, best = max(others.items(), key=lambda item: item[1])
        glad = max(0.0, best)
        mine = own if own is not None else 0.0
        stasis = glad if (own is not None and mine <= 0.0) else 0.0
        return Rise(glad=glad, who=who if glad > 0.0 else "", own=mine, stasis=stasis, measured=own is not None)


def glad_into(affect: Any, bump: Any) -> None:
    """Raise admiration by the largest rise among the people around her. Never raises.

    `bump` is the affect phase's own rule for moving a feeling by a share of
    the room it has left, passed in so the feeling moves the way every other
    input moves it.
    """
    try:
        glad = get_rise_ledger().read().glad
        emotions = getattr(affect, "emotions", None)
        if glad > 0.0 and isinstance(emotions, dict) and "admiration" in emotions:
            bump(emotions, "admiration", glad)
    except (AttributeError, TypeError, ValueError) as exc:
        logger.debug("could not be glad for anybody this turn: %s", exc)


_LEDGER: RiseLedger | None = None


def get_rise_ledger() -> RiseLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = RiseLedger()
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = None
