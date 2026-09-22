"""About to say something risky: the body quickens, and how it lands is watched.

Bryan, on being about to say something risky: time slows, you watch every word
and say it slowly, to give yourself time to say it right and to see how the
other person takes it as you go, with room to retreat. Three things: the
moment before a risky thing is said is felt; it is felt in proportion to the
risk; and the reaction to it is read more closely than the reaction to
something safe.

She sent every reply the same way. Nothing measured whether what she was about
to say could cost her the exchange, so a correction and a greeting left her
body alike, and the next thing the person said taught her the same amount
whatever she had risked.

    risk    how likely this person is to refuse a reply shaped like this one:
            (1 - fit) / 2, where fit is `core/social/the_form_they_welcome.py`'s
            per-person reading of how they have taken the forms this reply
            carries. An even half with no history; lower where they have
            welcomed replies like it, higher where they have refused them.
    middle  how risky what she says usually is
    shift   this reply against that middle

What the form ledger reads is two forms of regard, regard shown outright and
their own words named back. A reply that risks the exchange some other way,
by disagreeing, refusing or telling a hard truth, reads here only as far as it
carries one of those two. That is the limit of this reading, not a claim that
nothing else is risky.

The shift multiplies her pulse once, in the body reading after the reply goes
out. And the reaction to it teaches the form ledger by one plus the risk, so a
risky reply's reception moves what she learns about this person more than a
safe one's does.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from core.self.what_came_before import keep_across_stages

__all__ = ["Edge", "EdgeLedger", "get_edge_ledger", "reset_for_test", "risk_of"]

#: How much of her own history the middle is taken over.
_WINDOW = 256

#: Replies before a middle is a middle.
_ENOUGH = 3


def risk_of(reply: str, their_message: str = "", *, partner: str = "") -> float:
    """How likely this person is to refuse a reply shaped like this one, in [0, 1]."""
    try:
        from core.social.the_form_they_welcome import get_form_ledger

        fit = float(get_form_ledger().form_fit(reply, their_message, partner=partner))
    except (ImportError, AttributeError, TypeError, ValueError):
        return 0.5
    if fit != fit:
        return 0.5
    return max(0.0, min(1.0, (1.0 - fit) / 2.0))


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


@dataclass
class Edge:
    """How risky the last reply was, against how risky hers usually are."""

    risk: float = 0.0
    middle: float = 0.0
    shift: float = 0.0
    seen: int = 0
    measured: bool = False
    why: str = "nothing has been said yet"

    def as_dict(self) -> dict[str, Any]:
        return {
            "risk": round(self.risk, 6),
            "middle": round(self.middle, 6),
            "shift": round(self.shift, 6),
            "seen": self.seen,
            "measured": self.measured,
            "why": self.why,
        }


class EdgeLedger:
    """The risk of what she says, and whether the body has felt the last one yet."""

    def __init__(self) -> None:
        self._seen: deque[float] = deque(maxlen=_WINDOW)
        self._last = 0.0
        self._unfelt = False

    def note_sent(self, risk: float) -> None:
        try:
            value = float(risk)
        # not a failure: a risk that is not a number is nothing to record,
        # and the ledger keeps the window it already has.
        except (TypeError, ValueError):
            return
        if value != value:
            return
        self._last = max(0.0, min(1.0, value))
        self._seen.append(self._last)
        self._unfelt = True

    def last_risk(self) -> float:
        return self._last

    def read(self) -> Edge:
        seen = len(self._seen)
        if seen < _ENOUGH:
            return Edge(
                risk=self._last,
                seen=seen,
                why=f"{seen} of the {_ENOUGH} replies a middle needs a point either side of",
            )
        middle = _median(list(self._seen))
        shift = self._last - middle
        return Edge(
            risk=self._last,
            middle=middle,
            shift=shift,
            seen=seen,
            measured=True,
            why=(
                f"the last reply risked {self._last:.2f} against the {middle:.2f} "
                f"hers usually risk, so the body reads {shift:+.2f}"
            ),
        )

    def felt_once(self) -> float:
        """The shift for the body, once per reply: the moment, not a mood."""
        if not self._unfelt:
            return 0.0
        self._unfelt = False
        reading = self.read()
        return reading.shift if reading.measured else 0.0


#: Made at import rather than on first use. The subject-core fork carries
#: module globals that hold state, and one still None when an anchor is taken
#: is not carried, so a ledger first made inside one arm would reach the next
#: arm with the first arm's history in it. See core/social/owning_it_first.py.
_LEDGER: EdgeLedger = EdgeLedger()
#: Part of her history, so it is kept across her restarts along her own line.
#: See core/self/what_came_before.py.
keep_across_stages(__name__, "_LEDGER")


def get_edge_ledger() -> EdgeLedger:
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = EdgeLedger()
