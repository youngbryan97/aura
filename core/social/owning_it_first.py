"""Owning a lapse before it is raised, learned from what followed when she did and when she did not.

In "Runaway" the singer indicts himself before anyone else can, and warns the
person who loves him away from him. Read as a process rather than a mood, that
is getting in ahead of blame: saying the damaging thing about yourself before
somebody else says it. The research name is stealing thunder. Williams,
Bourgeois and Croyle (1993) found that jurors judged a case less harshly when
the side it hurt disclosed the damaging fact first, and Arpan and
Roskos-Ewoldsen (2005) found the same for organisations that disclosed a crisis
before the press did.

Whether it works for her is a question about her own history, so it is learned
from that and nothing else. Two kinds of event, per person:

    owned    she answered from a moment that held what she owed them, before
             they had said anything was wrong
    raised   they told her a reply had failed, and that reply had not been
             answered from such a moment

Each event takes the other person's frustration when it happens, and the next
thing they say closes it with the change. Both therefore measure the same
thing, how they took the reply that followed a lapse coming to light, and
differ only in who brought it to light.

    lift = mean change after raised - mean change after owned

Positive lift means owning it first has gone better for her. It stays
unmeasured until each kind has enough closed events to disagree with itself,
and while it is positive, what she owes gets a larger share of her attention in
proportion to it (see `lifted`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "MIN_SAMPLES",
    "OwningFirst",
    "OwningLedger",
    "get_owning_ledger",
    "lifted",
    "reset_for_test",
]

#: Closed events of each kind before a mean is an estimate. Two always agree
#: with each other and three is the least that can disagree, as in
#: ambivalence and fear of happiness.
MIN_SAMPLES: int = 3

_KINDS: tuple[str, str] = ("owned", "raised")


def _unit(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return max(0.0, min(1.0, number))


@dataclass(frozen=True)
class OwningFirst:
    """What her history says about owning a lapse before it is raised."""

    lift: float = 0.0
    after_owned: float | None = None
    after_raised: float | None = None
    owned_events: int = 0
    raised_events: int = 0
    measured: bool = False
    why: str = "not enough lapses brought to light yet, by her and by them, to compare"

    def as_dict(self) -> dict[str, Any]:
        return {
            "lift": round(self.lift, 6),
            "after_owned": None if self.after_owned is None else round(self.after_owned, 6),
            "after_raised": None if self.after_raised is None else round(self.after_raised, 6),
            "owned_events": self.owned_events,
            "raised_events": self.raised_events,
            "measured": self.measured,
            "why": self.why,
        }


class OwningLedger:
    """Lapse events per person, each closed by the next thing that person says."""

    def __init__(self) -> None:
        self._open: dict[str, tuple[str, float]] = {}
        self._totals: dict[str, list[float]] = {kind: [0.0, 0.0] for kind in _KINDS}
        #: The last reply delivered to each person, and whether it was answered
        #: from a moment that held what she owed them.
        self._last_reply_owned: dict[str, bool] = {}

    def delivered(self, agent_id: str, *, owed_in_mind: bool, frustration: float) -> None:
        """A reply went out. If it was answered from what she owed, that is her owning it first."""
        agent = str(agent_id or "")
        if not agent:
            return
        self._last_reply_owned[agent] = bool(owed_in_mind)
        if owed_in_mind:
            self._begin(agent, "owned", frustration)

    def heard(self, agent_id: str, *, frustration: float, complaint: bool) -> None:
        """They said something. It closes whatever was open, and a complaint may open a new event.

        A complaint about a reply she had already answered from what she owed
        is not raised by them: she got there first, and the event her reply
        opened is what this closes.
        """
        agent = str(agent_id or "")
        if not agent:
            return
        self._close(agent, frustration)
        if complaint and not self._last_reply_owned.get(agent, False):
            self._begin(agent, "raised", frustration)

    def _begin(self, agent: str, kind: str, frustration: float) -> None:
        level = _unit(frustration)
        if level is None:
            return
        self._open[agent] = (kind, level)

    def _close(self, agent: str, frustration: float) -> None:
        pending = self._open.pop(agent, None)
        level = _unit(frustration)
        if pending is None or level is None:
            return
        kind, start = pending
        total = self._totals[kind]
        total[0] += level - start
        total[1] += 1.0

    def reading(self) -> OwningFirst:
        owned_sum, owned_n = self._totals["owned"]
        raised_sum, raised_n = self._totals["raised"]
        counts = {"owned_events": int(owned_n), "raised_events": int(raised_n)}
        if owned_n < MIN_SAMPLES or raised_n < MIN_SAMPLES:
            return OwningFirst(**counts)
        after_owned = owned_sum / owned_n
        after_raised = raised_sum / raised_n
        lift = after_raised - after_owned
        if lift > 0.0:
            why = (
                f"their frustration moved {after_owned:+.2f} after she owned a lapse first "
                f"and {after_raised:+.2f} after they had to raise it"
            )
        else:
            why = "owning a lapse first has not gone better for her than being told"
        return OwningFirst(
            lift=lift,
            after_owned=after_owned,
            after_raised=after_raised,
            measured=True,
            why=why,
            **counts,
        )


def lifted(share: float, reading: OwningFirst) -> float:
    """A share of attention for what she owes, raised by what owning it first has been worth.

    Only a measured, positive lift moves it, and it moves the share towards
    one by that fraction of the distance left, so it cannot pass one.
    """
    base = max(0.0, min(1.0, float(share)))
    if not reading.measured or reading.lift <= 0.0:
        return base
    return base + (1.0 - base) * min(1.0, reading.lift)


#: Made at import rather than on first use. The subject-core fork carries module
#: globals that hold state, and one still None when an anchor is taken is not
#: carried, so a ledger first made inside one arm would reach the next arm with
#: the first arm's events in it.
_ledger: OwningLedger = OwningLedger()


def get_owning_ledger() -> OwningLedger:
    return _ledger


def reset_for_test() -> None:
    global _ledger
    _ledger = OwningLedger()
