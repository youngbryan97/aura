"""What passes between her and each person, both ways.

"That's Love" keeps two things that a ledger of debts cannot: a route kept open
because she may need it, and giving that does not wait to be repaid.
`core/social/receptivity.py` keeps what she is owed and nothing reads it; the
giving side had no record at all.

Kept per person, in the one unit receptivity already weighs acts in, the
log-likelihood a kind act contributes (so a costly kindness counts for more
than a free one, by the same rule the posterior uses):

    received   the kind acts that arrived from them
    given      the regard she sent them, each at the weight of a free kindness
    owed       received less given, when positive
    route      the share of everything she has received that came from them:
               how much of what reaches her would stop if this one closed

`gives_back` scores a reply for the person she is talking to. A reply that
carries regard to somebody she owes is worth more by the share of her world
that reaches her through them, so the routes she depends on are the ones she
keeps giving to. It is zero toward anybody she does not owe, which is what
leaves the other kind of giving alone: regard to somebody who has given her
nothing is scored by everything else and never by what they have returned.
`one_way` counts those people, the giving that has had no answer and kept on.

It is a feature of every candidate, weighed at the prior of the other fits to
the person, and the reply that wins is recorded as given, which is the loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "Between",
    "BetweenLedger",
    "get_between_ledger",
    "reset_for_test",
]


def _weight(cost_to_source: float = 0.0) -> float:
    from core.social.receptivity import evidence_weight

    return float(evidence_weight(True, cost_to_source=max(0.0, float(cost_to_source))))


@dataclass
class Between:
    """One person, both directions."""

    person: str
    received: float = 0.0
    given: float = 0.0

    @property
    def owed(self) -> float:
        return max(0.0, self.received - self.given)

    def as_dict(self) -> dict[str, Any]:
        return {
            "person": self.person,
            "received": round(self.received, 4),
            "given": round(self.given, 4),
            "owed": round(self.owed, 4),
        }


class BetweenLedger:
    """Received and given, per person."""

    def __init__(self) -> None:
        self._people: dict[str, Between] = {}

    def _person(self, name: str) -> Between | None:
        key = str(name or "").strip()
        if not key:
            return None
        held = self._people.get(key)
        if held is None:
            held = Between(person=key)
            self._people[key] = held
        return held

    def note_received(self, person: str, *, cost_to_source: float = 0.0) -> None:
        held = self._person(person)
        if held is not None:
            held.received += _weight(cost_to_source)

    def note_given(self, person: str) -> None:
        held = self._person(person)
        if held is not None:
            held.given += _weight(0.0)

    def route(self, person: str) -> float:
        total = sum(held.received for held in self._people.values())
        held = self._people.get(str(person or "").strip())
        if held is None or total <= 0.0:
            return 0.0
        return held.received / total

    def owed(self, person: str) -> float:
        held = self._people.get(str(person or "").strip())
        return held.owed if held is not None else 0.0

    def gives_back(self, shown: float, person: str) -> float:
        """What giving regard to this person is worth beyond the reply's other merits."""
        if shown <= 0.0 or self.owed(person) <= 0.0:
            return 0.0
        return max(0.0, min(1.0, float(shown) * self.route(person)))

    def one_way(self) -> list[str]:
        """People she has given to who have given nothing back."""
        return sorted(name for name, held in self._people.items() if held.given > 0.0 and held.received <= 0.0)

    def status(self) -> dict[str, Any]:
        return {
            "people": {name: held.as_dict() | {"route": round(self.route(name), 4)} for name, held in sorted(self._people.items())},
            "one_way": self.one_way(),
        }


_LEDGER: BetweenLedger | None = None


def get_between_ledger() -> BetweenLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = BetweenLedger()
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = None
