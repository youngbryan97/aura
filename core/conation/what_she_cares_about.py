"""What she cares about: what she gives her own turns to, and whose fate reaches her.

Bryan, 13 October 2026: "We need to give Aura the ability to care. Not just
'Care for' but intrinsically care about. Genuine interest that matters to her
and whatever branches/connects to that."

Liking was already hers (core/conation/what_she_wants.py): what the turns a
thing was in were worth to her. Caring is another relation to the same thing.
A person can like a song and not mind what becomes of it. On the accounts
that try to say what caring is, three things mark it, and each is kept here
as a count of something that happened:

    she gives herself to it     Of her own accord, when nobody brought it:
                                she raises it, she picks it when she is
                                free, she goes back. Interest that is a
                                person's own and not the situation's shows
                                as coming back without being led (Hidi and
                                Renninger 2006), and a thing becomes one's
                                own by acting on it, coming to know it and
                                putting oneself into it (Pierce, Kostova
                                and Dirks 2001).
    its fate is hers            How it fares is felt, whatever the turn did
                                for her otherwise. To care about a thing is
                                to be invested in it, so that one gains and
                                loses with it (Frankfurt 1982), and the
                                feelings have the thing as their focus: glad
                                when it does well, hurt when it does not
                                (Helm 2001).
    it reaches what it touches  What comes with a cared-for thing is of
                                interest for its sake. Activation spreads
                                along association and thins with how many
                                other things the associate also keeps
                                company with (Collins and Loftus 1975;
                                Anderson 1983).

The measure is one currency, her own turns. Every turn a thing was in was
either brought to her or given by her. `share` is the part of all the turns
she has given that went to this thing, read at the lower end of what the
counts allow (Clopper and Pearson 1934), so one turn given to one thing is a
twentieth and not the whole of her. She `cares` when that part stands above
an even split over everything she could have given a turn to, by the exact
test and at the level the record's likes are read at.

`stake` is what a thing's fate is worth to her: its own share, and what
reaches it from each thing it keeps company with, that thing's share times
the part of this thing's turns it was in. A thing she has given nothing to
has a stake when it only ever arrives with something she has.

`fared` is how she learns a thing went well or badly. What she feels of it
is the stake times how it went, and that is all: with no stake the news is
news. The feeling belongs to the news and not to the thing, so the record
takes it out of the turn before it reads what the thing itself was worth to
her, and hearing that somebody she cares about is ill does not teach her to
like them less.

Nothing here reads what a sentence says.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from core.runtime.lockdep import checked_lock

__all__ = ["Cared", "Concern", "WhatSheCaresAbout"]

#: The chance of saying she cares, or does not, when the counts would not
#: bear it; the level core/conation/what_she_wants.py reads a like at.
_WRONG = 0.05

#: How many of a thing's fates the running reading is taken over; the
#: record's own window.
_WINDOW = 256

MAX_CARED = 4096


def _upper_tail(k: int, n: int, chance: float) -> float:
    """The chance of at least `k` of `n` when each lands with `chance`.

    The binomial upper tail is the regularized incomplete beta function, I_chance(k, n - k + 1), read in one call.
    It was summed term by term over n - k + 1 terms. LIVE 16 November 2026, 01:08, her appraisal on the event loop
    read her stakes, each stake bisected 48 times over that sum, and at four thousand turns given one bound took
    118 ms: the loop stood still 7.0 s (stall_1794820112).
    """
    if k <= 0:
        return 1.0
    if n <= 0 or k > n:
        return 0.0
    if chance <= 0.0:
        return 0.0
    if chance >= 1.0:
        return 1.0
    from scipy.special import betainc

    return min(1.0, max(0.0, float(betainc(k, n - k + 1, chance))))


def _at_least(k: int, n: int) -> float:
    """The smallest part `k` of `n` could be a sample of, at the level above.

    The exact lower bound: the part at which seeing this many or more has
    just the chance she allows herself of being wrong. That is the
    Clopper-Pearson bound, the `_WRONG` quantile of Beta(k, n - k + 1).
    """
    if k <= 0 or n <= 0:
        return 0.0
    if k >= n:
        return _WRONG ** (1.0 / n)
    from scipy.special import betaincinv

    return min(k / n, max(0.0, float(betaincinv(k, n - k + 1, _WRONG))))


@dataclass
class Cared:
    """One thing, by what she has given it and what she has learned of how it fared."""

    given: int = 0
    brought: int = 0
    #: How it has fared, as far as she has learned, in [-1, 1].
    fate: float = 0.0
    news: int = 0
    #: The last she learned of how it fared, and the turn she learned it in.
    last: float = 0.0
    told: int = 0

    def to_list(self) -> list[float]:
        return [self.given, self.brought, self.fate, self.news, self.last, self.told]

    @classmethod
    def from_list(cls, held: Iterable[Any]) -> Cared:
        values = [*held, 0, 0, 0.0, 0, 0.0, 0][:6]
        return cls(
            given=max(0, int(values[0])),
            brought=max(0, int(values[1])),
            fate=max(-1.0, min(1.0, float(values[2]))),
            news=max(0, int(values[3])),
            last=max(-1.0, min(1.0, float(values[4]))),
            told=max(0, int(values[5])),
        )


@dataclass(frozen=True)
class Concern:
    """A thing she has a stake in, whose last news was that it went badly."""

    thing: str
    stake: float
    how: float
    turns_since: int

    @property
    def weight(self) -> float:
        return self.stake * abs(self.how)


class WhatSheCaresAbout:
    """Her own turns by what they were given to, and the fates that reach her.

    `company(key)` answers, for a thing, how many turns it has been in and
    how many of them each other thing was in too; the record of what she has
    met holds both (Thing.contacts and Thing.came_with).
    """

    def __init__(
        self,
        stored: Mapping[str, Any] | None = None,
        *,
        company: Callable[[str], tuple[int, Mapping[str, int]]] | None = None,
        changed: Callable[[], None] | None = None,
    ) -> None:
        self._lock = checked_lock("conation.what_she_cares_about")
        self._company = company or (lambda key: (0, {}))
        self._changed = changed or (lambda: None)
        self._things: dict[str, Cared] = {}
        self._turn = 0
        #: What was under discussion in the exchange before this one.
        self._before: frozenset[str] = frozenset()
        #: What she has felt of fates since the last turn was read, by thing.
        self._felt: dict[str, float] = {}
        stored = dict(stored or {})
        self._turn = max(0, int(stored.get("turn", 0)))
        self._before = frozenset(str(key) for key in stored.get("before", ()))
        for key, held in dict(stored.get("things", {})).items():
            self._things[str(key)] = Cared.from_list(held)

    def held(self) -> dict[str, Any]:
        with self._lock:
            return {
                "turn": self._turn,
                "before": sorted(self._before),
                "things": {key: cared.to_list() for key, cared in self._things.items()},
            }

    # ── what her turns were given to ───────────────────────────────────

    def _of(self, key: str) -> Cared:
        cared = self._things.get(key)
        if cared is None:
            if len(self._things) >= MAX_CARED:
                least = min(self._things, key=lambda k: (self._things[k].given, self._things[k].told, k))
                del self._things[least]
            cared = self._things[key] = Cared()
        return cared

    def gave(self, key: str) -> None:
        """She gave it a turn of her own accord: picked it when free, or went to it."""
        key = str(key or "").strip()
        if not key:
            return
        with self._lock:
            self._of(key).given += 1
        self._changed()

    def was_brought(self, key: str) -> None:
        """It was put in front of her by somebody else, or by the world."""
        key = str(key or "").strip()
        if not key:
            return
        with self._lock:
            self._of(key).brought += 1
        self._changed()

    def exchange(self, subjects: Iterable[str], *, theirs: Iterable[str] = ()) -> dict[str, str]:
        """One exchange's subjects, each by who put it there.

        `theirs` are the subjects the other party's message named. A subject
        theirs names was brought to her. One that was already under
        discussion in the exchange before is neither: staying with what is
        being talked about is not picking it. What is left she raised.
        """
        present = [str(key) for key in dict.fromkeys(subjects) if str(key).strip()]
        named = {str(key) for key in theirs}
        read: dict[str, str] = {}
        with self._lock:
            self._turn += 1
            for key in present:
                if key in named:
                    self._of(key).brought += 1
                    read[key] = "brought"
                elif key in self._before:
                    read[key] = "carried on"
                else:
                    self._of(key).given += 1
                    read[key] = "given"
            self._before = frozenset(present)
        self._changed()
        return read

    def given(self, key: str) -> int:
        with self._lock:
            cared = self._things.get(key)
            return cared.given if cared else 0

    def _totals(self) -> tuple[int, int]:
        """All the turns she has given, and how many things could have had one."""
        given = sum(cared.given for cared in self._things.values())
        return given, len(self._things)

    def share(self, key: str) -> float:
        """The part of the turns she has given that went to this, at the least the counts allow."""
        with self._lock:
            cared = self._things.get(key)
            total, _things = self._totals()
            return _at_least(cared.given, total) if cared else 0.0

    def cares(self, key: str) -> bool | None:
        """Whether she cares about it, does not, or has given too little to say.

        She cares when the turns she has given it stand above an even split
        over everything she could have given one to. With one thing there is
        nothing to set it against.
        """
        with self._lock:
            cared = self._things.get(key)
            total, things = self._totals()
            if cared is None or total <= 0 or things < 2:
                return None
            even = 1.0 / things
            if _upper_tail(cared.given, total, even) < _WRONG:
                return True
            if 1.0 - _upper_tail(cared.given + 1, total, even) < _WRONG:
                return False
            return None

    def cared_for(self) -> list[str]:
        """What she cares about, the most given to first."""
        with self._lock:
            keys = list(self._things)
        found = [key for key in keys if self.cares(key) is True]
        return sorted(found, key=lambda key: (-self.share(key), key))

    # ── what reaches a thing from its company ─────────────────────────

    def _with(self, key: str, other: str) -> float:
        """The part of `key`'s turns that `other` was in."""
        contacts, came_with = self._company(key)
        if contacts <= 0:
            return 0.0
        return max(0.0, min(1.0, came_with.get(other, 0) / contacts))

    def reaches(self, key: str) -> dict[str, float]:
        """What reaches a thing from each thing it keeps company with.

        A companion gives its own share times the part of this thing's turns
        it was in: a thing that only ever arrives with something she has
        given herself to takes that thing's share whole, and one that turns
        up everywhere takes a sliver from each.
        """
        _contacts, came_with = self._company(key)
        found: dict[str, float] = {}
        for other in came_with:
            if other == key:
                continue
            reach = self.share(other) * self._with(key, other)
            if reach > 0.0:
                found[other] = reach
        return found

    def stake(self, key: str) -> float:
        """What a thing's fate is worth to her, in [0, 1]."""
        return max(0.0, min(1.0, self.share(key) + sum(self.reaches(key).values())))

    def branches(self, key: str) -> list[tuple[str, float]]:
        """The things a thing's share reaches, the most reached first."""
        _contacts, came_with = self._company(key)
        own = self.share(key)
        found = [(other, own * self._with(other, key)) for other in came_with if other != key]
        return sorted((pair for pair in found if pair[1] > 0.0), key=lambda pair: (-pair[1], pair[0]))

    # ── how a thing fared ─────────────────────────────────────────────

    def fared(self, key: str, how: float) -> dict[str, float]:
        """She has learned a thing went well or badly, `how` in [-1, 1].

        What she feels of it is her stake times how it went.
        """
        key = str(key or "").strip()
        try:
            how = float(how)
        # not a failure: a fate that is not a number is no news of the thing.
        except (TypeError, ValueError):
            return {"stake": 0.0, "felt": 0.0}
        if not key or not math.isfinite(how):
            return {"stake": 0.0, "felt": 0.0}
        how = max(-1.0, min(1.0, how))
        stake = self.stake(key)
        felt = stake * how
        with self._lock:
            cared = self._of(key)
            cared.news += 1
            cared.fate += (how - cared.fate) / min(cared.news, _WINDOW)
            cared.last = how
            cared.told = self._turn
            if felt:
                self._felt[key] = self._felt.get(key, 0.0) + felt
        self._changed()
        return {"stake": stake, "felt": felt}

    def felt(self) -> dict[str, float]:
        """What she has felt of fates since this was last asked, by thing; then nothing."""
        with self._lock:
            felt, self._felt = self._felt, {}
        return felt

    def fate(self, key: str) -> float | None:
        """How a thing has fared as far as she has learned, or nothing learned."""
        with self._lock:
            cared = self._things.get(key)
            return cared.fate if cared and cared.news else None

    def concerns(self) -> list[Concern]:
        """What she has a stake in whose last news was bad, the heaviest first."""
        with self._lock:
            held = [(key, cared.last, self._turn - cared.told) for key, cared in self._things.items()
                    if cared.news and cared.last < 0.0]
        found = [Concern(key, self.stake(key), how, since) for key, how, since in held]
        return sorted((c for c in found if c.stake > 0.0), key=lambda c: (-c.weight, c.thing))

    def stakes(self) -> dict[str, float]:
        """Everything whose fate is worth something to her, with what it is worth."""
        with self._lock:
            given_to = [key for key, cared in self._things.items() if cared.given]
        keys = dict.fromkeys(given_to)
        for key in given_to:
            keys.update(dict.fromkeys(other for other, _reach in self.branches(key)))
        found = {key: self.stake(key) for key in keys}
        return {key: stake for key, stake in found.items() if stake > 0.0}

    def status(self) -> dict[str, Any]:
        cared = self.cared_for()
        with self._lock:
            turns_given = self._totals()[0]
        return {
            "cares_about": [{"thing": key, "share": round(self.share(key), 4)} for key in cared],
            "concerns": [
                {"thing": c.thing, "stake": round(c.stake, 4), "how": round(c.how, 3), "turns_since": c.turns_since}
                for c in self.concerns()
            ],
            "turns_given": turns_given,
        }
