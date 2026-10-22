"""What she wants, formed from what the things in her life have been worth to her.

Bryan, 13 October: she has to direct herself. "I want to do XYZ, I am going to
figure out how to do XYZ", from her own inner pull, and choosing among options
she gives herself. Things have to matter to her, and she has to be able to
know what they are.

Measured the same day, on her last 500 self-started choices (10 to 13 October):
she chose 11 distinct things, one of them, "Running a self-integrity scan", 379
times. Every option on every slate came from a budget draining at a fixed rate
or from a loop with a fixed menu, and the topic of a curiosity goal was drawn
with `random.choice`. The valuation in this package was registered and had
been asked to appraise nothing, because nothing in her life was handed to it.

So a want had no object. This is the record that gives it one.

    thing      whatever a turn was about that has a name: a person, a subject,
               something she made. Things she cannot go to (herself, the
               source that won her attention, how an act came out) are kept
               too, because they share the credit for a turn
    liking     the thing's own share of what its turns were worth to her. A
               turn's worth is read at its end against what she expected
               (core/affect/what_it_was_worth.py), signed and ranked among
               her recent turns, so it lies in [-1, 1]. The things present
               predict it together and divide the error between them by how
               unsure she is of each, which is the Kalman form of the
               Rescorla-Wagner rule (Dayan and Kakade 2000; Gershman 2015).
               A new thing beside a familiar one takes most of a surprise,
               and a thing that only ever arrives with something she already
               likes earns nothing of its own
    unsure     the variance of that share. It starts at the variance of a
               number that could be anywhere in [-1, 1], falls with contact,
               and stops falling at what the last 256 contacts would leave,
               the window every other reading of her history uses, so an old
               association keeps learning
    known      whether she can say she likes it or dislikes it: its turns were
               worth more than she expected more often, or less often, than
               her turns in general, by a two-sided sign test at the
               conventional level. Until then the answer is that she has not
               had enough of it to say
    ways       how she has gone to it, each with what going that way was
               worth. A way is an act she has actually taken, so the options
               she gives herself are her own
    open       what she started with it and did not finish, in the words she
               started it in

A want is read off the record each time it is asked for. Every thing she can
go to is appraised by `core/conation/engine.py`: what it has come to be worth
pulls, what one more contact would still teach her about it pulls (expected
information gain, from `unsure`), and having just had it holds the pull down
until it recovers. Whatever comes out above nothing is a want, with the thing,
the way she would go, and the measurements behind it.

What a turn was about is read from its words as counts and never as meaning.
A term is one of a turn's things when two measurements agree. The exchange
uses it more than her exchanges have before, by Dunning's log-likelihood ratio
(1993), the standard measure of which words are key to a text against a
reference, corrected for the number of terms the exchange offers. And it is
dwelt on where it comes up: it is found in fewer than half the exchanges its
count would reach if it were spread by chance, which is a residual inverse
document frequency above one bit (Church and Gale 1995). The second is what
tells a subject from a word that is merely frequent in one kind of talk.
Measured on her 351 stored exchanges on 13 October, "limpet" and "scar" read
2.4 and 2.6 bits and "its", "home" and "watch" 0.6, 0.5 and 0.3. Her
reference is her own exchanges, the stored ones included.

Nothing here reads what a sentence says. Being told she loves something adds
a thing she has met once and knows nothing about.
"""
from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.conation.state import Incentive
from core.conation.what_her_talk_is_about import WhatHerTalkIsAbout, terms_of
from core.conation.what_she_cares_about import WhatSheCaresAbout
from core.runtime.atomic_writer import atomic_write_text_behind
from core.runtime.errors import record_degradation
from core.runtime.lockdep import checked_lock
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.Conation.Wants")

__all__ = [
    "FIND_OUT",
    "REACHABLE_KINDS",
    "read_turn",
    "Thing",
    "Want",
    "Way",
    "WhatSheWants",
    "get_what_she_wants",
    "liking_of",
    "terms_of",
    "reset_for_test",
]

#: How many contacts a reading of her own history is taken over.
_WINDOW = 256

#: The chance of calling a like or a dislike that is not there; the level
#: core/agency/what_she_is_like.py reads her choices at.
_WRONG = 0.05

#: The variance of a share that could be anywhere in [-1, 1].
_UNKNOWN = 1.0 / 3.0

#: The kinds of thing a turn names that she can go to herself. The rest of
#: what a turn was about (herself, the source that won her attention, an act
#: and how it came out) shares the credit and is nowhere to go.
REACHABLE_KINDS: frozenset[str] = frozenset({"topic", "person", "game"})

#: How many of the things a thing has come with are kept for it.
MAX_COMPANY = 16

#: The way any named thing affords before she has gone to it by another:
#: finding out about it.
FIND_OUT = "find_out"

MAX_THINGS = 4096
MAX_OPEN = 8
SCHEMA = 2



def liking_of(reading: Any) -> float | None:
    """A turn's worth as liking at contact, in [-1, 1], or nothing to read.

    `size` is where the turn's worth sits among her recent turns, so the
    scale is her own and no channel's units reach it.
    """
    if reading is None or not getattr(reading, "measured", False):
        return None
    try:
        worth = float(getattr(reading, "worth", 0.0) or 0.0)
        size = float(getattr(reading, "size", 0.0) or 0.0)
    # not a failure: a reading that is not numbers is no reading.
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(worth) and math.isfinite(size)) or worth == 0.0:
        return 0.0
    return math.copysign(max(0.0, min(1.0, size)), worth)


def _two_sided(above: int, of: int, chance: float) -> float:
    """The chance of a count at least this far from chance, either way."""
    if of <= 0 or not 0.0 < chance < 1.0:
        return 1.0

    def mass(k: int) -> float:
        return math.exp(
            math.lgamma(of + 1) - math.lgamma(k + 1) - math.lgamma(of - k + 1)
            + k * math.log(chance) + (of - k) * math.log(1.0 - chance)
        )

    below = sum(mass(k) for k in range(0, above + 1))
    over = sum(mass(k) for k in range(above, of + 1))
    return min(1.0, 2.0 * min(below, over))


@dataclass
class Way:
    """One way she has gone to a thing, and what going that way was worth."""

    times: int = 0
    liking: float = 0.0

    def went(self, liking: float) -> None:
        self.times += 1
        self.liking += (liking - self.liking) / min(self.times, _WINDOW)


@dataclass
class Thing:
    key: str
    label: str = ""
    reachable: bool = False
    contacts: int = 0
    good: int = 0
    liking: float = 0.0
    unsure: float = _UNKNOWN
    channels: dict[str, float] = field(default_factory=dict)
    first: int = 0
    last: int = 0
    ways: dict[str, Way] = field(default_factory=dict)
    open: dict[str, str] = field(default_factory=dict)
    #: How much of an exchange's subject it was, on average over the exchanges
    #: that named it: its share of what the exchange dwelt on. Zero for a
    #: thing no exchange named.
    stood_out: float = 0.0
    named: int = 0
    #: What the turns were worth while something she had started with it was
    #: still under way: the doing, apart from how it ended.
    during: Way = field(default_factory=Way)
    #: What she is in the middle of with it, by name, and the turn it began:
    #: a conversation, a game being played, an act of hers in progress.
    under_way: dict[str, int] = field(default_factory=dict)
    #: How to get to it, where it has a place: a game's address.
    where: str = ""
    #: The things it has come with, and in how many turns.
    came_with: dict[str, int] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return self.key.split(":", 1)[0] if ":" in self.key else self.key

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "reachable": self.reachable,
            "contacts": self.contacts,
            "good": self.good,
            "liking": self.liking,
            "unsure": self.unsure,
            "channels": dict(self.channels),
            "first": self.first,
            "last": self.last,
            "ways": {name: [way.times, way.liking] for name, way in self.ways.items()},
            "open": dict(self.open),
            "stood_out": self.stood_out,
            "named": self.named,
            "during": [self.during.times, self.during.liking],
            "under_way": dict(self.under_way),
            "where": self.where,
            "came_with": dict(self.came_with),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Thing:
        thing = cls(key=str(data["key"]))
        thing.label = str(data.get("label", ""))
        thing.reachable = bool(data.get("reachable", False))
        thing.contacts = max(0, int(data.get("contacts", 0)))
        thing.good = max(0, min(thing.contacts, int(data.get("good", 0))))
        thing.liking = max(-1.0, min(1.0, float(data.get("liking", 0.0))))
        thing.unsure = max(0.0, min(_UNKNOWN, float(data.get("unsure", _UNKNOWN))))
        thing.channels = {str(k): float(v) for k, v in dict(data.get("channels", {})).items()}
        thing.first = int(data.get("first", 0))
        thing.last = int(data.get("last", 0))
        for name, (times, liking) in dict(data.get("ways", {})).items():
            thing.ways[str(name)] = Way(times=int(times), liking=float(liking))
        thing.open = {str(k): str(v) for k, v in dict(data.get("open", {})).items()}
        thing.stood_out = max(0.0, min(1.0, float(data.get("stood_out", 0.0))))
        thing.named = max(0, int(data.get("named", 0)))
        times, liking = data.get("during", [0, 0.0])
        thing.during = Way(times=int(times), liking=float(liking))
        thing.under_way = {str(k): int(v) for k, v in dict(data.get("under_way", {})).items()}
        thing.where = str(data.get("where", ""))
        thing.came_with = {str(k): int(v) for k, v in dict(data.get("came_with", {})).items()}
        return thing


@dataclass(frozen=True)
class Want:
    """One thing she wants now, how she would go to it, and why."""

    thing: str
    label: str
    way: str
    #: "finish" when she left something undone with it, "more" when what it
    #: has been worth is the larger pull, "find out" when what she does not
    #: yet know about it is.
    kind: str
    wanting: float
    utility: float
    #: How much of her attention and planning it should claim, in [0, 1].
    priority: float
    origin: str
    why: str
    liking: float
    contacts: int
    #: True when she likes it, False when she dislikes it, None while she has
    #: not had enough of it to say.
    known: bool | None
    unfinished: str = ""
    #: Where the thing is, when it has a place she can go back to.
    where: str = ""
    #: The episode she pictures: who or what it would be with, what it would
    #: be worth and in which kinds of good, and how many turns it is drawn from.
    episode: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "thing": self.thing,
            "label": self.label,
            "way": self.way,
            "kind": self.kind,
            "wanting": round(self.wanting, 6),
            "utility": round(self.utility, 6),
            "priority": round(self.priority, 6),
            "origin": self.origin,
            "why": self.why,
            "liking": round(self.liking, 6),
            "contacts": self.contacts,
            "known": self.known,
            "unfinished": self.unfinished,
            "where": self.where,
            "episode": dict(self.episode),
        }


class _WantReadouts:
    """Read the retained wants and their public summary under the record lock."""

    def standing(self) -> list[Want]:
        """What she wanted the last time it was asked, without asking again.

        Asking appraises every thing, and an appraisal is itself a look at
        it. A readout takes the last answer.
        """
        with self._lock:
            return list(self._standing)

    def status(self) -> dict[str, Any]:
        with self._lock:
            reachable = sum(1 for thing in self._things.values() if thing.reachable)
            return {
                "things": len(self._things),
                "reachable": reachable,
                "turns": self._turns,
                "noise": round(self._noise, 6),
                "likes": [thing.key for thing in self.likes()[:8]],
                "dislikes": [thing.key for thing in self.dislikes()[:8]],
                "wants": [want.as_dict() for want in self._standing[:8]],
                "care": self.care.status(),
            }



class WhatSheWants(_WantReadouts):
    """The things she has met, what each has been worth, and what follows."""

    def __init__(self, path: str | Path | None = None, *, engine: Any = None) -> None:
        self._lock = checked_lock("conation.what_she_wants", reentrant=True)
        self._things: dict[str, Thing] = {}
        self._turns = 0
        self._good_turns = 0
        #: Running mean of her squared prediction errors: how far a turn's
        #: liking usually lands from what the things in it predicted.
        self._noise = _UNKNOWN
        self._read = 0
        #: The share of her recent turns that were worth more than she
        #: expected. One half is a life going as expected.
        self._going = 0.5
        #: Where she went since the last turn was read, and how. The turn that
        #: is read next is the one that going is part of.
        self._went: list[tuple[str, str]] = []
        #: What her exchanges have said so far, and so what a new one is about.
        #: See core/conation/what_her_talk_is_about.py.
        self.talk = WhatHerTalkIsAbout(changed=lambda: _save(self, _held(self)))
        #: What she gives her own turns to, and whose fate reaches her.
        #: See core/conation/what_she_cares_about.py.
        self.care = _care_of(self)
        #: What she has decided to do when she next can, if anything: one of
        #: her wants, as it stood when she decided.
        self._decided: dict[str, Any] | None = None
        #: What she pictured a going would be worth, by thing, until the turn
        #: it belongs to is read; and how far her pictures have been from what
        #: the turns were worth, as a count and a running mean of the miss.
        self._forecast: dict[str, float] = {}
        self._forecasts = 0
        self._miss = 0.0
        #: Each time a view of hers formed, lapsed or turned: a liking came to
        #: sit more than two standard errors from indifference, fell back
        #: within them, or came to be held the other way. The last 256.
        self._minds: list[dict[str, Any]] = []
        self._standing: list[Want] = []
        self._engine = engine
        self._path = Path(path) if path is not None else None
        _load(self)

    # ── where it is kept ─────────────────────────────────────────────────


    # ── what a turn was about ────────────────────────────────────────────

    # ── a turn ───────────────────────────────────────────────────────────

    def met(
        self,
        present: Iterable[str],
        liking: float | None,
        *,
        reachable: Iterable[str] = (),
        labels: Mapping[str, str] | None = None,
        payoff: Mapping[str, float] | None = None,
        stood_out: Mapping[str, float] | None = None,
        fated: Iterable[str] = (),
    ) -> dict[str, Any]:
        """One turn's contact with everything it was about.

        `reachable` names the present things she could go to herself. A thing
        she went to since the last turn was read is part of this one, with the
        way she went (see `went`). `fated` names the things she learned the
        fate of this turn: what the turn was worth is then partly the news,
        and the news is not what the thing itself is worth to her, so the turn
        teaches nothing about them.
        """
        if liking is None:
            return {"read": False}
        liking = max(-1.0, min(1.0, float(liking)))
        labels = dict(labels or {})
        can_reach = set(reachable)
        keys = list(dict.fromkeys(str(key) for key in present if str(key).strip()))
        with self._lock:
            went, self._went = dict(self._went), []
            for key in went:
                if key not in keys:
                    keys.append(key)
                can_reach.add(key)
            # Whatever she has under way is part of this turn whether or not
            # the turn named it: the doing is lived turn by turn, and a thing
            # credited only when it ends is worth nothing until it is over.
            doing = [key for key, thing in self._things.items() if thing.open or thing.under_way]
            for key in doing:
                if key not in keys:
                    keys.append(key)
            if not keys:
                return {"read": False}
            self._turns += 1
            self._read += 1
            self._good_turns += liking > 0.0
            self._going += ((1.0 if liking > 0.0 else 0.0) - self._going) / min(self._read, _WINDOW)
            everything = [_touch(self, key, labels.get(key, ""), key in can_reach) for key in keys]
            news_of = {str(key) for key in fated}
            things = [thing for thing in everything if thing.key not in news_of]
            for thing in everything:
                if thing.key in news_of:
                    thing.last = self._turns
            predicted = max(-1.0, min(1.0, sum(thing.liking for thing in things)))
            error = liking - predicted
            total = sum(thing.unsure for thing in things) + self._noise
            floor = self._noise / _WINDOW
            shares: dict[str, float] = {}
            for thing in things:
                gain = thing.unsure / total
                shares[thing.key] = gain
                held_before = _way_held(thing)
                thing.liking = max(-1.0, min(1.0, thing.liking + gain * error))
                thing.unsure = max(floor, thing.unsure * (1.0 - gain))
                held_now = _way_held(thing)
                if held_now != held_before:
                    # Turned: held now the other way from the last way she held
                    # it. A reversal passes through indifference on the way.
                    last = next(
                        (row["to"] for row in reversed(self._minds) if row["thing"] == thing.key and row["to"]),
                        0,
                    )
                    self._minds.append({
                        "thing": thing.key, "label": thing.label, "turn": self._turns,
                        "from": held_before, "to": held_now, "turned": last * held_now < 0,
                    })
                    del self._minds[:-_WINDOW]
                thing.contacts += 1
                thing.good += liking > 0.0
                thing.last = self._turns
                rate = 1.0 / min(thing.contacts, _WINDOW)
                for channel, paid in (payoff or {}).items():
                    try:
                        paid = float(paid)
                    # not a failure: a payoff that is not a number is one
                    # channel that says nothing about this thing this turn.
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(paid):
                        held = thing.channels.get(channel, 0.0)
                        thing.channels[channel] = held + (paid - held) * rate
                if went.get(thing.key):
                    thing.ways.setdefault(went[thing.key], Way()).went(liking)
                    pictured = self._forecast.pop(thing.key, None)
                    if pictured is not None:
                        # What she pictured against what the turn was worth.
                        self._forecasts += 1
                        self._miss += (abs(liking - pictured) - self._miss) / min(self._forecasts, _WINDOW)
                if thing.key in doing:
                    thing.during.went(liking)
                if thing.reachable:
                    for other in things:
                        if other.reachable and other.key != thing.key:
                            thing.came_with[other.key] = thing.came_with.get(other.key, 0) + 1
                    if len(thing.came_with) > MAX_COMPANY:
                        thing.came_with = dict(
                            sorted(thing.came_with.items(), key=lambda kv: (-kv[1], kv[0]))[:MAX_COMPANY]
                        )
                share = (stood_out or {}).get(thing.key)
                if share is not None:
                    thing.named += 1
                    thing.stood_out += (max(0.0, min(1.0, float(share))) - thing.stood_out) / min(thing.named, _WINDOW)
            self._noise += (error * error - self._noise) / min(self._read, _WINDOW)
            self._noise = max(1e-9, self._noise)
            held = _held(self)
        _save(self, held)
        _teach(self, things, liking, error, shares)
        return {"read": True, "predicted": predicted, "error": error, "shares": shares}


    # ── where she went, and what she started and did not finish ─────────

    def went(self, key: str, way: str, *, label: str = "", where: str = "") -> None:
        """She went to a thing herself, this way. Credited when the turn is read.

        What she pictured the going would be worth is kept beside it, and the
        turn's reading is set against it.
        """
        if not key or not way:
            return
        pictured = self.imagine(key, way=way)
        with self._lock:
            thing = _touch(self, key, label, True)
            if where:
                thing.where = where[:500]
            self._went.append((key, way))
            if pictured is not None:
                self._forecast[key] = pictured["worth"]

    def started(self, key: str, item: str, *, label: str = "", where: str = "") -> None:
        """She is in the middle of something with a thing from now on.

        A conversation, a game being played, an act in progress. Every turn
        read until it stops is a turn of it, so the doing is worth what it is
        worth as it goes.
        """
        if not key or not item:
            return
        with self._lock:
            thing = _touch(self, key, label, True)
            if where:
                thing.where = where[:500]
            thing.under_way.setdefault(item, self._turns)
            held = _held(self)
        _save(self, held)

    def stopped(self, item: str, key: str | None = None) -> int:
        """What was under way by that name has stopped, with one thing or with all."""
        closed = 0
        with self._lock:
            for thing in self._things.values():
                if (key is None or thing.key == key) and thing.under_way.pop(item, None) is not None:
                    closed += 1
            held = _held(self) if closed else None
        if held is not None:
            _save(self, held)
        return closed

    # ── what she pictures ────────────────────────────────────────────────

    def trust_in_her_pictures(self) -> float:
        """How far a picture of hers is to be believed, from the ones she has checked.

        One when her pictures have matched what the turns were worth, falling
        with the mean miss over a span of two, the widest a miss can be. With
        none checked it is one half: a picture is half believed until it has
        been tried (people's forecasts of how a thing will feel run wide of
        how it does; Wilson and Gilbert 2005).
        """
        with self._lock:
            checked = min(self._forecasts, _WINDOW)
            return (checked * (1.0 - self._miss / 2.0) + 1.0) / (checked + 2.0)

    def imagine(self, key: str, *, way: str = "") -> dict[str, Any] | None:
        """The episode she pictures of going to a thing: with what, worth what, in which kinds of good.

        A person pictures a morning's walk from walks already had, put
        together anew: the place, how they would go, who would be there
        (Schacter and Addis 2007). Hers is put together from her record. What
        the thing has been worth, what going that way has been worth with it
        or with things of its kind, and what each thing it has come with has
        been worth, each weighed by how many turns stand behind it. The
        company pictured is whichever makes the episode worth most, and the
        kinds of good are those of the thing and that company.
        """
        with self._lock:
            thing = self._things.get(key)
            if thing is None:
                return None
            parts: list[tuple[float, float]] = []
            if thing.contacts:
                parts.append((thing.liking, float(thing.contacts)))
            if thing.during.times:
                parts.append((thing.during.liking, float(thing.during.times)))
            taken = thing.ways.get(way) if way else None
            if taken is not None and taken.times:
                parts.append((taken.liking, float(taken.times)))
            elif way:
                elsewhere = [
                    other.ways[way]
                    for other in self._things.values()
                    if other.kind == thing.kind and other.key != key and way in other.ways
                ]
                if elsewhere:
                    parts.append((
                        sum(w.liking for w in elsewhere) / len(elsewhere),
                        float(sum(w.times for w in elsewhere)),
                    ))

            def worth_with(company: Thing | None, turns: int) -> tuple[float, int]:
                together = list(parts)
                if company is not None and company.contacts:
                    together.append((company.liking, float(turns)))
                weight = sum(n / (n + 1.0) for _liking, n in together)
                if weight <= 0.0:
                    return 0.0, 0
                worth = sum(liking * n / (n + 1.0) for liking, n in together) / weight
                return worth, int(sum(n for _liking, n in together))

            best_worth, drawn_from = worth_with(None, 0)
            best_company: Thing | None = None
            for other_key, turns in sorted(thing.came_with.items()):
                other = self._things.get(other_key)
                if other is None:
                    continue
                worth, drawn = worth_with(other, turns)
                if worth > best_worth:
                    best_worth, drawn_from, best_company = worth, drawn, other
            if drawn_from <= 0:
                return None
            kinds: dict[str, float] = dict(thing.channels)
            if best_company is not None:
                for channel, paid in best_company.channels.items():
                    kinds[channel] = (kinds.get(channel, 0.0) + paid) / 2.0 if channel in kinds else paid
            return {
                "thing": key,
                "way": way,
                "with": best_company.key if best_company is not None else "",
                "worth": max(-1.0, min(1.0, best_worth)),
                "kinds": {channel: round(paid, 4) for channel, paid in sorted(kinds.items()) if abs(paid) > 1e-9},
                "drawn_from": drawn_from,
            }

    def began(self, key: str, item: str, words: str, *, label: str = "") -> None:
        """She started something with a thing; it stays open until she ends it."""
        if not key or not item or not words.strip():
            return
        with self._lock:
            thing = _touch(self, key, label, True)
            if item not in thing.open and len(thing.open) >= MAX_OPEN:
                thing.open.pop(next(iter(thing.open)))
            thing.open[item] = " ".join(words.split())[:500]
            held = _held(self)
        _save(self, held)

    def ended(self, key: str, item: str, *, finished: bool) -> None:
        """What she started with a thing came to an end, done or given up."""
        with self._lock:
            thing = self._things.get(key)
            if thing is None or item not in thing.open:
                return
            thing.open.pop(item, None)
            held = _held(self)
        _save(self, held)
        if finished:
            self.care.fared(key, 1.0)
        else:
            self.failed(key)

    def failed(self, key: str) -> None:
        """She went for a thing and did not get it.

        Failing at something she wants builds up, by how much she wanted it
        (core/conation/dynamics.py), and past the point that file sets she
        stops wanting it for now. It is also how the thing fared.
        """
        self.care.fared(key, -1.0)
        try:
            engine = _conation(self)
            record = engine.salience._records.get(key)
            wanting = float(getattr(record, "last_wanting", 0.0) or 0.0)
            engine.dynamics.observe_attempt(key, wanting=wanting, succeeded=False)
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "what_she_wants", exc, severity="debug",
                action="an attempt that failed left no frustration behind",
            )

    # ── what she can say about a thing ───────────────────────────────────

    def thing(self, key: str) -> Thing | None:
        return self._things.get(key)

    def known(self, key: str) -> bool | None:
        """Whether she likes it, dislikes it, or has not had enough of it to say."""
        with self._lock:
            thing = self._things.get(key)
            if thing is None or thing.contacts <= 0 or self._turns <= 0:
                return None
            chance = self._good_turns / self._turns
            if _two_sided(thing.good, thing.contacts, chance) >= _WRONG:
                return None
            return thing.good / thing.contacts > chance

    def likes(self) -> list[Thing]:
        """What she likes, most liked first; reachable or not."""
        with self._lock:
            found = [thing for thing in self._things.values() if self.known(thing.key) is True]
        return sorted(found, key=lambda thing: (-thing.liking, thing.key))

    def dislikes(self) -> list[Thing]:
        with self._lock:
            found = [thing for thing in self._things.values() if self.known(thing.key) is False]
        return sorted(found, key=lambda thing: (thing.liking, thing.key))

    def pictured(self, key: str) -> float:
        """What a thing is pictured to be worth from what it has come with, in [0, 1].

        A person wants a walk before taking it: the morning is put together
        from walks already had and it pulls. Worth spreads the same way to a
        thing that earned none of its own, from the things it has kept company
        with (Wimmer and Shohamy 2012). Each companion counts by the share of
        this thing's turns it was in and by what she has found it worth.
        """
        episode = self.imagine(key)
        if episode is None:
            return 0.0
        return max(0.0, min(1.0, episode["worth"] * self.trust_in_her_pictures()))

    def something_is_off(self) -> bool:
        """Whether her recent turns have been worth less than she expected more often than chance.

        A person turns to look at themselves when something does not match
        what they expected of it (Duval and Wicklund 1972; Carver and Scheier
        1981), and otherwise their attention is on what they are doing. Her
        self-scans ran on a timer. This is the mismatch they wait for: a
        one-sided sign test on her recent turns at the conventional level.
        """
        with self._lock:
            turns = min(self._read, _WINDOW)
            good = int(round(self._going * turns))
        if turns <= 0:
            return False
        below = sum(
            math.exp(math.lgamma(turns + 1) - math.lgamma(k + 1) - math.lgamma(turns - k + 1) - turns * math.log(2.0))
            for k in range(0, good + 1)
        )
        return below < _WRONG

    def how_far_off(self) -> float:
        """How far below what she expected her recent turns have run, in [0, 1].

        Nothing while half or more of them were worth more than she expected,
        and one when none were. The graded form of `something_is_off`, for
        what should rise with the mismatch and rest without one.
        """
        with self._lock:
            if self._read <= 0:
                return 0.0
            return max(0.0, min(1.0, (0.5 - self._going) * 2.0))

    def vigor(self) -> float:
        """How readily she acts, against a life going as she expected: 1.0.

        The time worth waiting before an act falls as the square root of the
        average rate at which things have been paying, because waiting costs
        what that time would have brought (Niv, Daw, Joel and Dayan 2007). The
        rate is the share of her recent turns worth more than she expected,
        and one half is the life she expected.
        """
        with self._lock:
            share = (self._going * min(self._read, _WINDOW) + 1.0) / (min(self._read, _WINDOW) + 2.0)
        return math.sqrt(share / 0.5)

    def expected_worth(self, goal: str, metadata: Mapping[str, Any] | None = None) -> float | None:
        """What her record says acting on an option would be worth, in [0, 1].

        One half is a turn worth exactly what she expected. For one of her
        wants it is her share of what turns with the thing were worth. For
        anything else it is what turns in which she did that act were worth,
        the act named as core/affect/feelings_about.py names it. Nothing when
        she has no turn with it to go on.
        """
        want = (metadata or {}).get("want")
        key = ""
        if isinstance(want, dict) and want.get("thing"):
            key = str(want["thing"])
        else:
            words = str(goal or "").split()
            key = f"act:{words[0].lower()}" if words else ""
        with self._lock:
            thing = self._things.get(key)
            if thing is None or thing.contacts <= 0:
                return None
            return (thing.liking + 1.0) / 2.0

    def regard_for_text(self, text: str) -> float:
        """Her strongest stated like or dislike among the things a text names.

        Zero when it names nothing she can say she likes or dislikes. A name
        is a thing's own label, whole, so this reads which of her things an
        option is about and nothing about how the option is worded.
        """
        lowered = " ".join(str(text or "").lower().split())
        strongest = 0.0
        with self._lock:
            things = [thing for thing in self._things.values() if len(thing.label) > 2]
        for thing in things:
            if thing.label.lower() not in lowered or self.known(thing.key) is None:
                continue
            if abs(thing.liking) > abs(strongest):
                strongest = thing.liking
        return strongest

    # ── what she wants ───────────────────────────────────────────────────


    def wants(self, *, ways: Iterable[str] = (), limit: int | None = None) -> list[Want]:
        """What she wants now, most wanted first."""
        offered = tuple(ways)
        with self._lock:
            candidates = [thing for thing in self._things.values() if thing.reachable and thing.label]
            noise = self._noise
        try:
            engine = _conation(self)
        except (ImportError, AttributeError) as exc:
            record_degradation(
                "what_she_wants", exc, severity="warning",
                action="no want could be priced; nothing is reported as wanted",
            )
            return []
        cap = int(getattr(engine, "MAX_TRACKED_STATES", len(candidates)) or len(candidates))
        candidates.sort(key=lambda t: (-(max(0.0, t.liking) + math.sqrt(t.unsure)), t.key))
        found: list[Want] = []
        for thing in candidates[:cap]:
            way = _way_for(self, thing, offered)
            # A person is reached by a way she has taken with them. There is
            # no looking somebody up to find out what they are to her.
            if thing.kind == "person" and way == FIND_OUT:
                continue
            gain = thing.unsure / (thing.unsure + noise)
            after = max(noise / _WINDOW, thing.unsure * (1.0 - gain))
            # What she has to go on. Curiosity needs a foothold: it rises with
            # how much of an exchange the thing was, and something mentioned
            # in passing leaves little to be curious about (Loewenstein 1994;
            # Silvia 2005). A thing no exchange named is one she went to, and
            # that is a whole foothold.
            # A stake in a thing is something to go on, however briefly it
            # was mentioned: she wants to find out about what comes with a
            # thing she cares about for that thing's sake.
            stake = self.care.stake(thing.key)
            foothold = max(thing.stood_out if thing.named else 1.0, stake)
            pictured = self.pictured(thing.key)
            state = engine.appraise(
                Incentive(
                    key=thing.key,
                    description=thing.label,
                    # Her own share of what its turns were worth, or what it
                    # is pictured to be worth from its company, whichever is
                    # more: she can want what she has only pictured.
                    cached_value=max(
                        max(0.0, thing.liking),
                        pictured,
                        # And what the doing was worth while it lasted: a
                        # thing worth doing is wanted whatever it came to.
                        max(0.0, thing.during.liking) if thing.during.times else 0.0,
                    ) if thing.contacts else None,
                    cue_salience=thing.stood_out,
                ),
                epistemic_affordance=foothold,
                prior_variance=[thing.unsure],
                posterior_variance=[after],
            )
            utility = float(engine.utility(state))
            known = self.known(thing.key)
            if utility <= 0.0 or known is False:
                continue
            if engine.dynamics.frustration(thing.key).should_disengage():
                continue
            origin = str(getattr(state.dominant_origin, "value", state.dominant_origin) or "")
            unfinished = next(iter(thing.open.values()), "")
            other_pull = state.magnitude_of(state.dominant_origin) if state.dominant_origin else 0.0
            if unfinished:
                kind = "finish"
            elif max(0.0, thing.liking) >= other_pull:
                kind = "more"
            else:
                kind = "find out"
            found.append(
                Want(
                    thing=thing.key,
                    label=thing.label,
                    way=way,
                    kind=kind,
                    wanting=float(state.wanting),
                    utility=utility,
                    priority=float(engine.attention_priority(state)),
                    origin=origin,
                    why=(
                        f"{thing.contacts} contacts, {thing.good} of them worth more than she expected; "
                        f"her share of their worth {thing.liking:+.3f}, variance {thing.unsure:.4f}; "
                        f"pictured from its company {pictured:.3f}; "
                        f"the doing was worth {thing.during.liking:+.3f} over {thing.during.times} turns; "
                        f"her stake in how it fares {stake:.3f}; "
                        f"{state.why()}"
                    ),
                    liking=thing.liking,
                    contacts=thing.contacts,
                    known=known,
                    unfinished=unfinished,
                    where=thing.where,
                    episode=self.imagine(thing.key, way=way) or {},
                )
            )
        found.sort(key=lambda want: (-want.utility, want.thing))
        with self._lock:
            self._standing = list(found)
        return found if limit is None else found[: max(0, int(limit))]

    # ── what she has decided ─────────────────────────────────────────────

    def decide(self) -> dict[str, Any] | None:
        """What she most wants, settled for the next time she is free to act.

        A person who wants a walk tomorrow decides once and does not argue it
        again in the morning (Bratman 1987; Gollwitzer 1999). Her slate was
        rebuilt every cycle and nothing on it was ever settled, so a want that
        formed while somebody was with her was gone by the time she was free.
        One thing is held at a time, and what is already held stays held.
        """
        with self._lock:
            if self._decided is not None:
                return dict(self._decided)
        wanted = self.wants(limit=1)
        if not wanted:
            return None
        with self._lock:
            self._decided = {**wanted[0].as_dict(), "decided_turn": self._turns}
            held = _held(self)
            decided = dict(self._decided)
        _save(self, held)
        self.care.gave(str(decided.get("thing") or ""))
        return decided

    def decided(self) -> dict[str, Any] | None:
        """What she is holding to do, while she still wants it.

        Let go when she has come to dislike the thing or has given up on it;
        a decision is kept against other wants, and it is dropped when the
        want under it is gone.
        """
        with self._lock:
            held = dict(self._decided) if self._decided is not None else None
        if held is None:
            return None
        key = str(held.get("thing") or "")
        gone = self.known(key) is False
        try:
            gone = gone or _conation(self).dynamics.frustration(key).should_disengage()
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "what_she_wants", exc, severity="debug",
                action="kept what she had decided without asking whether she has given up on it",
            )
        if gone:
            self.done_with(key)
            return None
        return held

    def done_with(self, key: str) -> None:
        """She did what she had decided, or let it go."""
        with self._lock:
            if self._decided is None or str(self._decided.get("thing")) != key:
                return
            self._decided = None
            held = _held(self)
        _save(self, held)



def read_turn(
    present: Iterable[str],
    reading: Any,
    record: WhatSheWants | None = None,
    *,
    said: Iterable[str] = (),
) -> dict[str, Any]:
    """Fold the turn that just ended into what she has met.

    `present` is what the turn was about, as core/affect/feelings_about.py
    names it, and `reading` is what the turn was worth. `said` is what was
    said in it, by anybody. The subjects are the terms the exchange dwells
    on, and the one topic label the turn arrived with is not used.
    """
    record = record or get_what_she_wants()
    keys = [str(key) for key in present if str(key).strip()]
    words = [str(text) for text in said if str(text or "").strip()]
    # The topic label a turn arrives with is one word for a whole conversation
    # and stays on her state after it: live, 13 October, it read "times"
    # through a morning about tide pools and on the ticks after it. Subjects
    # come from what was said, and a turn in which nothing was said has none.
    keys = [key for key in keys if not key.startswith("topic:")]
    stood_out: dict[str, float] = {}
    if words:
        about = record.talk.about(*words)
        keys.extend(f"topic:{term}" for term in about)
        stood_out = {f"topic:{term}": share for term, share in record.talk.shares().items()}
        # The first thing said is the other party's. A subject their message
        # named was brought to her; one only her answer names she raised.
        theirs = set(terms_of(words[0])) if len(words) > 1 else set(about)
        people = [key for key in keys if key.startswith("person:")]
        record.care.exchange(
            [*(f"topic:{term}" for term in about), *people],
            theirs=[*(f"topic:{term}" for term in about if term in theirs), *people],
        )
    reachable = [key for key in keys if ":" in key and key.split(":", 1)[0] in REACHABLE_KINDS]
    labels = {key: key.split(":", 1)[1].strip() for key in reachable}
    return record.met(
        keys,
        liking_of(reading),
        reachable=reachable,
        labels=labels,
        payoff=getattr(reading, "payoff", None),
        stood_out=stood_out,
        fated=record.care.felt(),
    )


_RECORD: WhatSheWants | None = None
_RECORD_LOCK = checked_lock("conation.what_she_wants.singleton")


def _file(self) -> Path:
    if self._path is None:
        self._path = state_root() / "data" / "conation" / "what_she_wants.json"
    return self._path


def _load(self) -> None:
    try:
        stored = json.loads(_file(self).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return
    except (OSError, ValueError) as exc:
        record_degradation(
            "what_she_wants", exc, severity="warning",
            action="started with nothing met; the stored record did not read back",
        )
        return
    try:
        if int(stored.get("schema", 0)) != SCHEMA:
            return
        self._turns = max(0, int(stored.get("turns", 0)))
        self._good_turns = max(0, min(self._turns, int(stored.get("good_turns", 0))))
        self._noise = max(1e-9, float(stored.get("noise", _UNKNOWN)))
        self._read = max(0, int(stored.get("read", 0)))
        self._going = max(0.0, min(1.0, float(stored.get("going", 0.5))))
        for row in stored.get("things", []):
            thing = Thing.from_dict(row)
            # What a process was in the middle of ended with it. Live, 13
            # October, two games were still "under way" from runs long gone,
            # so she was busy in every answer and every turn was theirs.
            thing.under_way.clear()
            self._things[thing.key] = thing
        self.talk = WhatHerTalkIsAbout(stored, changed=lambda: _save(self, _held(self)))
        self.care = _care_of(self, stored.get("care"))
        decided = stored.get("decided")
        self._decided = dict(decided) if isinstance(decided, dict) and decided.get("thing") else None
        self._forecasts = max(0, int(stored.get("forecasts", 0)))
        self._miss = max(0.0, min(2.0, float(stored.get("miss", 0.0))))
        self._minds = [dict(row) for row in stored.get("minds", []) if isinstance(row, dict)][-_WINDOW:]
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        self._things.clear()
        self._turns = self._good_turns = self._read = 0
        self._noise = _UNKNOWN
        record_degradation(
            "what_she_wants", exc, severity="warning",
            action="started with nothing met; the stored record had another shape",
        )


def _way_held(thing: Any) -> int:
    """Which way she holds a thing: +1 or -1 beyond two standard errors of indifference, else 0."""
    error = math.sqrt(thing.unsure) if thing.unsure > 0.0 else math.inf
    if abs(thing.liking) <= 2.0 * error:
        return 0
    return 1 if thing.liking > 0.0 else -1


def _held(self) -> dict[str, Any]:
    """Everything kept, copied while the lock is held."""
    return {
        "schema": SCHEMA,
        "turns": self._turns,
        "good_turns": self._good_turns,
        "noise": self._noise,
        "read": self._read,
        "going": self._going,
        "saved_at": time.time(),
        "things": [thing.to_dict() for thing in self._things.values()],
        **self.talk.held(),
        "care": self.care.held(),
        "decided": self._decided,
        "forecasts": self._forecasts,
        "miss": self._miss,
        "minds": list(self._minds),
    }


def _care_of(self, stored: Any = None) -> WhatSheCaresAbout:
    """Her caring, reading each thing's company from this record."""

    def company(key: str) -> tuple[int, Mapping[str, int]]:
        thing = self._things.get(key)
        return (thing.contacts, thing.came_with) if thing else (0, {})

    return WhatSheCaresAbout(
        stored if isinstance(stored, Mapping) else None,
        company=company,
        changed=lambda: _save(self, _held(self)),
    )


def _save(self, payload: dict[str, Any]) -> None:
    """Written after the lock is let go: a write can wait on the disk."""
    try:
        path = _file(self)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Behind the loop: a turn ends inside the kernel's async tick.
        atomic_write_text_behind(path, json.dumps(payload), encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        record_degradation(
            "what_she_wants", exc, severity="warning",
            action="kept what she met this turn in memory only",
        )


def _conation(self) -> Any:
    if self._engine is None:
        from core.conation.engine import get_conation

        self._engine = get_conation()
    return self._engine


def _touch(self, key: str, label: str, reachable: bool) -> Thing:
    thing = self._things.get(key)
    if thing is None:
        if len(self._things) >= MAX_THINGS:
            oldest = min(self._things.values(), key=lambda t: (t.contacts, t.last))
            self._things.pop(oldest.key, None)
        thing = Thing(key=key, first=self._turns)
        self._things[key] = thing
    if label and not thing.label:
        thing.label = label[:160]
    thing.reachable = thing.reachable or reachable
    return thing


def _teach(self, things: list[Thing], liking: float, error: float, shares: Mapping[str, float]) -> None:
    """Fold the contact into the valuation that prices her wants.

    Having had a thing holds the want for it down by as much as the
    contact paid, and what she expected against what she got is the error
    her curiosity about it learns from.
    """
    try:
        engine = _conation(self)
        for thing in things:
            if not thing.reachable:
                continue
            engine.learn(
                thing.key,
                experienced_liking=liking,
                realised_pull=max(0.0, thing.liking),
                completeness=max(0.0, liking),
                prediction_error=min(1.0, abs(error) * shares.get(thing.key, 0.0)),
            )
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        record_degradation(
            "what_she_wants", exc, severity="warning",
            action="kept the contact on her record; the valuation was not taught by it",
        )


def _way_for(self, thing: Thing, offered: Iterable[str]) -> str:
    """The way to go to it, from the ways she has gone to things.

    Each way's worth is its mean with the optimism of UCB1 (Auer,
    Cesa-Bianchi and Fischer 2002) over the times she has gone to this
    thing, so a way never tried on it comes first. Untried ways are
    ordered by what they have been worth on other things of its kind.
    """
    ways = set(thing.ways) | set(offered)
    elsewhere: dict[str, list[float]] = {}
    for other in self._things.values():
        if other.kind != thing.kind or other.key == thing.key:
            continue
        for name, way in other.ways.items():
            elsewhere.setdefault(name, []).append(way.liking)
            ways.add(name)
    if not ways:
        return FIND_OUT
    went = sum(way.times for way in thing.ways.values())

    def score(name: str) -> tuple[float, float, str]:
        pooled = elsewhere.get(name, [])
        prior = sum(pooled) / len(pooled) if pooled else 0.0
        tried = thing.ways.get(name)
        if tried is None or tried.times <= 0 or went <= 0:
            return (math.inf, prior, name)
        # Liking spans two units, so the bound does.
        bound = tried.liking + 2.0 * math.sqrt(2.0 * math.log(went) / tried.times)
        return (bound, prior, name)

    return max(sorted(ways), key=score)


def get_what_she_wants() -> WhatSheWants:
    global _RECORD
    if _RECORD is None:
        with _RECORD_LOCK:
            if _RECORD is None:
                _RECORD = WhatSheWants()
    return _RECORD


def reset_for_test(path: str | Path | None = None, *, engine: Any = None) -> WhatSheWants:
    global _RECORD
    with _RECORD_LOCK:
        _RECORD = WhatSheWants(path, engine=engine)
    return _RECORD
