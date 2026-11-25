"""What each area comes to: a situation read a part at a time, and the parts put back together to choose where to act.

Bryan, 24 November 2026: "given an area, here are the mathematical possibilities and positions that can exist, these
put me in this position, these put me in that position ... do that multiple times in a given area, and combine the
learnings from each and use it to inform a decision"; "if this area gives me 10 possibilities, 4 lead to a negative
outcome, 3 put me in a worse position than not, 3 put me in a better position ... stitch together the outcomes of other
areas and map out how this will play out, which paths are the best to take, which put me in a worse position or give the
opponent the option of a better position"; "if you know a few, you may be able to deduce what other paths forward may
look like".

The mathematics is the one Conway and Berlekamp worked out for games that fall apart into parts (combinatorial game
theory, and its use in the endgames of Go). Where the parts of a situation do not reach each other, the whole is the
sum of its parts, and each part can be read alone. Three numbers read a part, from her side: what it comes to where she
acts in it next, where the other side does, and where neither does. Half the difference between the first two is what
acting there first is worth, the part's urgency (its temperature); their middle is what it is worth whoever acts there
first (its mean). The side to act takes the most urgent part, the other side the next, and so on while acting first
still gains anything: put back together, that is what the whole comes to, and the order it plays out in. Each act is
weighed by what the whole comes to after it. An act in a part that is not the most urgent leaves the other side the more
urgent one, and an act after which its part is worth less than as it stood leaves her worse off than acting nowhere.

The same reading runs the other way. Each part keeps the other side's acts with what each would do to her
(``threats``: what she must answer or forestall); the parts the other side keeps acting in, and makes hotter, are where
it is going (``where_they_work``); what any change did to every part is read as the readings before and after it
(``what_it_changes``); and ``what_to_do`` names her act that leaves the whole best after their best reply anywhere, the
most good of her acts where one leaves her better than acting nowhere, and otherwise the least bad.

Where the rules are known, a part is also worked out as far as there is time (``what_can_happen``): not what the other
side will do but everything it can do, and what each comes to, so that her answer to each of their acts is ready before
they act, and a turn away from what she expected of them is one she has already worked out.

A part she has not looked at is read from the parts like it she has (``PartsSeen``), nearest first by what each looked
like, and says that it was not looked at and how far the parts it was read from disagree.

Nothing here knows a game. A part is whatever the caller says can be acted on apart: a cluster of pieces on a board
that reach each other and nothing else, one project among several, a front among fronts. Where parts do reach each
other the sum is a reading, not a proof, and the caller decides what it takes to be apart.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

__all__ = ["HERS", "THEIRS", "Choice", "PartsSeen", "Possibilities", "Reading", "Threat", "Weighed", "Whole",
           "put_together", "read_part", "threats", "weigh_acts", "what_can_happen", "what_it_changes", "what_to_do",
           "where_they_work"]

#: The two sides: hers, which every number here is read for, and the other.
HERS, THEIRS = 0, 1


@dataclass(frozen=True)
class Reading:
    """What one part comes to, from her side: where she acts in it next, where they do, and where neither does."""

    part: Hashable
    if_hers_first: float
    if_theirs_first: float
    as_it_stands: float
    #: Each of her acts in the part with what the part comes to after it, the other side's answers looked at; best
    #: first.
    acts: tuple[tuple[Any, float], ...] = ()
    #: Each of the other side's acts in the part with what the part comes to for her after it, her answers looked at;
    #: worst for her first. What they can do here, and what it would do to her.
    their_acts: tuple[tuple[Any, float], ...] = ()
    #: False where the part was not looked at, and was read from the parts like it.
    looked: bool = True
    #: Where it was read from parts like it, how far their urgencies spread (0 for a part looked at).
    spread: float = 0.0

    @property
    def urgency(self) -> float:
        """What acting in this part first is worth over the other side acting in it first, halved: its temperature."""
        return (self.if_hers_first - self.if_theirs_first) / 2.0

    @property
    def mean(self) -> float:
        return (self.if_hers_first + self.if_theirs_first) / 2.0

    def kinds_of_act(self, *, losing_at: float | None = None) -> dict[str, list[Any]]:
        """Her acts here as ones that leave the part better than as it stands, worse, or lost (at or below
        ``losing_at``, where the caller says what losing is)."""
        kinds: dict[str, list[Any]] = {"better": [], "worse": [], "losing": [], "no_change": []}
        for act, comes_to in self.acts:
            if losing_at is not None and comes_to <= losing_at:
                kinds["losing"].append(act)
            elif comes_to > self.as_it_stands:
                kinds["better"].append(act)
            elif comes_to < self.as_it_stands:
                kinds["worse"].append(act)
            else:
                kinds["no_change"].append(act)
        return kinds


def read_part(part: Hashable, state: Any, *, acts: Callable[[Any, int], Iterable[Any]],
              after: Callable[[Any, Any], Any], judge: Callable[[Any], float], plies: int,
              seconds: float | None = None) -> Reading:
    """Read one part: ``acts(state, side)`` the acts each side has in it, ``after(state, act)`` the state an act leaves,
    ``judge(state)`` what the part stands at for her. The side that acts first in the part must act there; after that,
    either side may leave the part as it is and act elsewhere, which is how a part is played inside a whole. Looked
    ``plies`` acts deep, or until ``seconds`` have gone, where a look not yet finished stands at what it has reached."""
    until = None if seconds is None else time.monotonic() + seconds

    def value(st: Any, side: int, left: int, low: float, high: float) -> float:
        """What the part comes to with ``side`` to act in it, within the bounds ``low`` and ``high`` that matter to the
        acts above (outside them, which value it is does not change any choice)."""
        here = judge(st)  # leaving the part as it is
        if left <= 0 or (until is not None and time.monotonic() > until):
            return here
        best = here
        for act in acts(st, side):
            if side == HERS:
                if best >= high:
                    break
                best = max(best, value(after(st, act), THEIRS, left - 1, max(low, best), high))
            else:
                if best <= low:
                    break
                best = min(best, value(after(st, act), HERS, left - 1, low, min(high, best)))
        return best

    def first(side: int) -> tuple[float, tuple[tuple[Any, float], ...]]:
        options = [(act, value(after(state, act), 1 - side, plies - 1, -math.inf, math.inf))
                   for act in acts(state, side)]
        if not options:
            return judge(state), ()
        options.sort(key=lambda option: option[1], reverse=(side == HERS))
        return options[0][1], tuple(options)

    stands = judge(state)
    hers, her_acts = first(HERS)
    theirs, their_acts = first(THEIRS)
    return Reading(part=part, if_hers_first=hers, if_theirs_first=theirs, as_it_stands=stands, acts=her_acts,
                   their_acts=their_acts)


@dataclass(frozen=True)
class Possibilities:
    """What a part can come to, worked out as far as there was time: not what the other side will do, but everything it
    can do, and what each comes to.

    For each of her acts: what the other side can force after it whatever she does (the worst for her over all their
    answers, her answers worked out too), what it comes to if they answer it as badly as they can, and how many of
    their answers leave her worse than the part stands now. For each act of theirs available now: her answer to it and
    what that comes to, ready before they act. Bryan, 24 November 2026: "i know what you CAN do and i know what would
    happen 5-10 steps ahead (or as many as she can calculate)", so that where they leave a plan she expected she has
    already worked out what to do."""

    part: Hashable
    #: How many acts deep every line was worked out.
    deep: int
    #: (her act, what they can force after it, what it comes to at best for her, their answers, answers that hurt her)
    hers: tuple[tuple[Any, float, float, int, int], ...]
    #: (their act, her answer or None where she leaves the part, what that comes to)
    counters: tuple[tuple[Any, Any, float], ...]

    def counter_to(self, act: Any) -> tuple[Any, float] | None:
        return next(((answer, comes_to) for theirs, answer, comes_to in self.counters if theirs == act), None)


def what_can_happen(part: Hashable, state: Any, *, acts: Callable[[Any, int], Iterable[Any]],
                    after: Callable[[Any, Any], Any], judge: Callable[[Any], float], plies_at_most: int,
                    seconds: float, name_of: Callable[[Any], Hashable] | None = None,
                    may_leave: bool = True) -> Possibilities:
    """Every act of each side in the part followed, a move deeper each pass, until ``plies_at_most`` or ``seconds``: the
    deepest pass finished is what is kept. Where ``may_leave``, either side may leave the part and act elsewhere after
    the first act, as in ``read_part``: a part of a whole; where not, the part is the whole world, and a side acts in it
    whenever it can. ``name_of`` gives a state its name where states do not name themselves, so that a state reached two
    ways is worked out once."""
    until = time.monotonic() + seconds
    name = name_of or (lambda st: st)

    class _OutOfTime(Exception):
        pass

    def value(st: Any, side: int, left: int, low: float, high: float, known: dict) -> float:
        if time.monotonic() > until:
            raise _OutOfTime
        key = (name(st), side, left)
        if key in known:
            return known[key]
        options = list(acts(st, side)) if left > 0 else []
        if not options or may_leave:
            best = judge(st)  # leaving the part as it is, or nothing left to do in it
        else:
            best = -math.inf if side == HERS else math.inf
        if options:
            for act in options:
                if side == HERS:
                    if best >= high:
                        break
                    best = max(best, value(after(st, act), THEIRS, left - 1, max(low, best), high, known))
                else:
                    if best <= low:
                        break
                    best = min(best, value(after(st, act), HERS, left - 1, low, min(high, best), known))
        if low < best < high:
            known[key] = best
        return best

    def one_pass(plies: int) -> Possibilities:
        known: dict = {}
        stands = judge(state)
        hers = []
        for act in acts(state, HERS):
            made = after(state, act)
            answers = [value(after(made, answer), HERS, plies - 2, -math.inf, math.inf, known)
                       for answer in acts(made, THEIRS)] if plies >= 2 else []
            outcomes = [judge(made), *answers] if may_leave or not answers else answers
            hers.append((act, min(outcomes), max(outcomes), len(answers),
                         sum(1 for comes_to in answers if comes_to < stands)))
        hers.sort(key=lambda row: (row[1], row[2]), reverse=True)
        counters = []
        for act in acts(state, THEIRS):
            made = after(state, act)
            answers = list(acts(made, HERS)) if plies >= 2 else []
            best_answer, best = None, (judge(made) if may_leave or not answers else -math.inf)
            if answers:
                for answer in answers:
                    comes_to = value(after(made, answer), THEIRS, plies - 2, -math.inf, math.inf, known)
                    if comes_to > best:
                        best_answer, best = answer, comes_to
            counters.append((act, best_answer, best))
        return Possibilities(part=part, deep=plies, hers=tuple(hers), counters=tuple(counters))

    found = Possibilities(part=part, deep=0, hers=(), counters=())
    for plies in range(1, plies_at_most + 1):
        try:
            found = one_pass(plies)
        except _OutOfTime:
            break
    return found


@dataclass(frozen=True)
class Whole:
    """What the parts come to put back together, with ``first`` to act: her side's number, and which side takes which
    part in turn."""

    comes_to: float
    order: tuple[tuple[Hashable, int], ...]


def put_together(readings: Sequence[Reading], *, first: int = HERS) -> Whole:
    """The whole as the sum of its parts. The side to act takes the most urgent part, at what acting there first gives
    it; the other side the next most urgent; and so on while acting first in a part still gains anything. A part no one
    gains by acting in first stays as it stands."""
    by_urgency = sorted(readings, key=lambda reading: reading.urgency, reverse=True)
    total, side, order = 0.0, first, []
    for reading in by_urgency:
        if reading.urgency > 0:
            total += reading.if_hers_first if side == HERS else reading.if_theirs_first
            order.append((reading.part, side))
            side = 1 - side
        else:
            total += reading.as_it_stands
    return Whole(comes_to=total, order=tuple(order))


@dataclass(frozen=True)
class Weighed:
    """One act of hers, weighed by what the whole comes to after it."""

    part: Hashable
    act: Any
    whole: float
    #: What the whole comes to if she acts nowhere (the other side then acts first).
    acting_nowhere: float
    #: The part the other side takes next after this act, and how urgent it is; None where they gain nothing anywhere.
    leaves_them: tuple[Hashable, float] | None

    @property
    def worse_than_acting_nowhere(self) -> bool:
        return self.whole < self.acting_nowhere


def weigh_acts(readings: Sequence[Reading]) -> list[Weighed]:
    """Every act of hers in every part, best first, each by what the whole comes to after it: the part acted in at
    what the act leaves it, the rest put together with the other side to act."""
    nowhere = put_together(readings, first=THEIRS).comes_to
    weighed: list[Weighed] = []
    for reading in readings:
        rest = [other for other in readings if other.part != reading.part]
        then = put_together(rest, first=THEIRS)
        taken = next(iter(then.order), None)
        leaves = None
        if taken is not None:
            urgency = next(other.urgency for other in rest if other.part == taken[0])
            leaves = (taken[0], urgency)
        for act, comes_to in reading.acts:
            weighed.append(Weighed(part=reading.part, act=act, whole=comes_to + then.comes_to,
                                   acting_nowhere=nowhere, leaves_them=leaves))
    weighed.sort(key=lambda w: w.whole, reverse=True)
    return weighed


@dataclass(frozen=True)
class Choice:
    """The act the whole favours: the most good of her acts where one leaves her better off than acting nowhere, else
    the least bad of them."""

    weighed: Weighed | None
    most_good: bool


def what_to_do(readings: Sequence[Reading]) -> Choice:
    """Her act that leaves the whole best after the other side's best reply anywhere: called most good where it leaves
    her better than acting nowhere would, least bad where every act leaves her no better."""
    weighed = weigh_acts(readings)
    if not weighed:
        return Choice(None, False)
    best = weighed[0]
    return Choice(best, best.whole > best.acting_nowhere)


@dataclass(frozen=True)
class Threat:
    """One act of the other side's, by what it would do to her if she let it stand."""

    part: Hashable
    act: Any
    #: What the part would come to for her after it, her answers there looked at.
    leaves_her: float
    #: How much worse that is than the part as it stands.
    costs_her: float


def threats(readings: Sequence[Reading]) -> list[Threat]:
    """Every act the other side has, in every part, by what it would cost her there, worst first: what they can do to
    her, and so what she must answer or forestall."""
    found = [Threat(part=reading.part, act=act, leaves_her=comes_to, costs_her=reading.as_it_stands - comes_to)
             for reading in readings for act, comes_to in reading.their_acts]
    found.sort(key=lambda threat: threat.costs_her, reverse=True)
    return found


def where_they_work(acts_of_theirs: Sequence[Hashable], before: Sequence[Reading], after: Sequence[Reading]) -> list[
        tuple[Hashable, float]]:
    """The parts the other side is working in, by what their recent acts were in (``acts_of_theirs``: the part each
    was in, latest last, later counting more) and by how much more urgent their acts left each part
    (``before`` and ``after`` their acts): a part they keep acting in, and one they are making hotter, is where they are
    going. Most worked first, each with its share."""
    worked: dict[Hashable, float] = {}
    for when, part in enumerate(acts_of_theirs, start=1):
        worked[part] = worked.get(part, 0.0) + when / len(acts_of_theirs)
    then = {reading.part: reading.urgency for reading in before}
    for reading in after:
        warmer = reading.urgency - then.get(reading.part, reading.urgency)
        if warmer > 0:
            worked[reading.part] = worked.get(reading.part, 0.0) + warmer / max(1e-9, abs(reading.urgency) + warmer)
    total = sum(worked.values()) or 1.0
    return sorted(((part, share / total) for part, share in worked.items()), key=lambda pair: pair[1], reverse=True)


def what_it_changes(before: Sequence[Reading], after: Sequence[Reading]) -> dict[Hashable, dict[str, float]]:
    """What a change did to each part: how much what it stands at, and how urgent it is, moved. A part that appears or
    goes is a change of all of it."""
    then = {reading.part: reading for reading in before}
    now = {reading.part: reading for reading in after}
    changed: dict[Hashable, dict[str, float]] = {}
    for part in set(then) | set(now):
        old, new = then.get(part), now.get(part)
        stands = (new.as_it_stands if new else 0.0) - (old.as_it_stands if old else 0.0)
        urgency = (new.urgency if new else 0.0) - (old.urgency if old else 0.0)
        if stands or urgency:
            changed[part] = {"stands": stands, "urgency": urgency}
    return changed


@dataclass
class PartsSeen:
    """Parts read before, by what each looked like (a row of numbers its world gives it), so that a part not looked at
    is read from the parts like it. Kept to ``most``, the latest."""

    most: int
    seen: list[tuple[tuple[float, ...], Reading]] = field(default_factory=list)

    def keep(self, looks: Sequence[float], reading: Reading) -> None:
        self.seen.append((tuple(float(x) for x in looks), reading))
        del self.seen[:-self.most]

    def like(self, part: Hashable, looks: Sequence[float]) -> Reading | None:
        """A part read from the parts seen like it, each weighed by how near it looked (one that looked the same
        decides); None where none has been seen."""
        if not self.seen:
            return None
        looks = tuple(float(x) for x in looks)
        weights, readings = [], []
        for seen_looks, reading in self.seen:
            if len(seen_looks) != len(looks):
                continue
            distance = math.dist(seen_looks, looks)
            if distance == 0.0:
                return replace(reading, part=part, acts=(), looked=False, spread=0.0)
            weights.append(1.0 / (distance * distance))
            readings.append(reading)
        if not readings:
            return None
        total = sum(weights)

        def mean(of: Callable[[Reading], float]) -> float:
            return sum(w * of(r) for w, r in zip(weights, readings, strict=True)) / total

        urgency = mean(lambda r: r.urgency)
        spread = math.sqrt(max(0.0, mean(lambda r: (r.urgency - urgency) ** 2)))
        return Reading(part=part, if_hers_first=mean(lambda r: r.if_hers_first),
                       if_theirs_first=mean(lambda r: r.if_theirs_first), as_it_stands=mean(lambda r: r.as_it_stands),
                       looked=False, spread=spread)
