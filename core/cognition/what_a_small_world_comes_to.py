"""What a small world comes to: every situation that can follow worked out exactly, when there are few enough to list.

A search looks ahead as far as its time allows and judges where it stops. Where every situation that can follow from
here can be listed, nothing has to be judged: list them all, follow every act from each, and work back from the ends.
That settles each situation outright, with how many turns it takes, and rests on nothing but what each act leads to.

The same working-out serves any world that is finite and fully seen, whoever acts in it:

- a world she acts in alone (a puzzle, a form, a menu): from each situation, whether she can reach a good end and in
  how few acts;
- a world where another side acts against her in turn (a board game): whether she can force a good end however they
  answer, whether they can force a bad one however she does, or neither, and in how many turns under best play on both
  sides (the quicker win, the longer loss);
- a world that deals something of its own between her acts, by known shares (a dealt tile): what she can force, and
  her chance of a good end when she plays for it.

The caller says what a situation is, what each act from it leads to, who decides there, and what an end is worth to
her. Nothing here knows any world. Where there are more situations than ``most``, or the listing runs past its time,
what was not listed counts as neither good nor bad, and the result says it is incomplete.

Suggestions of 23 November 2026, item 9 ("exact solving of small subgames"); Bryan, the same day: as generally
applicable as it can be.
"""
from __future__ import annotations

import time
from array import array
from collections import deque
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["AGAINST", "CHANCE", "HERS", "Worked", "worked_out"]

#: Who decides at a situation: she does, someone against her does, or the world deals by known shares.
HERS, AGAINST, CHANCE = "hers", "against", "chance"
_WHO = {HERS: 0, AGAINST: 1, CHANCE: 2}
#: How close her chance of a good end must come to settling, a pass to the next, before it is taken as settled: the
#: precision a float holds a share to.
SETTLED_TO = 1e-12


@dataclass
class Worked:
    """Every listed situation, worked out. ``forced`` is +1 where she can force a good end, -1 where the other side
    (or the world at its worst) can force a bad one, 0 where neither can; ``turns`` how many acts that takes under best
    play, -1 where neither. ``chance`` is her chance of a good end playing for it, where the world deals."""

    names: dict[Hashable, int]
    states: list[Any] = field(repr=False)
    acts: list[list[tuple[Any, int]]] = field(repr=False)
    who: np.ndarray = field(repr=False)
    forced: np.ndarray = field(repr=False)
    turns: np.ndarray = field(repr=False)
    chance: np.ndarray | None = field(repr=False)
    shares: list[list[float]] = field(repr=False)
    complete: bool = True
    #: Whether her chance of a good end settled, every situation's to within what a float holds; where it did not
    #: (out of time, in a world whose dealing goes round and round), ``chance`` is a bound from below: each pass only
    #: raises it toward the true chance.
    chance_settled: bool = True
    name_of: Callable[[Any], Hashable] = field(default=lambda state: state, repr=False)

    def __len__(self) -> int:
        return len(self.forced)

    def at(self, state: Any) -> int | None:
        return self.names.get(self.name_of(state))

    def what_it_comes_to(self, state: Any) -> tuple[int, int] | None:
        """What a situation comes to for her (+1, -1 or 0) and in how many acts; None where it was not listed."""
        index = self.at(state)
        if index is None:
            return None
        return int(self.forced[index]), int(self.turns[index])

    def best(self, state: Any) -> Any:
        """The act whoever decides here takes under best play: for her the quickest forced good end, else the act
        that keeps a good end most likely or a bad one unforced, else the longest way to a bad end; for the side
        against her the same, the other way. None at an end, or where the situation was not listed."""
        index = self.at(state)
        if index is None or not self.acts or not self.acts[index]:
            return None
        hers = self.who[index] != _WHO[AGAINST]

        def preference(child: int) -> tuple:
            # A situation past what was listed comes to neither, as far as anything here knows.
            if child < 0:
                forced, turns, chance = 0, -1, 0.0
            else:
                forced, turns = int(self.forced[child]), int(self.turns[child])
                chance = float(self.chance[child]) if self.chance is not None else 0.0
            if not hers:
                forced, chance = -forced, 1.0 - chance
            if forced > 0:
                return 0, turns, -chance
            if forced == 0:
                return 1, -chance, 0
            return 2, -turns, -chance

        act, _child = min(self.acts[index], key=lambda pair: preference(pair[1]))
        return act


def worked_out(starts: Iterable[Any], next_of: Callable[[Any], Iterable[tuple]],
               end_of: Callable[[Any], float | None], whose: Callable[[Any], str], *,
               name_of: Callable[[Any], Hashable] = lambda state: state, most: int = 1_000_000,
               seconds: float | None = None, keep: bool = True) -> Worked:
    """Every situation reachable from ``starts``, worked out.

    ``next_of(state)`` gives each act from a situation as ``(act, what it leads to)``, or ``(act, what it leads to,
    share)`` where the world deals; ``end_of(state)`` what an end is worth to her (above nought good, below bad,
    nought neither), or None where the situation goes on; ``whose(state)`` who decides there (HERS, AGAINST or
    CHANCE); ``name_of(state)`` one name for one situation. At most ``most`` situations are listed, within
    ``seconds`` where given. With ``keep`` False the situations and acts are not kept, only what each comes to: half a
    million situations of a board ending, kept whole, run to gigabytes beside her."""
    began = time.monotonic()
    names: dict[Hashable, int] = {}
    states: list[Any] = []
    ends: list[float | None] = []
    who = array("b")
    acts: list[list[tuple[Any, int]]] = []
    shares: list[list[float]] = []
    # Each situation's acts as the situations they lead to, end to end, with where each situation's begin.
    leads, begins, counts = array("q"), array("q"), array("q")
    dealt = array("d")
    waiting: deque[tuple[int, Any]] = deque()
    complete = True

    def listed(state: Any) -> int:
        nonlocal complete
        name = name_of(state)
        index = names.get(name)
        if index is not None:
            return index
        # Counted against every situation listed: with ``keep`` False none is kept, and a cap counted against what
        # was kept never applied (suggestions of 23 November 2026: asked to stop at 10, it listed 51).
        if len(ends) >= most or (seconds is not None and time.monotonic() - began > seconds):
            complete = False
            return -1
        index = len(ends)
        names[name] = index
        if keep:
            states.append(state)
            acts.append([])
            shares.append([])
        ends.append(None)
        who.append(0)
        begins.append(0)
        counts.append(0)
        waiting.append((index, state))
        return index

    for state in starts:
        listed(state)
    while waiting:
        if seconds is not None and time.monotonic() - began > seconds:
            # Out of time: what is still waiting is listed and not followed, and comes to neither.
            complete = False
            break
        index, state = waiting.popleft()
        worth = end_of(state)
        if worth is not None:
            ends[index] = float(worth)
            continue
        who[index] = _WHO[whose(state)]
        begins[index] = len(leads)
        for step in next_of(state):
            act, after = step[0], step[1]
            child = listed(after)
            share = float(step[2]) if len(step) > 2 and step[2] is not None else 0.0
            leads.append(child)
            dealt.append(share)
            counts[index] += 1
            if keep:
                acts[index].append((act, child))
                shares[index].append(share)
    count = len(ends)
    end = np.array([np.nan if e is None else e for e in ends], dtype=np.float64)
    deciding = np.frombuffer(who, dtype=np.int8).copy() if count else np.zeros(0, dtype=np.int8)
    forced = np.zeros(count, dtype=np.int8)
    turns = np.full(count, -1, dtype=np.int32)
    forced[end > 0], forced[end < 0] = 1, -1
    turns[~np.isnan(end)] = 0
    many = np.frombuffer(counts, dtype=np.int64) if count else np.zeros(0, dtype=np.int64)
    # A situation that goes on and offers no act is an end worth nothing to either side.
    owners = np.flatnonzero(np.isnan(end) & (many > 0)).astype(np.int64)
    if len(owners):
        # Situations were taken up in the order they were listed and their acts kept as they came, so the acts of
        # those that go on lie end to end in that order, each run beginning where ``begins`` says.
        first = np.frombuffer(begins, dtype=np.int64)[owners].copy()
        child = np.frombuffer(leads, dtype=np.int64)
        listed_child = child >= 0
        safe = np.where(listed_child, child, 0)
        kind = deciding[owners]
        for side in (1, -1):
            _forced_by(side, forced, turns, owners, first, safe, listed_child, kind)
    chance, settled = None, True
    if (deciding == _WHO[CHANCE]).any() and len(owners):
        chance, settled = _chance_of_a_good_end(end, owners, first, safe, listed_child, kind,
                                                np.frombuffer(dealt, dtype=np.float64),
                                                None if seconds is None else began + seconds)
    return Worked(names=names, states=states, acts=acts, who=deciding, forced=forced, turns=turns, chance=chance,
                  chance_settled=settled,
                  shares=shares, complete=complete, name_of=name_of)


def _forced_by(side: int, forced: np.ndarray, turns: np.ndarray, owners: np.ndarray, first: np.ndarray,
               child: np.ndarray, listed_child: np.ndarray, kind: np.ndarray) -> None:
    """The situations from which an end worth ``side`` to her can be forced, back from the ends, a pass a turn: where
    the one who decides wants that end (her for a good end, the side against her for a bad one), any act that leads to
    one; where they do not, or the world deals, every act. Each pass settles exactly the situations one more turn
    away, so the turns counted are those of best play: the quicker forced end, the longer delay."""
    never = np.int32(1 << 30)
    wants = 0 if side > 0 else 1
    while True:
        there = listed_child & (forced[child] == side)
        reaching = np.where(there, turns[child], never)
        soonest = np.minimum.reduceat(reaching, first)
        every = np.minimum.reduceat(there.astype(np.int8), first) == 1
        latest = np.maximum.reduceat(np.where(there, turns[child], -1), first)
        open_ = forced[owners] == 0
        by_any = open_ & (kind == wants) & (soonest < never)
        by_every = open_ & (kind != wants) & every
        if not (by_any.any() or by_every.any()):
            return
        forced[owners[by_any]] = side
        turns[owners[by_any]] = 1 + soonest[by_any]
        forced[owners[by_every]] = side
        turns[owners[by_every]] = 1 + latest[by_every]


def _chance_of_a_good_end(end: np.ndarray, owners: np.ndarray, first: np.ndarray, child: np.ndarray,
                          listed_child: np.ndarray, kind: np.ndarray, share: np.ndarray,
                          until: float | None) -> tuple[np.ndarray, bool]:
    """Her chance of a good end from each situation, playing for it against a side that plays against it, with the
    world dealing by its shares, and whether it settled. From nothing, each pass can only raise a situation's chance
    toward the true one, so whatever pass it stops at is a bound from below; it stops where no situation's chance
    moves by more than a float holds, or at ``until`` (a time). A cap of as many passes as there are situations
    stopped a world that deals the same situation again and again short of its answer: a chance of a half each time,
    over and over, came out three in four where it is certain (suggestions of 23 November 2026)."""
    chance = np.where(np.isnan(end), 0.0, (end > 0).astype(np.float64))
    totals = np.add.reduceat(share, first)
    while True:
        theirs = np.where(listed_child, chance[child], 0.0)
        best = np.maximum.reduceat(theirs, first)
        worst = np.minimum.reduceat(theirs, first)
        dealt = np.add.reduceat(theirs * share, first) / np.where(totals > 0, totals, 1.0)
        now = np.where(kind == 0, best, np.where(kind == 1, worst, dealt))
        moved = float(np.max(np.abs(now - chance[owners]))) if len(owners) else 0.0
        chance[owners] = now
        if moved <= SETTLED_TO:
            return chance, True
        if until is not None and time.monotonic() >= until:
            return chance, False
