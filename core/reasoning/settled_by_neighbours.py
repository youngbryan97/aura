"""A constraint problem, each part settling by its neighbours, worked three ways.

Google's "Reasoning with Neural Cellular Automata" (arXiv 2609.36126) solves
Sudoku, mazes and other constraint puzzles with nothing but a rule each cell
applies to its neighbours, repeated. What makes it reason rather than guess is
not the rule but what is done around it, and that part needs no network:

    asynchronous, stochastic updates   cells fire at random, so an attempt can
                                       leave a configuration a lockstep update
                                       would hold it in
    firing by doubt                    a cell that is sure of itself rarely
                                       changes, so the work goes to what is
                                       still unresolved
    noise                              which lets an attempt out of a place no
                                       local change improves, by breaking more
                                       for a while
    many attempts in parallel          judged by their own confidence
    pruning that keeps them different  attempts that reach the same
                                       configuration are one niche; each niche
                                       keeps its best, and the half kept spans
                                       as many niches as it can

A variable is a cell, its neighbours are the variables it shares a constraint
with, and its rule has two halves. The first rules out the values its
neighbours leave no room for, and goes on until nothing changes; a cell's state
carries what it may still become, not only what it is. The second takes the
value that breaks fewest of its constraints, which on its own is min-conflicts
local search.

Three ways of working follow from those, used in the order they can prove
things:

    ruling out          a variable left with nothing proves there is no
                        solution; one value left everywhere is the solution
                        and the only one
    search with ruling  try a value for the most constrained variable, rule
      out at each step  out again, and back up on a contradiction; finished,
                        it proves how many solutions there are, counted to two
    many attempts       what is left of the budget, when searching could not
                        finish: the ensemble above, which finds solutions and
                        proves nothing

Measured on this machine, 2 October 2026. Ruling out alone settled an easy
Sudoku and proved it unique in under a tenth of a second. On planted
3-colourings of 120 vertices at 2.3 edges per vertex, four seconds each, the
ensemble solved 5 of 8 with noise for stuck attempts and 0 of 8 without it.
Pruning across niches measured no gain over plain halving there (6 of 8) or
on medium Sudoku: attempts at a search this size almost never land in the same
configuration, so the niches rarely bind. It is kept because it costs nothing
when they do not and the paper's case for it is where they do; it is not
credited with anything here.

Nothing here is a number picked for a puzzle. A round of the ensemble is split
from the budget the caller gives; how many attempts start is the largest power
of two whose first round still gives each one a full width of sweeps, the
width being the longest shortest path between two variables, because that is
how long a change takes to reach every part; an attempt counts as stuck once it
has gone a width of sweeps without improving.
"""

from __future__ import annotations

import random
import time
from collections import deque
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Constraint", "Problem", "Settled", "a_problem", "narrowed", "settle"]


@dataclass(frozen=True)
class Constraint:
    """A condition over some variables, called with their values in ``scope`` order."""

    scope: tuple[str, ...]
    holds: Callable[..., bool]
    said: str = ""


@dataclass
class Problem:
    """Variables with the values each may take, and the constraints between them.

    ``all_different`` names groups whose members must all take different
    values. Their pairwise constraints still have to be among ``constraints``;
    the group adds what pairs cannot say, that a value with only one place left
    in the group must go there.
    """

    domains: dict[str, tuple[Hashable, ...]]
    constraints: list[Constraint]
    all_different: list[tuple[str, ...]] = field(default_factory=list)
    _touching: dict[str, list[int]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._touching = {name: [] for name in self.domains}
        for at, constraint in enumerate(self.constraints):
            for name in set(constraint.scope):
                if name not in self._touching:
                    raise ValueError(f"constraint {constraint.said or at} names an unknown variable {name!r}")
                self._touching[name].append(at)

    def broken_by(self, values: Mapping[str, Hashable], name: str) -> int:
        """How many of this variable's constraints these values break."""
        return sum(1 for at in self._touching[name] if not self._holds(at, values))

    def broken(self, values: Mapping[str, Hashable]) -> int:
        return sum(1 for at in range(len(self.constraints)) if not self._holds(at, values))

    def _holds(self, at: int, values: Mapping[str, Hashable]) -> bool:
        constraint = self.constraints[at]
        return bool(constraint.holds(*(values[name] for name in constraint.scope)))

    def neighbours(self) -> dict[str, set[str]]:
        near: dict[str, set[str]] = {name: set() for name in self.domains}
        for constraint in self.constraints:
            for name in constraint.scope:
                near[name].update(other for other in constraint.scope if other != name)
        return near

    def width(self) -> int:
        """The longest shortest path between two connected variables."""
        near = self.neighbours()
        widest = 0
        for start in near:
            seen = {start: 0}
            queue = deque([start])
            while queue:
                here = queue.popleft()
                for there in near[here]:
                    if there not in seen:
                        seen[there] = seen[here] + 1
                        queue.append(there)
            widest = max(widest, max(seen.values()))
        return max(1, widest)


@dataclass
class Settled:
    """What the work came to."""

    solution: dict[str, Hashable] | None
    #: Every distinct configuration found in which every constraint holds.
    solutions: list[dict[str, Hashable]]
    #: Share of constraints holding in the best configuration at the end.
    confidence: float
    attempts: int
    sweeps: int
    why: str
    #: What ``solutions`` says was proved, not found: by ruling out or by a
    #: search that finished. Then one solution is the only one, two mean it is
    #: not unique, and none means there is none.
    proven: bool = False

    @property
    def unique_as_far_as_seen(self) -> bool:
        """One solution found. Proof when ``proven``; evidence otherwise."""
        return len(self.solutions) == 1


# ── ruling out ───────────────────────────────────────────────────────────────


def _has_room(constraint: Constraint, here: str, value: Hashable, others: list[Hashable] | None) -> bool:
    """Whether ``here`` taking ``value`` leaves the constraint some way to hold."""
    scope = constraint.scope
    if others is None:
        return bool(constraint.holds(*(value for _ in scope)))
    return any(
        constraint.holds(*(value if name == here else other for name in scope)) for other in others
    )


def _narrow(problem: Problem, domains: dict[str, list[Hashable]]) -> bool:
    """Strike out values with no room, in place, until nothing changes. False on a wipe-out."""
    pairs = [c for c in problem.constraints if 1 <= len(set(c.scope)) <= 2]
    changed = True
    while changed:
        changed = False
        for constraint in pairs:
            scope = constraint.scope
            names = list(dict.fromkeys(scope))
            for here in names:
                there = [name for name in names if name != here]
                kept = [
                    value
                    for value in domains[here]
                    if _has_room(constraint, here, value, domains[there[0]] if there else None)
                ]
                if len(kept) != len(domains[here]):
                    if not kept:
                        return False
                    domains[here] = kept
                    changed = True
        for group in problem.all_different:
            places: dict[Hashable, list[str]] = {}
            for name in group:
                for value in domains[name]:
                    places.setdefault(value, []).append(name)
            if len(places) < len(group):
                return False
            for value, where in places.items():
                if len(where) == 1 and len(domains[where[0]]) > 1:
                    domains[where[0]] = [value]
                    changed = True
            taken = [domains[name][0] for name in group if len(domains[name]) == 1]
            if len(taken) != len(set(taken)):
                return False
    return True


def narrowed(problem: Problem) -> Problem | None:
    """The problem with every value no neighbour leaves room for struck out; None if none can be."""
    domains = {name: list(values) for name, values in problem.domains.items()}
    if not _narrow(problem, domains):
        return None
    return Problem(
        {name: tuple(values) for name, values in domains.items()},
        list(problem.constraints),
        list(problem.all_different),
    )


# ── search with ruling out at each step ──────────────────────────────────────


class _OutOfTimeError(Exception):
    """The search ran past its deadline."""


def _searched(problem: Problem, deadline: float, want: int = 2) -> tuple[list[dict[str, Hashable]], bool]:
    """Every solution up to ``want``, and whether the search finished."""
    found: list[dict[str, Hashable]] = []

    def go(domains: dict[str, list[Hashable]]) -> None:
        if time.monotonic() > deadline:
            raise _OutOfTimeError
        if not _narrow(problem, domains):
            return
        open_ = [name for name, values in domains.items() if len(values) > 1]
        if not open_:
            values = {name: values[0] for name, values in domains.items()}
            if problem.broken(values) == 0:
                found.append(values)
            return
        name = min(open_, key=lambda one: len(domains[one]))
        for value in list(domains[name]):
            trial = {one: list(values) for one, values in domains.items()}
            trial[name] = [value]
            go(trial)
            if len(found) >= want:
                return

    try:
        go({name: list(values) for name, values in problem.domains.items()})
    except _OutOfTimeError:
        return found, False
    return found, True


# ── many attempts ────────────────────────────────────────────────────────────


@dataclass
class _Attempt:
    values: dict[str, Hashable]
    broken: int = 0
    #: The fewest constraints it has broken, and sweeps since it last did better.
    fewest: int = 1 << 30
    since_better: int = 0

    def key(self, order: Sequence[str]) -> tuple[Hashable, ...]:
        return tuple(self.values[name] for name in order)

    def copy(self) -> _Attempt:
        return _Attempt(dict(self.values), self.broken, self.fewest, self.since_better)


def _settle_one(problem: Problem, attempt: _Attempt, name: str, roll: random.Random) -> None:
    """The rule: take the value that breaks fewest of this variable's constraints."""
    values = attempt.values
    best: list[Hashable] = []
    fewest = None
    for value in problem.domains[name]:
        values[name] = value
        broken = problem.broken_by(values, name)
        if fewest is None or broken < fewest:
            fewest, best = broken, [value]
        elif broken == fewest:
            best.append(value)
    values[name] = roll.choice(best)


def _sweep(problem: Problem, attempt: _Attempt, order: list[str], roll: random.Random, width: int) -> None:
    """Every variable gets one chance to fire, in a fresh random order.

    A variable fires with the share of its constraints it breaks: one breaking
    none of them is sure of itself and is left alone. An attempt that has gone
    a whole width of sweeps without breaking fewer constraints is in a place
    nothing local improves, and there a variable in doubt may take a value at
    random instead — the temporary rise in broken constraints the automata
    used to get out.
    """
    roll.shuffle(order)
    stuck = attempt.since_better >= width
    for name in order:
        touching = len(problem._touching[name]) or 1  # noqa: SLF001 - its own index
        doubt = problem.broken_by(attempt.values, name) / touching
        if not doubt or roll.random() >= doubt:
            continue
        if stuck and roll.random() < doubt:
            attempt.values[name] = roll.choice(problem.domains[name])
        else:
            _settle_one(problem, attempt, name, roll)
    attempt.broken = problem.broken(attempt.values)
    if attempt.broken < attempt.fewest:
        attempt.fewest, attempt.since_better = attempt.broken, 0
    else:
        attempt.since_better += 1


def _kept_across_niches(attempts: list[_Attempt], order: list[str], keep: int) -> list[_Attempt]:
    """The best of each niche, then as many niches as the half allows, then copies of the best."""
    best_of: dict[tuple[Hashable, ...], _Attempt] = {}
    for one in attempts:
        key = one.key(order)
        if key not in best_of or one.broken < best_of[key].broken:
            best_of[key] = one
    ranked = sorted(best_of.values(), key=lambda one: one.broken)
    kept = [one.copy() for one in ranked[:keep]]
    at = 0
    while len(kept) < keep and ranked:
        kept.append(ranked[at % len(ranked)].copy())
        at += 1
    return kept


def _many_attempts(
    problem: Problem,
    *,
    budget_s: float,
    roll: random.Random,
    find_every: bool,
    start: Mapping[str, Hashable] | None,
) -> tuple[list[dict[str, Hashable]], _Attempt, int, int]:
    """Successive halving across niches: solutions found, the best attempt, attempts, sweeps."""
    order = list(problem.domains)
    began = time.monotonic()
    width = problem.width()

    def fresh() -> _Attempt:
        values = {
            name: (start[name] if start and name in start else roll.choice(problem.domains[name]))
            for name in order
        }
        return _Attempt(values, problem.broken(values))

    # What one sweep of one attempt costs here, measured, decides how many start.
    probe = fresh()
    timed = time.monotonic()
    _sweep(problem, probe, list(order), roll, width)
    a_sweep = max(1e-6, time.monotonic() - timed)
    affordable = max(0.0, budget_s - (time.monotonic() - began)) / a_sweep
    population, rounds = 1, 1
    while (population * 2) * (rounds + 1) * width <= affordable:
        population, rounds = population * 2, rounds + 1
    attempts = [probe] + [fresh() for _ in range(population - 1)]
    started_with = len(attempts)

    found: dict[tuple[Hashable, ...], dict[str, Hashable]] = {}
    if not probe.broken:
        found[probe.key(order)] = dict(probe.values)
    sweeps = 1
    round_at = 0
    a_round = max(0.0, budget_s - (time.monotonic() - began)) / rounds
    while True:
        round_ends = began + (round_at + 1) * a_round if round_at + 1 < rounds else began + budget_s
        while time.monotonic() < round_ends:
            moving = [one for one in attempts if one.broken]
            if not moving:
                break
            for one in moving:
                _sweep(problem, one, list(order), roll, width)
                if not one.broken:
                    found.setdefault(one.key(order), dict(one.values))
            sweeps += 1
            if found and not find_every:
                break
        round_at += 1
        if (found and not find_every) or time.monotonic() - began >= budget_s:
            break
        if all(not one.broken for one in attempts):
            # Every attempt has settled; another answer has to come from new starts.
            attempts = [fresh() for _ in attempts]
            continue
        if len(attempts) > 1:
            attempts = _kept_across_niches(attempts, order, max(1, len(attempts) // 2))
    best = min(attempts, key=lambda one: one.broken)
    return list(found.values()), best, started_with, sweeps


# ── the three, in order ──────────────────────────────────────────────────────


def settle(
    problem: Problem,
    *,
    budget_s: float,
    seed: int = 0,
    find_every: bool = True,
    start: Mapping[str, Hashable] | None = None,
) -> Settled:
    """Rule out, then search while there is time to finish, then many attempts.

    ``find_every`` keeps the attempts looking for further solutions until the
    budget is spent, so a second answer is seen if there is one. ``start`` seeds
    every attempt with a configuration, which is repair rather than search.
    """
    began = time.monotonic()
    whole = problem
    if not problem.domains:
        return Settled({}, [{}], 1.0, 0, 0, "nothing to settle", proven=True)
    narrow = narrowed(problem)
    if narrow is None:
        return Settled(None, [], 0.0, 0, 0, "ruling values out left a variable with none: no solution exists", proven=True)
    if all(len(values) == 1 for values in narrow.domains.values()):
        only = {name: values[0] for name, values in narrow.domains.items()}
        if whole.broken(only) == 0:
            return Settled(only, [only], 1.0, 0, 0, "ruling values out left one value everywhere: the only solution", proven=True)
        return Settled(None, [], 0.0, 0, 0, "the one value left everywhere breaks a constraint: no solution exists", proven=True)
    # Half the budget for a search that can prove; the rest, if it cannot
    # finish, for attempts that can only find.
    searched, finished = _searched(narrow, began + budget_s / 2.0)
    if finished:
        if not searched:
            why = "searched every possibility with ruling out at each step: no solution exists"
        elif len(searched) == 1:
            why = "searched every possibility with ruling out at each step: this is the only solution"
        else:
            why = "searched with ruling out at each step and found more than one solution"
        return Settled(
            searched[0] if searched else None, searched, 1.0 if searched else 0.0, 0, 0, why, proven=True
        )
    left = max(0.0, budget_s - (time.monotonic() - began))
    found, best, attempts, sweeps = _many_attempts(
        narrow, budget_s=left, roll=random.Random(seed), find_every=find_every, start=start
    )
    solutions = [values for values in [*searched, *found] if whole.broken(values) == 0]
    unique: dict[tuple[Hashable, ...], dict[str, Hashable]] = {}
    for values in solutions:
        unique.setdefault(tuple(values[name] for name in whole.domains), values)
    solutions = list(unique.values())
    total = max(1, len(whole.constraints))
    if solutions:
        why = (
            f"{len(solutions)} distinct solution(s) found by {attempts} attempt(s) after a search "
            "that could not finish; every constraint checked again, uniqueness not proven"
        )
    else:
        why = f"no attempt satisfied every constraint; the best broke {best.broken} of {total}"
    return Settled(
        solution=solutions[0] if solutions else None,
        solutions=solutions,
        confidence=1.0 if solutions else 1.0 - best.broken / total,
        attempts=attempts,
        sweeps=sweeps,
        why=why,
    )


def a_problem(
    domains: Mapping[str, Sequence[Hashable]],
    constraints: Sequence[Any],
    all_different: Sequence[Sequence[str]] = (),
) -> Problem:
    """A problem from plain parts: ``constraints`` as Constraint or (scope, holds[, said])."""
    made = []
    for one in constraints:
        if isinstance(one, Constraint):
            made.append(one)
        else:
            scope, holds, *said = one
            made.append(Constraint(tuple(scope), holds, said[0] if said else ""))
    return Problem(
        {name: tuple(values) for name, values in domains.items()},
        made,
        [tuple(group) for group in all_different],
    )
