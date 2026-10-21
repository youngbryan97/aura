"""Condition orders that keep the schedule from standing in for an effect.

Every recorded round takes the eight conditions in one fixed order, so a
variable that follows the schedule follows the condition before as well as the
one it is in, and a synergy between two sources can be the schedule's. Shuffling
a recording afterwards would break whatever real dependence runs from one turn
to the next along with the schedule's, so the order is varied when the data are
collected instead.

A Williams design (Williams 1949) for an even number of conditions is a Latin
square in which every condition comes first, last and in every position once,
and immediately follows every other condition exactly once. Its rows are drawn
in a random order for each block of rounds, from the run's seed, so no row
falls on the same rounds in every run.
"""
from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

__all__ = ["round_orders", "williams_square"]


def williams_square(n: int) -> list[list[int]]:
    """A Williams square of order `n`, balanced for the condition before when `n` is even.

    The first row runs 0, 1, n-1, 2, n-2, ...; every other row adds its index
    to each entry, modulo n. For odd `n` the mirrored rows are added, which
    balances carry-over across the doubled square.
    """
    first = [0]
    low, high = 1, n - 1
    while len(first) < n:
        first.append(low)
        low += 1
        if len(first) < n:
            first.append(high)
            high -= 1
    rows = [[(entry + shift) % n for entry in first] for shift in range(n)]
    if n % 2:
        rows += [list(reversed(row)) for row in rows]
    return rows


def round_orders(conditions: Sequence[Any], rounds: int, *, seed: int = 0) -> list[list[Any]]:
    """One order of the conditions per round, rows of the square drawn afresh each block."""
    square = williams_square(len(conditions))
    rng = random.Random(seed)
    out: list[list[Any]] = []
    while len(out) < rounds:
        block = list(range(len(square)))
        rng.shuffle(block)
        for row in block:
            out.append([conditions[index] for index in square[row]])
            if len(out) == rounds:
                break
    return out
