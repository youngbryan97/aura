"""Prospective subset swaps for effects hidden by a whole-periphery swap.

For example, a*b*(1-c)*(1-d) is zero at both 0000 and 1111, after
every single swap from either state, and after a whole swap. Swapping just
a and b from 0000 exposes its effect. A negative whole swap therefore does
not exclude every interaction. These plans cover additional combinations;
they make no claim about donors, horizons, or units the experiment omits.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CoalitionPlan:
    units: tuple[tuple[str, str | None], ...]
    subsets: tuple[tuple[int, ...], ...]
    seed: int
    requested: int

    @property
    def possible(self) -> int:
        # Empty, single-unit and whole swaps are already separate arms.
        return max(0, (1 << len(self.units)) - len(self.units) - 2)

    def description(self) -> dict[str, Any]:
        return {
            "units": [[name, key] for name, key in self.units],
            "subsets": [list(indices) for indices in self.subsets],
            "seed": self.seed,
            "requested": self.requested,
            "selected": len(self.subsets),
            "possible": str(self.possible),
            "complete_on_declared_units": len(self.subsets) == self.possible,
            "scope": "Proper subsets of these units, on the collected donors and declared horizon only.",
        }


def plan_coalitions(units: list[tuple[str, str | None]], *, budget: int, seed: int) -> CoalitionPlan:
    """Enumerate when the budget covers all subsets, otherwise sample without replacement.

    The unit order and random seed fix the plan before any outcome is read.
    Each sampled bit is fair; rejecting existing arms and duplicate masks
    gives a uniform sample of the remaining subsets. At most 1,024 added
    arms are accepted so an accidental argument cannot start an unbounded run.
    """
    if not 0 <= budget <= 1024:
        raise ValueError("coalition budget must be between 0 and 1024")
    if len(set(units)) != len(units):
        raise ValueError("coalition units must be distinct")
    width = len(units)
    possible = max(0, (1 << width) - width - 2)
    count = min(budget, possible)
    subsets: list[tuple[int, ...]] = []
    if count == possible and count:
        subsets = [indices for size in range(2, width)
                   for indices in combinations(range(width), size)]
    elif count:
        rng = np.random.default_rng(seed)
        seen: set[tuple[int, ...]] = set()
        # Near-exhaustive small plans can require many duplicate proposals.
        # Sampling from their explicit list is bounded and uniform instead.
        if possible <= 4096:
            all_subsets = [indices for size in range(2, width)
                           for indices in combinations(range(width), size)]
            subsets = [all_subsets[int(i)] for i in rng.choice(possible, size=count, replace=False)]
        else:
            for _ in range(20 * count):
                indices = tuple(int(i) for i in np.flatnonzero(rng.integers(0, 2, width)))
                if 2 <= len(indices) < width and indices not in seen:
                    seen.add(indices)
                    subsets.append(indices)
                    if len(subsets) == count:
                        break
            if len(subsets) != count:
                raise RuntimeError("could not fill the declared coalition budget")
    return CoalitionPlan(tuple(units), tuple(subsets), int(seed), int(budget))
