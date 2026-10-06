"""Identify an action's responder from independent, repeated experiments.

Callers provide measurements and observations; this executor neither labels
objects by appearance nor infers private application state. Its receipts
describe the evidence used, rather than a calibrated probability of truth.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Hashable

from core.verify import invariant

@dataclass(frozen=True)
class CausalWitness:
    identity: Hashable
    epoch: int
    independent_trials: tuple[tuple[str, int], ...]
    statistic: float
    effect: float


class CausalIdentification:
    """Keep an established attribution until absence or contradiction resets its experiment."""

    def __init__(self, *, statistic_over: float, effect_over: float) -> None:
        self.statistic_over, self.effect_over = statistic_over, effect_over
        self.epoch = 0
        self.receipts: deque[dict] = deque(maxlen=64)

    def reset(self, reason: str) -> None:
        self.epoch += 1
        self.receipts.append({"epoch": self.epoch, "reason": reason, "selected": None})

    def choose(self, witnesses: list[CausalWitness], *, visible: set[Hashable],
               established: Hashable | None, excluded: set[Hashable]) -> Hashable | None:
        """An independent trial may establish identity; feedback chosen by a plan may not."""
        if established is not None and established in visible and established not in excluded:
            return established
        accepted = [w for w in witnesses
                    if w.epoch == self.epoch and w.identity in visible and w.identity not in excluded
                    and len(dict(w.independent_trials)) == len(w.independent_trials)
                    and sum(type(n) is int and n >= 2 for _key, n in w.independent_trials) >= 2
                    and math.isfinite(w.statistic) and math.isfinite(w.effect)
                    and w.statistic > self.statistic_over and w.effect > self.effect_over]
        accepted.sort(key=lambda w: w.statistic, reverse=True)
        # Equal measurements do not distinguish two responders. Another
        # experiment must distinguish them before either becomes an identity.
        if not accepted or (len(accepted) > 1 and accepted[0].statistic == accepted[1].statistic):
            return None
        best = accepted[0]
        self.receipts.append({"epoch": self.epoch, "reason": "independent repeated trials",
                              "selected": best.identity, "statistic": best.statistic,
                              "effect": best.effect, "independent_trials": dict(best.independent_trials),
                              "eligible_candidates": len(accepted)})
        return best.identity


def within_observed_reach(previous: tuple[float, float], position: tuple[float, float], *,
                          extent: tuple[float, float], velocity: tuple[float, float], gap: float) -> bool:
    """A continuing identity must lie within its measured motion and visible extent."""
    values = (*previous, *position, *extent, *velocity, gap)
    if not all(math.isfinite(v) for v in values) or gap < 0 or min(extent) <= 0:
        return False
    # Tracking uncertainty is the visible object's extent plus a working
    # pixel margin. A long unseen interval supplies no spatial attribution.
    if gap > 0.5:
        return False
    return all(abs(now - before) < 6.0 + size + 1.5 * abs(speed) * gap
               for before, now, size, speed in zip(previous, position, extent, velocity, strict=True))


def _stale_and_ambiguous_identity_witnesses_remain_unknown() -> bool:
    ledger = CausalIdentification(statistic_over=20, effect_over=20)
    first = CausalWitness("one", 0, (("a", 3), ("b", 3)), 100, 80)
    second = CausalWitness("two", 0, first.independent_trials, 100, 80)
    ambiguous = ledger.choose([first, second], visible={"one", "two"}, established=None, excluded=set()) is None
    ledger.reset("new experiment")
    stale = ledger.choose([first], visible={"one"}, established=None, excluded=set()) is None
    return ambiguous and stale


@invariant("agency.identity_requires_current_distinguishing_evidence", scope="agency",
           owner="core/agency/causal_identification.py", observational=False)
def _causal_identity_invariant() -> tuple:
    assert _stale_and_ambiguous_identity_witnesses_remain_unknown(), "stale or ambiguous evidence established identity"
    return ()
