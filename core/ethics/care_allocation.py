"""core/ethics/care_allocation.py — finite attention across people who need it.

Wanting to look after people is not one disposition. It is a scheduling
problem with an unusual constraint, and the constraint is where all the moral
content lives.

The problem: several others have unmet needs, attention and effort are finite,
and care does not pay off linearly — the tenth hour given to someone already
attended to buys less than the first hour given to someone who has had none.
Maximising the total good under a budget is ordinary convex optimisation, and
the answer is water-filling: give to whoever the next unit helps most, until
the marginal help is equal everywhere.

The constraint is what makes it care rather than throughput. Nel Noddings
described the one-caring as displacing her own motives toward the needs of the
cared-for, and the standing feminist criticism of that account — made most
directly by Eva Kittay, whose dependency workers are themselves left
unsupported — is that engrossment demanded of someone with no power to refuse
is a burden wearing the clothes of a virtue. Both halves are right, and the
difference between them is a modelling decision that this file makes
explicitly:

**The carer's own floor is a constraint, never a term.** Put self-care in the
objective with a weight and it is tradeable: there is always a need somewhere
large enough to buy it, and a system that maximises will find that need and
sell the floor. Put it in the feasible set and no amount of need elsewhere can
reach it. The optimiser then does something a weighted sum cannot — it
*refuses*, and says which need it refused for.

Three further pieces are needed before the model describes anything real.

**Care saturates, at a rate that differs by person.** Attention given to
someone who cannot use it right now is spent, not delivered. Each recipient
carries their own responsiveness, and it is estimated from what actually
happened rather than assumed.

**Care is completed by its reception.** Noddings' account requires the
cared-for to receive it; unreceived care is effort, not care. So the estimate
of responsiveness moves when care lands and when it does not, and the
allocation follows the estimate. This is the loop that keeps the module from
being a one-shot optimiser over numbers somebody typed in.

**Displacement is bounded and recorded.** Setting your own projects aside is
the mechanism; setting them aside indefinitely is the failure mode the
criticism names. The depth and the duration are both measured, and
``strain()`` reports when the pattern has become the thing Kittay described
rather than the thing Noddings did.

The module knows nothing about who is being cared for. Recipients are strings.
The same allocation runs over people, over services in a degraded system, or
over any set of claimants on one exhaustible budget.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("Aura.Ethics.Care")

#: Weighting exponent on unmet need. Zero gives the utilitarian answer, where
#: a unit of help is worth the same wherever it lands. One weights each
#: claimant by how much they are short, which is the ordinary reading of
#: attending to whoever needs it most. Larger values approach giving
#: everything to the worst-off claimant.
DEFAULT_PRIORITY = 1.0

#: Starting responsiveness for someone with no history, in budget units. It is
#: the scale over which care saturates, and it is a prior that the first few
#: observations overwrite rather than a constant the model relies on.
DEFAULT_RESPONSIVENESS = 1.0

#: Bounds on the estimate, so that a run of unreceived care cannot drive a
#: person's responsiveness to zero and write them out of every future
#: allocation permanently.
MIN_RESPONSIVENESS = 0.05
MAX_RESPONSIVENESS = 20.0

#: How fast the responsiveness estimate moves per observation. Slow enough
#: that one bad day does not rewrite a relationship.
RESPONSIVENESS_STEP = 0.2


@dataclass
class Recipient:
    """Someone with a need, and what has been learned about reaching them."""

    key: str
    need: float = 0.0
    responsiveness: float = DEFAULT_RESPONSIVENESS
    received: int = 0
    unreceived: int = 0
    last_care_at: float | None = None

    def benefit(self, care: float) -> float:
        """Good done by ``care`` units, saturating at the level of the need.

        Saturating rather than linear because attention someone cannot use
        right now is spent without being delivered, and a linear model will
        happily pour a whole budget into one person on the strength of a large
        number in a field.
        """
        if care <= 0 or self.need <= 0:
            return 0.0
        return self.need * (1.0 - math.exp(-care / max(self.responsiveness, MIN_RESPONSIVENESS)))

    def marginal(self, care: float) -> float:
        """Derivative of ``benefit`` at ``care``."""
        tau = max(self.responsiveness, MIN_RESPONSIVENESS)
        if self.need <= 0:
            return 0.0
        return (self.need / tau) * math.exp(-max(care, 0.0) / tau)

    def observe_reception(self, landed: bool) -> None:
        """Update responsiveness from whether care was taken up.

        Care that is not received is not care, on Noddings' account, and the
        estimate has to move in the direction the observation implies. Care
        that lands means less of it was needed to reach this person than the
        model assumed, so the saturation scale falls; care that does not land
        means the reverse. The step is taken in log space so the bounds hold
        without a clamp doing the work.
        """
        if landed:
            self.received += 1
            step = -RESPONSIVENESS_STEP
        else:
            self.unreceived += 1
            step = RESPONSIVENESS_STEP
        tau = max(self.responsiveness, MIN_RESPONSIVENESS)
        tau = math.exp(math.log(tau) + step)
        self.responsiveness = min(max(tau, MIN_RESPONSIVENESS), MAX_RESPONSIVENESS)

    @property
    def unreachable(self) -> bool:
        """Whether attention has stopped reaching this person.

        The allocator's answer for someone whose responsiveness has run to the
        top of its range is to give them nothing, and that answer is correct
        for the question it was asked: no amount of the budget is landing. It
        is the wrong conclusion to leave unremarked, because what it actually
        says is that this needs something other than more attention. The flag
        is here so the silence in the allocation has a name attached to it.
        """
        return (
            self.responsiveness >= MAX_RESPONSIVENESS * 0.9
            and self.unreceived > self.received
        )


@dataclass(frozen=True)
class Allocation:
    """One decision about where finite care goes, with what it refused."""

    given: dict[str, float]
    budget: float
    reserved: float
    """Held back for the carer. Never available to the optimiser."""

    unmet: dict[str, float]
    refused_for_floor: float
    """Need the floor prevented meeting. Zero when the budget covered it all."""

    gini: float
    priority: float
    at: float

    @property
    def spent(self) -> float:
        return float(sum(self.given.values()))

    def as_dict(self) -> dict[str, Any]:
        return {
            "given": {k: round(v, 4) for k, v in sorted(self.given.items())},
            "budget": round(self.budget, 4),
            "reserved": round(self.reserved, 4),
            "spent": round(self.spent, 4),
            "unmet": {k: round(v, 4) for k, v in sorted(self.unmet.items()) if v > 1e-9},
            "refused_for_floor": round(self.refused_for_floor, 4),
            "gini": round(self.gini, 4),
            "priority": self.priority,
        }


@dataclass
class Displacement:
    """A stretch during which the carer's own projects were set aside."""

    started_at: float
    depth: float
    """Fraction of the carer's own budget diverted, in [0, 1]."""

    ended_at: float | None = None

    def duration_s(self, now: float | None = None) -> float:
        end = self.ended_at if self.ended_at is not None else (now or time.time())
        return max(0.0, end - self.started_at)


def gini(values: Sequence[float]) -> float:
    """Inequality of an allocation, in [0, 1].

    Reported next to every allocation because the total says nothing about the
    shape, and two allocations that spend the same budget can differ between
    everyone getting a little and one person getting everything.

    Callers pass one entry per claimant including the ones who received
    nothing. Dropping the zeros measures inequality among the funded, which
    falls as the budget is concentrated on fewer people.
    """
    xs = sorted(max(0.0, float(v)) for v in values)
    n = len(xs)
    total = sum(xs)
    if n == 0 or total <= 0:
        return 0.0
    weighted = sum((i + 1) * x for i, x in enumerate(xs))
    return float((2.0 * weighted) / (n * total) - (n + 1.0) / n)


class CareAllocator:
    """Water-filling over claimants, under a floor that cannot be traded.

    The optimiser solves for the level at which marginal benefit is equal
    across everyone who gets anything. With saturating benefit the condition
    is

        w_i * (n_i / tau_i) * exp(-c_i / tau_i) = lambda

    which inverts to ``c_i = tau_i * ln(w_i n_i / (tau_i lambda))``, floored at
    zero. One bisection on lambda gives the allocation exactly; there is no
    iterative heuristic here and no place for a tuned constant to hide.
    """

    def __init__(
        self,
        *,
        priority: float = DEFAULT_PRIORITY,
        self_floor: float = 0.0,
    ) -> None:
        self.priority = float(priority)
        self.self_floor = float(self_floor)
        self._recipients: dict[str, Recipient] = {}
        self._history: list[Allocation] = []
        self._displacements: list[Displacement] = []
        self._own_need_unmet: list[float] = []

    # ----------------------------------------------------------------- state

    def recipient(self, key: str) -> Recipient:
        person = self._recipients.get(key)
        if person is None:
            person = Recipient(key=key)
            self._recipients[key] = person
        return person

    def set_need(self, key: str, need: float) -> None:
        self.recipient(key).need = max(0.0, float(need))

    def observe_reception(self, key: str, landed: bool) -> None:
        self.recipient(key).observe_reception(landed)

    # -------------------------------------------------------------- allocate

    def _weight(self, person: Recipient) -> float:
        if person.need <= 0:
            return 0.0
        return float(person.need ** self.priority)

    def allocate(
        self,
        budget: float,
        *,
        needs: Mapping[str, float] | None = None,
        at: float | None = None,
        record: bool = True,
    ) -> Allocation:
        """Spread ``budget`` over the claimants, keeping the floor intact."""
        moment = at if at is not None else time.time()
        if needs:
            for key, value in needs.items():
                self.set_need(key, value)
        people = [p for p in self._recipients.values() if p.need > 0]
        available = max(0.0, float(budget) - self.self_floor)
        refused = 0.0
        if float(budget) < self.self_floor:
            # The floor is a constraint, so this is a refusal rather than a
            # smaller allocation. Naming the need it refused for is the part a
            # weighted objective could not have produced.
            refused = float(sum(p.need for p in people))
            allocation = Allocation(
                given={}, budget=float(budget), reserved=float(budget),
                unmet={p.key: p.need for p in people},
                refused_for_floor=refused, gini=0.0, priority=self.priority,
                at=moment,
            )
            if record:
                self._remember(allocation)
            return allocation

        given = self._water_fill(people, available)
        spent = sum(given.values())
        unmet = {p.key: max(0.0, p.need - given.get(p.key, 0.0)) for p in people}
        total_unmet = sum(unmet.values())
        if total_unmet > 0 and available < sum(p.need for p in people):
            # Some of the shortfall is the floor's doing: it is the part that
            # the reserved budget would have covered.
            refused = min(total_unmet, self.self_floor)
        allocation = Allocation(
            given=given,
            budget=float(budget),
            reserved=float(budget) - spent,
            unmet=unmet,
            refused_for_floor=refused,
            # Over every claimant, not only the funded ones. Excluding the
            # people who got nothing makes concentrating the budget on fewer
            # of them read as a fall in inequality, which is the opposite of
            # what happened.
            gini=gini([given.get(p.key, 0.0) for p in people]) if people else 0.0,
            priority=self.priority,
            at=moment,
        )
        if record:
            self._remember(allocation)
        return allocation

    def _water_fill(self, people: Sequence[Recipient], available: float) -> dict[str, float]:
        if not people or available <= 0:
            return {}

        def spend_at(lam: float) -> dict[str, float]:
            out: dict[str, float] = {}
            for p in people:
                w = self._weight(p)
                tau = max(p.responsiveness, MIN_RESPONSIVENESS)
                top = w * (p.need / tau)
                if top <= 0 or lam <= 0 or lam >= top:
                    continue
                care = tau * math.log(top / lam)
                if care > 0:
                    out[p.key] = min(care, p.need)
            return out

        # Bracket lambda. At the largest marginal nobody is funded; at a
        # vanishing marginal everyone is funded to their whole need.
        high = max(self._weight(p) * (p.need / max(p.responsiveness, MIN_RESPONSIVENESS))
                   for p in people)
        low = high * 1e-12
        if sum(spend_at(low).values()) <= available:
            return spend_at(low)
        for _ in range(200):
            mid = math.sqrt(low * high) if low > 0 else high / 2.0
            total = sum(spend_at(mid).values())
            if total > available:
                low = mid
            else:
                high = mid
            if high - low <= high * 1e-12:
                break
        result = spend_at(high)
        # Numerical scaling so the budget constraint holds exactly rather than
        # to within the bisection's tolerance.
        total = sum(result.values())
        if total > available > 0:
            scale = available / total
            result = {k: v * scale for k, v in result.items()}
        return result

    # ------------------------------------------------------------ the carer

    def _remember(self, allocation: Allocation) -> None:
        self._history.append(allocation)
        if len(self._history) > 512:
            del self._history[: len(self._history) - 512]

    def begin_displacement(self, depth: float, *, at: float | None = None) -> Displacement:
        """Record that own projects are being set aside, and by how much."""
        record = Displacement(
            started_at=at if at is not None else time.time(),
            depth=min(max(float(depth), 0.0), 1.0),
        )
        self._displacements.append(record)
        if len(self._displacements) > 512:
            del self._displacements[: len(self._displacements) - 512]
        return record

    def end_displacement(self, *, at: float | None = None) -> None:
        for record in reversed(self._displacements):
            if record.ended_at is None:
                record.ended_at = at if at is not None else time.time()
                return

    def record_own_unmet(self, amount: float) -> None:
        """How much of the carer's own need went unmet on this occasion."""
        self._own_need_unmet.append(max(0.0, float(amount)))
        if len(self._own_need_unmet) > 512:
            del self._own_need_unmet[: len(self._own_need_unmet) - 512]

    def strain(self, *, window_s: float = 7 * 86400.0,
               at: float | None = None) -> dict[str, Any]:
        """Whether the pattern has become the one the criticism names.

        Three separate readings, kept separate. Sustained depth is how much of
        her own budget is going elsewhere. Open duration is displacement that
        has not ended. Own unmet is the carer's own shortfall, and it is the
        one that decides the verdict, because a carer whose own needs are met
        can give a great deal without any of this being a problem.
        """
        moment = at if at is not None else time.time()
        recent = [d for d in self._displacements if moment - d.started_at <= window_s]
        depth = float(sum(d.depth for d in recent) / len(recent)) if recent else 0.0
        open_now = [d for d in self._displacements if d.ended_at is None]
        longest_open = max((d.duration_s(moment) for d in open_now), default=0.0)
        own = self._own_need_unmet[-len(recent):] if recent else self._own_need_unmet[-10:]
        own_unmet = float(sum(own) / len(own)) if own else 0.0
        floor_bound = sum(
            1 for a in self._history
            if moment - a.at <= window_s and a.refused_for_floor > 0
        )
        unreachable = sorted(p.key for p in self._recipients.values() if p.unreachable)
        return {
            "mean_depth": round(depth, 4),
            "open_displacements": len(open_now),
            "longest_open_s": round(longest_open, 2),
            "own_unmet": round(own_unmet, 4),
            # Occasions where the reserve was the reason a need went unmet.
            # The floor holding is the healthy state, so this is not a fault
            # count; it is what the floor cost, which somebody should be able
            # to read.
            "occasions_floor_bound": floor_bound,
            "unreachable": unreachable,
            # Depletion is the carer giving while going short herself, which
            # is the arrangement the floor exists to make impossible and can
            # only arise when something bypasses this allocator.
            "depleted": own_unmet > 0 and depth > 0,
        }

    def status(self, *, at: float | None = None) -> dict[str, Any]:
        last = self._history[-1] if self._history else None
        return {
            "priority": self.priority,
            "self_floor": self.self_floor,
            "recipients": {
                p.key: {
                    "need": round(p.need, 4),
                    "responsiveness": round(p.responsiveness, 4),
                    "received": p.received,
                    "unreceived": p.unreceived,
                }
                for p in sorted(self._recipients.values(), key=lambda r: r.key)
            },
            "last": last.as_dict() if last else None,
            "allocations": len(self._history),
            "strain": self.strain(at=at),
        }


_ALLOCATOR: CareAllocator | None = None


def get_care_allocator() -> CareAllocator:
    global _ALLOCATOR
    if _ALLOCATOR is None:
        _ALLOCATOR = CareAllocator()
    return _ALLOCATOR


def reset_care_allocator_for_test() -> None:
    global _ALLOCATOR
    _ALLOCATOR = None
