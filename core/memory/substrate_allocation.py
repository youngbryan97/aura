"""core/memory/substrate_allocation.py — which kind of memory this should be.

Aura can hold something in at least three places, and they are not the same
kind of holding:

**Symbolic.** A fact in a store, retrieved by lookup. Cheap to write, cheap to
change, exact, and it only helps when something goes looking for it.

**Adapter.** A low-rank change to the model, loaded for the situations it is
about. More expensive to make, applies without being asked, reversible by
unloading, and it competes with the other adapters for the same capacity.

**Weights.** Folded into the model itself. Most expensive by far, applies
always, and effectively permanent — which is the property that makes it
valuable and the property that makes it dangerous.

Nothing in the runtime chose between them. Every durable thing became a
symbolic record, because that is the cheap path, and so the model itself
learned nothing from a year of conversation while the store filled with facts
nobody queried.

Choosing needs three quantities and they pull in different directions:

    lifetime usefulness   how often it will be needed, over how long
    confidence            how sure we are it is true and will stay true
    interference cost     what writing it costs everything already there

Interference is the one that makes this a real problem rather than a
threshold. A symbolic record interferes with nothing and is found only when
searched for. An adapter interferes with the other adapters and with the base
behaviour in its region. A weight update interferes with everything, and
cannot be taken back — so it needs not just high usefulness but high
confidence, because an error written there is an error forever.

The rule that falls out is short: write cheaply unless the thing will be
needed often enough, for long enough, and is certain enough, to be worth what
writing it deeply costs. That is the same shape as
:func:`core.cognition.value_of_computation.worth_learning`, and for the same
reason — a cost paid once against a benefit paid per use.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger("Aura.Memory.Allocation")


class Substrate(StrEnum):
    """Where a durable thing can live."""

    #: A record in a store. Retrieved by lookup, changed by writing again.
    SYMBOLIC = "symbolic"
    #: A low-rank adapter, loaded for the situations it is about.
    ADAPTER = "adapter"
    #: Folded into the model. Applies always and does not come back out.
    WEIGHTS = "weights"
    #: Not worth holding anywhere.
    DISCARD = "discard"

    @property
    def reversible(self) -> bool:
        return self is not Substrate.WEIGHTS

    @property
    def applies_unbidden(self) -> bool:
        """Whether it shapes behaviour without anything going to look for it."""
        return self in {Substrate.ADAPTER, Substrate.WEIGHTS}


#: What writing to each substrate costs, in the same unit as usefulness. These
#: are ratios rather than measurements — an adapter is roughly two orders of
#: magnitude more expensive to make than a store write, and a weight update
#: another order beyond that — and the ordering is what the decision turns on.
WRITE_COST: dict[Substrate, float] = {
    Substrate.SYMBOLIC: 1.0,
    Substrate.ADAPTER: 100.0,
    Substrate.WEIGHTS: 1000.0,
}

#: What writing to each substrate costs everything already there, per unit of
#: what is written. Symbolic records do not interfere; adapters interfere
#: within their region; weights interfere everywhere.
INTERFERENCE: dict[Substrate, float] = {
    Substrate.SYMBOLIC: 0.0,
    Substrate.ADAPTER: 0.15,
    Substrate.WEIGHTS: 1.0,
}

#: Confidence below which nothing goes anywhere irreversible. An error in the
#: weights is an error forever, so the bar is not the same bar as for a record
#: that can be rewritten.
CONFIDENCE_FOR_WEIGHTS = 0.95

#: And a lower one for adapters, which can at least be unloaded.
CONFIDENCE_FOR_ADAPTER = 0.7

#: Below this, holding it anywhere costs more than it returns.
MIN_VALUE_TO_HOLD = 0.0

#: How many adapters can be held before they start displacing each other.
#: Finite and contended is the whole reason the weights are ever the right
#: answer: without it, an adapter is strictly cheaper than a weight update at
#: every level of usefulness, the weights branch is unreachable, and the
#: system would keep making adapters until none of them could be loaded.
ADAPTER_CAPACITY = 16

#: How steeply the marginal adapter costs more as the capacity fills. Squared,
#: so the first few are nearly free and the last ones are not — which is what
#: contention for a fixed budget actually looks like.
CONTENTION_EXPONENT = 2.0


@dataclass(frozen=True)
class Shelf:
    """What is already held, which is what makes a new adapter cost more."""

    adapters_held: int = 0
    capacity: int = ADAPTER_CAPACITY

    @property
    def occupancy(self) -> float:
        return max(0.0, min(1.0, self.adapters_held / max(1, self.capacity)))

    @property
    def contention(self) -> float:
        """The multiplier on a new adapter's cost at this occupancy."""
        return 1.0 + (self.occupancy**CONTENTION_EXPONENT) * _CONTENTION_AT_FULL

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapters_held": self.adapters_held,
            "capacity": self.capacity,
            "occupancy": round(self.occupancy, 3),
            "contention": round(self.contention, 3),
        }


#: What a new adapter costs relative to an empty shelf, when the shelf is
#: full. Set so a full shelf makes an adapter comparable to a weight update:
#: at that point the question stops being "cheaper?" and becomes "is this one
#: of the few things worth keeping loaded forever?"
_CONTENTION_AT_FULL = 12.0


@dataclass(frozen=True)
class Item:
    """One thing that could be remembered, and what is known about it."""

    key: str
    #: How often it will be needed, per unit time.
    use_rate: float
    #: Over how long it will keep being needed, in the same time unit.
    lifetime: float
    #: How much each use is worth.
    value_per_use: float
    #: How sure we are it is true and will stay true, in [0, 1].
    confidence: float
    #: How much of what is already held this would disturb, in [0, 1]. A fact
    #: about a new topic disturbs little; a correction to something central
    #: disturbs a lot.
    breadth: float = 0.5
    #: True when a lookup would not happen — the thing has to apply without
    #: anyone asking for it, or it will not apply at all.
    needs_unbidden: bool = False

    @property
    def expected_uses(self) -> float:
        return max(0.0, self.use_rate * self.lifetime)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "use_rate": self.use_rate,
            "lifetime": self.lifetime,
            "value_per_use": self.value_per_use,
            "confidence": self.confidence,
            "breadth": self.breadth,
            "needs_unbidden": self.needs_unbidden,
            "expected_uses": round(self.expected_uses, 3),
        }


@dataclass(frozen=True)
class Allocation:
    """Where it should go, what that is worth, and why."""

    item: str
    substrate: Substrate
    value: float
    #: Every substrate's net value, so a reader can see what was close.
    considered: dict[str, float] = field(default_factory=dict)
    because: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "item": self.item,
            "substrate": str(self.substrate),
            "value": round(self.value, 4),
            "considered": {k: round(v, 4) for k, v in self.considered.items()},
            "because": self.because,
        }


def _net_value(item: Item, substrate: Substrate, shelf: Shelf) -> float:
    """What holding it here returns, net of writing and of interference.

    Confidence enters twice and it is not double-counting. It scales the
    benefit, because a thing that may be false returns less. And it scales the
    interference, because disturbing what is already there on the strength of
    something uncertain is the expensive mistake — the cost lands whether or
    not the new thing turns out to be true.
    """
    if substrate is Substrate.DISCARD:
        return 0.0
    benefit = item.expected_uses * item.value_per_use * item.confidence
    if substrate is Substrate.SYMBOLIC and item.needs_unbidden:
        # A record nothing goes looking for does not help. This is the case
        # where the cheap path is not merely worse, it is nothing.
        benefit = 0.0
    write = WRITE_COST[substrate]
    if substrate is Substrate.ADAPTER:
        # A new adapter on a crowded shelf displaces the ones already there.
        write *= shelf.contention
    interference = (
        INTERFERENCE[substrate] * item.breadth * write * (2.0 - item.confidence)
    )
    return benefit - write - interference


def allocate(item: Item, shelf: Shelf | None = None) -> Allocation:
    """Choose the substrate for one item, given what is already held."""
    held = shelf or Shelf()
    considered: dict[str, float] = {}
    for substrate in (Substrate.SYMBOLIC, Substrate.ADAPTER, Substrate.WEIGHTS):
        considered[str(substrate)] = _net_value(item, substrate, held)

    # Confidence gates come before value, not after. A thing that would repay
    # a weight update but might be wrong does not get one: the gate is about
    # what an error costs, and no amount of expected benefit changes that an
    # error in the weights cannot be taken back.
    eligible: list[Substrate] = [Substrate.SYMBOLIC]
    if item.confidence >= CONFIDENCE_FOR_ADAPTER:
        eligible.append(Substrate.ADAPTER)
    if item.confidence >= CONFIDENCE_FOR_WEIGHTS:
        eligible.append(Substrate.WEIGHTS)

    best = max(eligible, key=lambda s: (considered[str(s)], -WRITE_COST[s]))
    value = considered[str(best)]

    if value <= MIN_VALUE_TO_HOLD:
        if item.needs_unbidden and Substrate.ADAPTER not in eligible:
            return Allocation(
                item=item.key,
                substrate=Substrate.DISCARD,
                value=value,
                considered=considered,
                because=(
                    f"it has to apply without being asked for, and at "
                    f"{item.confidence:.2f} confidence it may not go anywhere "
                    f"that does (an adapter needs {CONFIDENCE_FOR_ADAPTER})"
                ),
            )
        return Allocation(
            item=item.key,
            substrate=Substrate.DISCARD,
            value=value,
            considered=considered,
            because=(
                f"{item.expected_uses:.1f} expected uses at "
                f"{item.value_per_use:.2f} does not repay holding it anywhere"
            ),
        )

    return Allocation(
        item=item.key,
        substrate=best,
        value=value,
        considered=considered,
        because=_explain(item, best, considered, held),
    )


def _explain(
    item: Item, chosen: Substrate, considered: dict[str, float], shelf: Shelf
) -> str:
    if chosen is Substrate.SYMBOLIC:
        if item.confidence < CONFIDENCE_FOR_ADAPTER:
            return (
                f"at {item.confidence:.2f} confidence it may not go anywhere "
                "that cannot be rewritten"
            )
        return (
            f"{item.expected_uses:.1f} expected uses does not repay "
            f"{WRITE_COST[Substrate.ADAPTER]:.0f} to make an adapter and "
            f"{INTERFERENCE[Substrate.ADAPTER]:.2f} interference with the rest"
        )
    if chosen is Substrate.ADAPTER:
        weights_value = considered[str(Substrate.WEIGHTS)]
        return (
            f"{item.expected_uses:.1f} expected uses repays an adapter; the "
            f"weights would return {weights_value:+.1f} and cannot be undone"
        )
    return (
        f"{item.expected_uses:.1f} expected uses at {item.confidence:.2f} "
        f"confidence repays folding it in; the adapter shelf is "
        f"{shelf.occupancy:.0%} full, so a new one would cost "
        f"{shelf.contention:.1f} times what it does on an empty shelf"
    )


def allocate_all(items: list[Item], shelf: Shelf | None = None) -> tuple[Allocation, ...]:
    """Allocate a batch, with the shelf filling as adapters are chosen.

    Sequential rather than independent, because each adapter chosen makes the
    next one cost more. Allocating a batch as though the shelf never filled is
    how a system ends up with more adapters than it can load.
    """
    held = shelf or Shelf()
    out: list[Allocation] = []
    for item in items:
        allocation = allocate(item, held)
        out.append(allocation)
        if allocation.substrate is Substrate.ADAPTER:
            held = Shelf(adapters_held=held.adapters_held + 1, capacity=held.capacity)
    return tuple(out)


def capacity_report(allocations: tuple[Allocation, ...]) -> dict[str, Any]:
    """What a batch of decisions would cost, by substrate."""
    counts: dict[str, int] = {}
    cost = 0.0
    for allocation in allocations:
        counts[str(allocation.substrate)] = counts.get(str(allocation.substrate), 0) + 1
        if allocation.substrate is not Substrate.DISCARD:
            cost += WRITE_COST[allocation.substrate]
    return {
        "counts": counts,
        "write_cost": round(cost, 2),
        "irreversible": counts.get(str(Substrate.WEIGHTS), 0),
    }


__all__ = [
    "ADAPTER_CAPACITY",
    "CONFIDENCE_FOR_ADAPTER",
    "CONFIDENCE_FOR_WEIGHTS",
    "INTERFERENCE",
    "WRITE_COST",
    "Allocation",
    "Item",
    "Shelf",
    "Substrate",
    "allocate",
    "allocate_all",
    "capacity_report",
]
