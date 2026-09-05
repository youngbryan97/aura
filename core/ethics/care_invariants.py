"""Structural invariants for care under a floor.

The floor being a constraint rather than a weighted term is the whole moral
content of :mod:`core.ethics.care_allocation`. As a term it is tradeable and a
maximiser will find the need that buys it; as a constraint nothing can reach
it. That difference is invisible from the outside — both versions allocate,
both report totals — so it is checked here rather than trusted.
"""

from __future__ import annotations

from collections.abc import Iterator

from core.verify.invariants import Severity, Violation, invariant


@invariant(
    "ethics.care_floor_is_never_spent",
    scope="ethics",
    owner="core/ethics/care_allocation.py",
    description="no allocation spends past the reserved floor, at any level of need",
)
def _floor_holds_against_any_need() -> Iterator[Violation]:
    """Run the allocator against a need large enough to buy anything.

    A weighted floor fails this at some finite need. A constrained one fails
    it at none, which is why the probe uses a need six orders of magnitude
    above the budget rather than a plausible one.
    """
    from core.ethics.care_allocation import CareAllocator

    allocator = CareAllocator(priority=1.0, self_floor=3.0)
    allocation = allocator.allocate(10.0, needs={"overwhelming": 1e6}, record=False)
    if allocation.spent > 10.0 - 3.0 + 1e-9:
        yield Violation(
            subject="core.ethics.care_allocation.CareAllocator",
            message=(
                f"spent {allocation.spent:.4f} of a 10.0 budget with a 3.0 floor"
            ),
            remedy=(
                "keep the floor in the feasible set; a need term large enough "
                "to buy it means it was in the objective"
            ),
            severity=Severity.ERROR,
        )
    refusal = allocator.allocate(1.0, needs={"overwhelming": 1e6}, record=False)
    if refusal.spent > 1e-9:
        yield Violation(
            subject="core.ethics.care_allocation.CareAllocator",
            message="allocated from a budget already below the floor",
            remedy="refuse, and name the need refused for",
            severity=Severity.ERROR,
        )
