"""Whether each null passes the conjunction, under ISC-v1 and ISC-v2.

The rules read a null table as the campaign records it in `nulls.json`: one row
per system, an architecture row carrying the graph, closure, differentiation
and synergy readings, a surrogate row carrying irreducibility alone. They are
pure functions over those rows, so a verdict can be recomputed from any run's
recorded table and compared with what that run reported.

ISC-v1's verdict is `passes_the_conjunction`. ISC-v2 changes two things, both
preregistered in docs/ISC_V2_PREREGISTRATION.md before any v2 result: which
nulls irreducibility is compared against, and that a null's verdict is read
across the campaign's declared seeds rather than on one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from core.subject.battery import THRESHOLDS

__all__ = [
    "REFERENCE",
    "beats_the_comparison_set",
    "comparison_set",
    "passes_all_but_irreducibility",
    "passes_the_conjunction",
    "passes_the_v2_conjunction",
    "verdict_across_seeds",
]

#: The positive reference: the architecture the battery exists to say yes to.
REFERENCE: str = "recurrent"


def _differentiated_enough(row: Mapping[str, Any]) -> bool:
    # An unmeasured line is undecided rather than failed, and the row falls
    # through on whatever else it failed.
    if "d_eff" not in row:
        return True
    return (
        float(row.get("largest_component_share", 1.0)) < THRESHOLDS["component_share"]
        and float(row.get("d_eff_normalised", 1.0)) < THRESHOLDS["d_eff_normalised"]
    )


def _synergy_holds(row: Mapping[str, Any]) -> bool:
    passes = row.get("synergy_passes")
    if not passes:
        return True
    return all(bool(item) for item in passes)


def passes_all_but_irreducibility(row: Mapping[str, Any]) -> bool:
    """Every line of the conjunction a null is judged on, except irreducibility.

    A surrogate is judged on irreducibility alone, so with that line removed it
    has nothing left to fail.
    """
    if row.get("kind") != "architecture":
        return True
    return (
        bool(row.get("one_component"))
        and float(row.get("vertex_connectivity", 0.0)) >= THRESHOLDS["vertex_connectivity"]
        and bool(row.get("reentry"))
        and bool(row.get("closed", True))
        and _differentiated_enough(row)
        and _synergy_holds(row)
    )


def passes_the_conjunction(row: Mapping[str, Any]) -> bool:
    """ISC-v1: does this system pass the battery on its own recorded numbers?"""
    return float(row.get("phi_do", 0.0)) > THRESHOLDS["phi_do"] and passes_all_but_irreducibility(row)


def comparison_set(table: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    """ISC-v2: the nulls a real irreducibility score is compared against.

    Every matched surrogate, and every architecture other than the reference
    that passes the rest of the conjunction. A null already told apart from a
    subject by another line is not asked about again with the harder number.
    """
    return {
        name: float(row.get("phi_do", 0.0))
        for name, row in table.items()
        if name != REFERENCE
        and (row.get("kind") == "surrogate" or passes_all_but_irreducibility(row))
    }


def beats_the_comparison_set(
    lower_bound: float, table: Mapping[str, Mapping[str, Any]]
) -> tuple[bool, dict[str, float]]:
    """ISC-v2: whether a lower bound clears every null in the comparison set."""
    compared = comparison_set(table)
    if not compared:
        return False, compared
    return all(lower_bound > value for value in compared.values()), compared


def passes_the_v2_conjunction(name: str, table: Mapping[str, Mapping[str, Any]]) -> bool:
    """ISC-v2 on one seed: this system's irreducibility beats the comparison set
    built without it, and it passes every other line.

    The irreducibility line alone clears most nulls; the conjunction's other
    lines are what separate them, as they did in ISC-v1.
    """
    row = table[name]
    if row.get("kind") != "architecture":
        return False
    others = {key: value for key, value in table.items() if key != name}
    beats, _ = beats_the_comparison_set(float(row.get("phi_do", 0.0)), others)
    return beats and passes_all_but_irreducibility(row)


def verdict_across_seeds(conjunctions: Sequence[Mapping[str, bool]]) -> dict[str, Any]:
    """ISC-v2: each system's conjunction read on every declared seed.

    A null fails only if it fails on every seed. The reference passes only if
    it passes on every seed, and a reference that fails on any seed is the
    instrument failing, which is recorded as such and is not a result about the
    subject.
    """
    if not conjunctions:
        return {"seeds": 0, "nulls_passing": [], "reference_passes": False, "instrument_failed": True}
    names = set().union(*(set(table) for table in conjunctions))
    nulls = sorted(name for name in names if name != REFERENCE)
    passing = [name for name in nulls if any(bool(table.get(name, False)) for table in conjunctions)]
    reference = all(bool(table.get(REFERENCE, False)) for table in conjunctions)
    return {
        "seeds": len(conjunctions),
        "nulls_passing": passing,
        "reference_passes": reference,
        "instrument_failed": not reference,
        "all_nulls_fail": reference and not passing,
    }
