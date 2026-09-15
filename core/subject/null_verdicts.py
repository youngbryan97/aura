"""Whether each null passes the conjunction, under ISC-v1, ISC-v2 and ISC-v3.

The rules read a null table as the campaign records it in `nulls.json`: one row
per system, an architecture row carrying the graph, closure, differentiation
and synergy readings, a surrogate row carrying irreducibility alone. They are
pure functions over those rows, so a verdict can be recomputed from any run's
recorded table and compared with what that run reported.

ISC-v1's verdict is `passes_the_conjunction`. ISC-v2 changes two things, both
preregistered in docs/ISC_V2_PREREGISTRATION.md before any v2 result: which
nulls irreducibility is compared against, and that a null's verdict is read
across the campaign's declared seeds rather than on one. ISC-v3 judges every
null by its own synergy line in both of those places
(docs/ISC_V3_PREREGISTRATION.md).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from core.subject.battery import THRESHOLDS

__all__ = [
    "REFERENCE",
    "beats_the_comparison_set",
    "beats_the_v3_comparison_set",
    "comparison_set",
    "comparison_set_v3",
    "passes_all_but_irreducibility",
    "passes_the_conjunction",
    "passes_the_v2_conjunction",
    "passes_the_v3_conjunction",
    "v2_conjunctions",
    "v3_conjunctions",
    "v3_readable",
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


def _synergy_v2_holds(row: Mapping[str, Any]) -> bool:
    """Synergy on the target's change, where the row recorded it; v1's reading otherwise."""
    passes = row.get("synergy_v2_passes")
    if passes is None:
        return _synergy_holds(row)
    return all(bool(item) for item in passes) if passes else True


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


def _passes_v2_all_but_irreducibility(row: Mapping[str, Any]) -> bool:
    """v1's other lines with synergy read on the change, where the row recorded it."""
    return passes_all_but_irreducibility({**row, "synergy_passes": None}) and _synergy_v2_holds(row)


def comparison_set(table: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    """ISC-v2: the nulls a real irreducibility score is compared against.

    Every matched surrogate, and every architecture other than the reference
    that passes the rest of the v2 conjunction. A null already told apart from a
    subject by another line is not asked about again with the harder number.

    Until the fourth amendment to docs/ISC_V2_PREREGISTRATION.md this read the
    rest of v1's conjunction, which judges synergy on the level.
    """
    return {
        name: float(row.get("phi_do", 0.0))
        for name, row in table.items()
        if name != REFERENCE
        and (row.get("kind") == "surrogate" or _passes_v2_all_but_irreducibility(row))
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
    return beats and _passes_v2_all_but_irreducibility(row)


def v2_conjunctions(table: Mapping[str, Mapping[str, Any]]) -> dict[str, bool]:
    """ISC-v2's conjunction for every system in one seed's recorded null table.

    This is what section 3 reads across the declared seeds. A surrogate is
    judged on irreducibility alone and never passes a conjunction.
    """
    return {name: passes_the_v2_conjunction(name, table) for name in table}


def v3_readable(table: Mapping[str, Mapping[str, Any]]) -> bool:
    """Whether every architecture in a recorded null table carries ISC-v3's synergy reading.

    v3 judges each null by the synergy line it judges Aura by, and a table
    recorded without that reading cannot be read under v3 at all.
    """
    return bool(table) and all(
        row.get("synergy_v3_passes") is not None
        for row in table.values()
        if row.get("kind") == "architecture"
    )


def _passes_v3_all_but_irreducibility(row: Mapping[str, Any]) -> bool:
    """v1's other lines with synergy read by ISC-v3's line; an architecture without the reading fails."""
    if row.get("kind") != "architecture":
        return True
    passes = row.get("synergy_v3_passes")
    if passes is None:
        return False
    return passes_all_but_irreducibility({**row, "synergy_passes": None}) and all(bool(item) for item in passes)


def comparison_set_v3(table: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    """ISC-v3: section 1's comparison set, with every architecture judged by the v3 synergy line."""
    return {
        name: float(row.get("phi_do", 0.0))
        for name, row in table.items()
        if name != REFERENCE
        and (row.get("kind") == "surrogate" or _passes_v3_all_but_irreducibility(row))
    }


def beats_the_v3_comparison_set(
    lower_bound: float, table: Mapping[str, Mapping[str, Any]]
) -> tuple[bool, dict[str, float]]:
    """ISC-v3: whether a lower bound clears the v3 comparison set, and never on a table without the reading."""
    if not v3_readable(table):
        return False, {}
    compared = comparison_set_v3(table)
    if not compared:
        return False, compared
    return all(lower_bound > value for value in compared.values()), compared


def passes_the_v3_conjunction(name: str, table: Mapping[str, Mapping[str, Any]]) -> bool:
    """ISC-v3 on one seed: the v2 conjunction with this system's synergy and the comparison set read by v3's line."""
    row = table[name]
    if row.get("kind") != "architecture" or not v3_readable(table):
        return False
    others = {key: value for key, value in table.items() if key != name}
    beats, _ = beats_the_v3_comparison_set(float(row.get("phi_do", 0.0)), others)
    return beats and _passes_v3_all_but_irreducibility(row)


def v3_conjunctions(table: Mapping[str, Mapping[str, Any]]) -> dict[str, bool]:
    """ISC-v3's conjunction for every system in one seed's recorded table, or nothing without the reading."""
    if not v3_readable(table):
        return {}
    return {name: passes_the_v3_conjunction(name, table) for name in table}


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
