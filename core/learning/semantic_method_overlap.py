"""Paired outcome and agreement analysis for frozen semantic method proposals."""

from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict


METHODS = ("incumbent", "ranker", "direct", "prototype")


def _wilson(successes: int, population: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if population < 1 or not 0 <= successes <= population:
        raise ValueError("Wilson interval needs a nonempty counted population")
    fraction = successes / population
    denominator = 1 + z * z / population
    center = (fraction + z * z / (2 * population)) / denominator
    half = (z / denominator * math.sqrt(
        fraction * (1 - fraction) / population + z * z / (4 * population * population)))
    return max(0., center - half), min(1., center + half)


def _mcnemar_exact(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.
    tail = sum(math.comb(discordant, index) for index in range(min(left_only, right_only) + 1))
    return min(1., 2. * tail / (2 ** discordant))


def analyze_method_overlap(rows: list[dict]) -> dict:
    """Grade fixed proposals; no labels enter a policy or alter any choice."""
    if not rows or len({row["source"] for row in rows}) != len(rows):
        raise ValueError("overlap needs distinct source identities")
    by_construction: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if (not row.get("construction") or set(row["methods"]) != set(METHODS)
                or any(type(row["methods"][name]["correct"]) is not bool
                       or not isinstance(row["methods"][name]["program_sha256"], (str, type(None)))
                       for name in METHODS)):
            raise ValueError("overlap rows need four bound method outcomes")
        by_construction[row["construction"]].append(row)
    population = len(rows)
    successes = {name: sum(row["methods"][name]["correct"] for row in rows)
                 for name in METHODS}
    unique = {name: sum(row["methods"][name]["correct"] and not any(
        row["methods"][other]["correct"] for other in METHODS if other != name)
        for row in rows) for name in METHODS}
    pairwise = {}
    for left, right in itertools.combinations(METHODS, 2):
        both = sum(row["methods"][left]["correct"] and row["methods"][right]["correct"]
                   for row in rows)
        left_only = sum(row["methods"][left]["correct"]
                        and not row["methods"][right]["correct"] for row in rows)
        right_only = sum(not row["methods"][left]["correct"]
                         and row["methods"][right]["correct"] for row in rows)
        agreeing = [row for row in rows if row["methods"][left]["program_sha256"] is not None
                    and row["methods"][left]["program_sha256"]
                    == row["methods"][right]["program_sha256"]]
        pairwise[f"{left}:{right}"] = {
            "both_correct": both, "left_only": left_only, "right_only": right_only,
            "both_wrong": population - both - left_only - right_only,
            "mcnemar_exact_p": _mcnemar_exact(left_only, right_only),
            "program_agreements": len(agreeing),
            "correct_when_agree": sum(row["methods"][left]["correct"] for row in agreeing),
        }
    consensus_covered = consensus_correct = 0
    for row in rows:
        votes = Counter(outcome["program_sha256"] for outcome in row["methods"].values()
                        if outcome["program_sha256"] is not None)
        if not votes:
            continue
        program, count = votes.most_common(1)[0]
        if count < 2 or sum(n == count for n in votes.values()) != 1:
            continue
        consensus_covered += 1
        consensus_correct += int(next(outcome["correct"] for outcome in row["methods"].values()
                                      if outcome["program_sha256"] == program))
    return {
        "population": population,
        "correct": successes,
        "correct_wilson_95": {name: _wilson(count, population)
                              for name, count in successes.items()},
        "unique_successes": unique,
        "oracle_union": sum(any(row["methods"][name]["correct"] for name in METHODS)
                            for row in rows),
        "all_wrong": sum(not any(row["methods"][name]["correct"] for name in METHODS)
                         for row in rows),
        "pairwise": pairwise,
        "consensus_covered": consensus_covered,
        "consensus_correct": consensus_correct,
        "consensus_abstained": population - consensus_covered,
        "by_construction": {name: {
            "population": len(group),
            "correct": {method: sum(row["methods"][method]["correct"] for row in group)
                        for method in METHODS},
        } for name, group in sorted(by_construction.items())},
    }
