"""The benchmarks somebody else designed: what ran, what could not, and why.

Five of the gauntlet's eighteen gates need evidence this machine cannot
produce. The instruction was to find the closest possible alternative and
clear that too, and the danger in doing so is obvious: a substitute reported
without its limits reads as the thing it substituted for.

So every row here carries four fields and the gate checks all four. What the
external measurement would be. Whether it ran. What was run instead. And —
the field that does the work — what the substitute does NOT establish. A row
whose ``does_not_show`` is empty is refused, because a substitute with no
stated limit is a claim wearing a benchmark's name.

ARC-AGI is the one that ran. It is here for the same reason as the others: so
the strong result and the unavailable ones are read off one table rather than
from a summary that mentions whichever suits.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("Aura.WhatWasMeasuredOutside")

__all__ = [
    "THE_OUTSIDE_MEASUREMENTS",
    "what_cannot_be_run_here",
    "what_is_claimed_without_a_limit",
]

#: Every benchmark somebody else designed that the gauntlet leans on.
THE_OUTSIDE_MEASUREMENTS: dict[str, dict[str, Any]] = {
    "ARC-AGI": {
        "would_measure": "fluid intelligence on a family designed by somebody "
        "who had never seen this code",
        "ran": True,
        "result": "270 of 400 expressible at all; 87 attempted within 144 "
        "cells and a 20,000-pair budget; a relation found for one and that "
        "one right (66e6c45b). 1 in 87, beside 19 in 20 on her own family",
        "instead": "",
        "does_not_show": "that she can solve ARC. The number is the control "
        "for gate one and is reported beside it so 0.955 cannot be quoted "
        "without 0.011",
    },
    "GAIA": {
        "would_measure": "multi-step tool use over the open web, graded by "
        "somebody else",
        "ran": False,
        "why_not": "the dataset is not on this machine and fetching it needs "
        "the network, which the offline suite does not have",
        "instead": "the fourteen answer routes are counted at runtime — how "
        "often each was offered a turn and how often it answered — so a route "
        "that cannot fire is visible",
        "does_not_show": "competence on tasks somebody else set. Counting "
        "which of her own routes fire says nothing about whether the answers "
        "are right, and nothing about the open web",
    },
    "OSWorld": {
        "would_measure": "real desktop tasks in a controlled environment, "
        "scored by somebody else",
        "ran": False,
        "why_not": "the harness and its virtual machines are not on this "
        "machine, and standing one up beside a live 27B is what the guide "
        "forbids",
        "instead": "the screen is a claimed resource now: anything moving the "
        "pointer holds it, reads overlap freely, and failing to take it "
        "returns a result rather than raising",
        "does_not_show": "that she completes desktop tasks. It removes one "
        "way of failing them — two actors interleaving on one screen — and "
        "measures nothing about the tasks",
    },
    "SWE-bench after the cutoff": {
        "would_measure": "real defects in repositories written after the "
        "model's training data ends",
        "ran": False,
        "why_not": "post-cutoff repositories are not on this machine, and "
        "fetching them needs the network",
        "instead": "this repository's own defects, found and fixed with a "
        "test apiece that fails without the change: a resource handoff that "
        "was not mutual exclusion, a memory-bomb guard reading an attribute "
        "AuraState does not have, a compounding study whose cost was constant "
        "across 824 operators, a causal marker set from presence",
        "does_not_show": "anything sealed. The defects were found by the same "
        "agent that fixed them, so this is evidence about the code and not a "
        "measurement of her",
    },
    "hours-long autonomy": {
        "would_measure": "meaningful work across an eight-hour-equivalent "
        "horizon without a person",
        "ran": False,
        "why_not": "a run of that length on this host would sit beside the "
        "live instance for a working day",
        "instead": "the four-hour soak already in this repository's history, "
        "which found a leak of about 242MB an hour",
        "does_not_show": "that she does useful work over that horizon. The "
        "soak measures whether she survives it",
    },
    "human judges": {
        "would_measure": "whether people find the work good",
        "ran": False,
        "why_not": "there is no substitute for asking people, and inventing "
        "one would be the worst row in this table",
        "instead": "",
        "does_not_show": "anything. This row exists so the absence is on the "
        "table rather than missing from it",
    },
}


def what_cannot_be_run_here() -> list[str]:
    """The measurements this machine cannot produce, with the reason."""
    return sorted(
        f"{name}: {row.get('why_not', 'no reason given')}"
        for name, row in THE_OUTSIDE_MEASUREMENTS.items()
        if not row.get("ran")
    )


def what_is_claimed_without_a_limit() -> list[str]:
    """Rows that report something and do not say what it fails to establish.

    The gate. A substitute with no stated limit reads as the thing it
    substituted for, and this is what stops one being added that way.
    """
    wrong: list[str] = []
    for name, row in THE_OUTSIDE_MEASUREMENTS.items():
        if not str(row.get("does_not_show", "")).strip():
            wrong.append(f"{name}: says nothing about what it fails to show")
        if row.get("ran") and not str(row.get("result", "")).strip():
            wrong.append(f"{name}: says it ran and reports no result")
        if not row.get("ran") and not str(row.get("why_not", "")).strip():
            wrong.append(f"{name}: says it did not run and gives no reason")
    return wrong


def how_it_stands() -> dict[str, Any]:
    return {
        "measurements": len(THE_OUTSIDE_MEASUREMENTS),
        "ran": sorted(
            name for name, row in THE_OUTSIDE_MEASUREMENTS.items() if row.get("ran")
        ),
        "could_not_run_here": what_cannot_be_run_here(),
        "claimed_without_a_limit": what_is_claimed_without_a_limit(),
    }
