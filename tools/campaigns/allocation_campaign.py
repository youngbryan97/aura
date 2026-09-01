#!/usr/bin/env python3
"""Does deciding where to spend beat spending everywhere, at the same total?

  035  adaptive allocation beats static frequencies at equal compute
  039  search keeps high-value branches better than fixed depth at equal
       compute

Both cards make the same demand and it is the demand that matters: EQUAL
COMPUTE. A scheduler that thinks harder and does better has proved nothing.
Every arm here is given the same number of units to spend and differs only in
where it puts them, so the comparison is about the policy and not the budget.

Card 035's world is a set of methods with different, initially unknown returns.
The static arms run a fixed rotation — the fixed cadences a scheduler
replaces. The adaptive arm uses core.cognition.cognitive_cost's
ValueOfComputation: try each once, then spend on measured return.

Card 039's world is a search tree whose branches have different values. The
fixed-depth arm expands every branch to the same depth. The value-guided arm
spends the same total expansions, putting them where the branch has been
paying. What is scored is the best leaf found, at identical expansion counts.

Both include the arm that is easy to forget: an ORACLE that knows the returns
in advance, so the result reads as a fraction of what was available rather
than as a bare win.

    python tools/campaigns/allocation_campaign.py --trials 200
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.cognition.cognitive_cost import ValueOfComputation  # noqa: E402


# ── 035: where to spend, when the returns are not known in advance ────────


def allocation_trial(
    *, methods: int, budget: int, rng: random.Random
) -> dict[str, float]:
    """One task: a budget of units, several methods, returns unknown."""
    # Returns are drawn per task, so a policy cannot memorise them across
    # tasks and the comparison is about learning within the budget.
    returns = [rng.uniform(0.0, 1.0) for _ in range(methods)]
    noise = 0.25

    def _run(index: int) -> float:
        return max(0.0, returns[index] + rng.gauss(0.0, noise))

    # Adaptive: one bounded trial each, then spend on measured return.
    voc = ValueOfComputation()
    adaptive_total = 0.0
    spent = 0
    for index in range(min(methods, budget)):
        gain = _run(index)
        voc.observe(f"m{index}", cost=1.0, gain=gain)
        adaptive_total += gain
        spent += 1
    while spent < budget:
        ranked = voc.rank([f"m{i}" for i in range(methods)])
        best = int(ranked[0]["method"][1:])
        gain = _run(best)
        voc.observe(f"m{best}", cost=1.0, gain=gain)
        adaptive_total += gain
        spent += 1

    # Static rotation: the fixed cadence, every method equally often.
    rotation_total = sum(_run(i % methods) for i in range(budget))

    # Static single: commit the whole budget to one method, chosen blind.
    committed = rng.randrange(methods)
    single_total = sum(_run(committed) for _ in range(budget))

    # Oracle: knows the returns, spends everything on the best.
    best_known = max(range(methods), key=lambda i: returns[i])
    oracle_total = sum(_run(best_known) for _ in range(budget))

    return {
        "adaptive": adaptive_total,
        "static_rotation": rotation_total,
        "static_single": single_total,
        "oracle": oracle_total,
        "units_spent": float(budget),
    }


# ── 039: where to look, when every branch costs the same to open ──────────


#: How fast a branch's payoff saturates with depth. Without saturation deeper
#: is always better, fixed depth is capped by construction, and the guided arm
#: wins for a reason that is in the world rather than in the policy.
_SATURATION_DEPTH = 4.0


def _branch_value(branch: int, depth: int, rng: random.Random) -> float:
    """A branch's payoff at a depth, with diminishing returns.

    Deepening a good branch pays less each time, so the guided arm cannot win
    by digging: it has to find the good branch and then stop digging the bad
    ones. That is the trade fixed depth cannot make and this measures.
    """
    quality = _BRANCH_QUALITY[branch]
    return quality * (1.0 - math.exp(-depth / _SATURATION_DEPTH)) + rng.gauss(0.0, 0.05)


_BRANCH_QUALITY: list[float] = []


def search_trial(
    *, branches: int, expansions: int, rng: random.Random
) -> dict[str, float]:
    """One search: a fixed number of expansions, spent two ways and an oracle."""
    global _BRANCH_QUALITY
    _BRANCH_QUALITY = [rng.uniform(0.0, 1.0) for _ in range(branches)]

    # Fixed depth: every branch opened to the same depth.
    per_branch = max(1, expansions // branches)
    fixed_best = max(
        _branch_value(b, d, rng)
        for b in range(branches)
        for d in range(1, per_branch + 1)
    )

    # Value-guided: the same total expansions, put where the branch is paying.
    voc = ValueOfComputation()
    depths = [0] * branches
    guided_best = float("-inf")
    spent = 0
    for branch in range(min(branches, expansions)):
        depths[branch] += 1
        value = _branch_value(branch, depths[branch], rng)
        guided_best = max(guided_best, value)
        voc.observe(f"b{branch}", cost=1.0, gain=max(0.0, value))
        spent += 1
    while spent < expansions:
        ranked = voc.rank([f"b{i}" for i in range(branches)])
        branch = int(ranked[0]["method"][1:])
        depths[branch] += 1
        value = _branch_value(branch, depths[branch], rng)
        guided_best = max(guided_best, value)
        voc.observe(f"b{branch}", cost=1.0, gain=max(0.0, value))
        spent += 1

    # Oracle: knows which branch is best, spends every expansion on it.
    best_branch = max(range(branches), key=lambda b: _BRANCH_QUALITY[b])
    oracle_best = max(
        _branch_value(best_branch, d, rng) for d in range(1, expansions + 1)
    )

    return {
        "fixed_depth": fixed_best,
        "value_guided": guided_best,
        "oracle": oracle_best,
        "expansions": float(expansions),
        "deepest_reached": float(max(depths)),
        "fixed_depth_reached": float(per_branch),
    }


def _summarise(rows: list[dict[str, float]], arms: tuple[str, ...]) -> dict:
    oracle = statistics.fmean(r["oracle"] for r in rows)
    out = {}
    for arm in arms:
        mean = statistics.fmean(r[arm] for r in rows)
        out[arm] = {
            "mean": round(mean, 4),
            "share_of_oracle": round(mean / oracle, 4) if oracle else 0.0,
        }
    out["oracle"] = {"mean": round(oracle, 4), "share_of_oracle": 1.0}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--methods", type=int, default=6)
    parser.add_argument("--budget", type=int, default=24)
    parser.add_argument("--branches", type=int, default=8)
    parser.add_argument("--expansions", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1_618_033)
    parser.add_argument("--out", default="docs/evidence/allocation_campaign.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    allocation = [
        allocation_trial(methods=args.methods, budget=args.budget, rng=rng)
        for _ in range(args.trials)
    ]
    search = [
        search_trial(branches=args.branches, expansions=args.expansions, rng=rng)
        for _ in range(args.trials)
    ]

    payload = {
        "schema": "aura.allocation_campaign.v1",
        "cards": ["035", "039"],
        "claim_boundary": (
            "synthetic method-return and search-tree worlds at a fixed unit "
            "budget; measures the allocation policy in "
            "core.cognition.cognitive_cost, not Aura's scheduling of any real "
            "cognitive work"
        ),
        "config": {
            "trials": args.trials,
            "methods": args.methods,
            "budget": args.budget,
            "branches": args.branches,
            "expansions": args.expansions,
            "seed": args.seed,
        },
        "equal_compute": {
            "allocation_units_per_arm": args.budget,
            "search_expansions_per_arm": args.expansions,
            "basis": "every arm spends the same number of units; only where differs",
        },
        "allocation": _summarise(
            allocation, ("adaptive", "static_rotation", "static_single")
        )
        | {
            "adaptive_beats_rotation": sum(
                1 for r in allocation if r["adaptive"] > r["static_rotation"]
            ),
            "adaptive_beats_single": sum(
                1 for r in allocation if r["adaptive"] > r["static_single"]
            ),
            "of": len(allocation),
        },
        "search": _summarise(search, ("value_guided", "fixed_depth"))
        | {
            "value_guided_beats_fixed_depth": sum(
                1 for r in search if r["value_guided"] > r["fixed_depth"]
            ),
            "of": len(search),
            "median_deepest_reached": statistics.median(
                r["deepest_reached"] for r in search
            ),
            "fixed_depth_reached": search[0]["fixed_depth_reached"],
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    out = ROOT / args.out
    with local_internal_governed_scope("allocation_campaign"):
        get_file_write_gateway().write_text(
            out, json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
