#!/usr/bin/env python3
"""Does ECAN find what matters faster than scanning, chance, or degree?

Gap Atlas card 101 asks for an attention campaign on a large learned graph,
scored against scan, random and top-degree. Attention is only worth its
bookkeeping if a fixed budget of reads finds more of what a task needs than
the same budget spent three cheaper ways, so all four are given the same
budget and scored on the same hidden relevant set.

The graph is built with community structure and a planted relevant set: a
task touches a handful of seed atoms, and what the task actually needs is
their community. That is the shape attention claims to exploit — relevance is
correlated with structure, and spreading follows structure.

Scored on recall@budget over repeated trials at equal compute. The null that
matters is breadth-first from the same seeds: it touches the same graph, is
given the same number of atom-touches ECAN spent, and needs no economy, no
STI, no rent and no fund. If plain traversal finds the community just as
well, the attention economy is bookkeeping.

    python tools/campaigns/ecan_stress.py --atoms 40000 --trials 40
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.knowledge.atomspace_attention import (  # noqa: E402
    neighbours,
    reset_attention,
    spread_importance_touches,
)
from core.knowledge.atomspace import (  # noqa: E402
    EVALUATION,
    LIST,
    AtomSpace,
    Link,
    Node,
    TruthValue,
)

CONCEPT = "ConceptNode"
PREDICATE = "PredicateNode"


def build(
    *, atoms: int, communities: int, cross_rate: float, seed: int
) -> tuple[AtomSpace, list[list[Node]], dict[Node, int]]:
    """A graph with communities, and the degree of every concept in it."""
    rng = random.Random(seed)
    space = AtomSpace(max_atoms=atoms * 8, focus_size=64)
    related = Node(PREDICATE, "related")

    per = max(2, atoms // communities)
    groups: list[list[Node]] = []
    for c in range(communities):
        group = [Node(CONCEPT, f"c{c}_{i}") for i in range(per)]
        for node in group:
            space.add(node, TruthValue(0.9, 12.0), source="ecan")
        groups.append(group)

    degree: dict[Node, int] = dict.fromkeys(
        (n for g in groups for n in g), 0
    )
    edges = atoms * 2
    for _ in range(edges):
        g = groups[rng.randrange(len(groups))]
        a = g[rng.randrange(len(g))]
        if rng.random() < cross_rate:
            other = groups[rng.randrange(len(groups))]
            b = other[rng.randrange(len(other))]
        else:
            b = g[rng.randrange(len(g))]
        if a == b:
            continue
        space.add(
            Link(EVALUATION, (related, Link(LIST, (a, b)))),
            TruthValue(0.8, 8.0),
            source="ecan",
        )
        degree[a] += 1
        degree[b] += 1
    return space, groups, degree


def _breadth_first(
    space: AtomSpace, seeds: list[Node], *, touches: int, budget: int
) -> list[Node]:
    """The same seeds, the same touch budget, no economy at all.

    This is the arm that makes the comparison about attention rather than
    about knowing where to start: it is handed the seeds ECAN was paid on and
    stops at the number of atoms ECAN's spreading touched.
    """
    seen: set[Node] = set()
    order: list[Node] = []
    frontier = list(dict.fromkeys(seeds))
    spent = 0
    while frontier and spent < touches:
        node = frontier.pop(0)
        if node in seen:
            continue
        seen.add(node)
        spent += 1
        if isinstance(node, Node) and node.atype == CONCEPT:
            order.append(node)
        for neighbour in neighbours(space, node):
            if neighbour not in seen:
                frontier.append(neighbour)
    return order[:budget]


def _recall(picked: list[Node], relevant: set[Node]) -> float:
    if not relevant:
        return 0.0
    return len(set(picked) & relevant) / len(relevant)


def one_trial(
    space: AtomSpace,
    groups: list[list[Node]],
    degree: dict[Node, int],
    *,
    budget: int,
    seeds: int,
    ticks: int,
    rng: random.Random,
    all_concepts: list[Node],
    by_degree: list[Node],
) -> dict[str, float]:
    """One task: seed atoms in a community, then spend a budget four ways."""
    group = groups[rng.randrange(len(groups))]
    relevant = set(group)
    seeded = [group[rng.randrange(len(group))] for _ in range(seeds)]

    # ECAN: pay the seeds, run the economy, read the focus.
    reset_attention(space)
    for node in seeded:
        space.stimulate(node)
    began = time.perf_counter()
    touched = 0
    for _ in range(ticks):
        touched += spread_importance_touches(space)
        space.collect_rent()
    ecan_s = time.perf_counter() - began
    focus = [
        atom
        for atom, _ in space.attentional_focus(budget * 4)
        if isinstance(atom, Node) and atom.atype == CONCEPT
    ][:budget]

    # Scan: the first `budget` atoms in insertion order — no policy at all.
    scan = all_concepts[:budget]
    # Random: the same budget spent uniformly.
    chance = rng.sample(all_concepts, min(budget, len(all_concepts)))
    # Top-degree: the strongest static-structure heuristic.
    top_degree = by_degree[:budget]
    # Breadth-first from the same seeds, stopped at the same number of atom
    # touches. This is the equal-compute null the card asks for.
    began = time.perf_counter()
    walked = _breadth_first(space, seeded, touches=touched, budget=budget)
    walk_s = time.perf_counter() - began

    return {
        # No arm can exceed budget/|relevant|; a recall of 0.05 against a
        # ceiling of 0.064 is a different result from 0.05 against 1.0.
        "ceiling": min(1.0, budget / len(relevant)),
        "ecan": _recall(focus, relevant),
        "scan": _recall(scan, relevant),
        "random": _recall(chance, relevant),
        "top_degree": _recall(top_degree, relevant),
        "breadth_first": _recall(walked, relevant),
        "ecan_seconds": ecan_s,
        "breadth_first_seconds": walk_s,
        "atom_touches": touched,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", type=int, default=40_000)
    parser.add_argument("--communities", type=int, default=40)
    parser.add_argument("--cross-rate", type=float, default=0.08)
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--ticks", type=int, default=3)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1_618_033)
    parser.add_argument("--out", default="docs/evidence/ecan_stress.json")
    args = parser.parse_args()

    space, groups, degree = build(
        atoms=args.atoms,
        communities=args.communities,
        cross_rate=args.cross_rate,
        seed=args.seed,
    )
    all_concepts = [n for g in groups for n in g]
    by_degree = sorted(all_concepts, key=lambda n: -degree[n])

    rng = random.Random(args.seed + 1)
    rows = [
        one_trial(
            space,
            groups,
            degree,
            budget=args.budget,
            seeds=args.seeds,
            ticks=args.ticks,
            rng=rng,
            all_concepts=all_concepts,
            by_degree=by_degree,
        )
        for _ in range(args.trials)
    ]

    arms = ("ecan", "scan", "random", "top_degree", "breadth_first")
    ceiling = statistics.fmean(r["ceiling"] for r in rows)
    summary = {
        arm: {
            "mean_recall": round(statistics.fmean(r[arm] for r in rows), 4),
            "median_recall": round(statistics.median(r[arm] for r in rows), 4),
            "share_of_ceiling": round(
                statistics.fmean(r[arm] for r in rows) / ceiling, 4
            )
            if ceiling
            else 0.0,
        }
        for arm in arms
    }
    # The null that matters: attention must beat the best cheap heuristic, not
    # merely beat chance. Paired over trials, because the arms share the task.
    wins = sum(1 for r in rows if r["ecan"] > r["top_degree"])
    ties = sum(1 for r in rows if r["ecan"] == r["top_degree"])
    walk_wins = sum(1 for r in rows if r["ecan"] > r["breadth_first"])
    walk_ties = sum(1 for r in rows if r["ecan"] == r["breadth_first"])
    payload = {
        "schema": "aura.ecan_stress.v1",
        "card": "101",
        "claim_boundary": (
            "recall at a fixed read budget on a synthetic community graph "
            "where relevance is correlated with structure; not a claim about "
            "attention on Aura's live knowledge"
        ),
        "atoms_held": len(space),
        "config": {
            "atoms": args.atoms,
            "communities": args.communities,
            "cross_rate": args.cross_rate,
            "budget": args.budget,
            "seeds": args.seeds,
            "ticks": args.ticks,
            "trials": args.trials,
            "seed": args.seed,
        },
        "recall_ceiling": round(ceiling, 4),
        "arms": summary,
        "ecan_beats_top_degree": {"wins": wins, "ties": ties, "of": len(rows)},
        "ecan_beats_breadth_first": {
            "wins": walk_wins, "ties": walk_ties, "of": len(rows)
        },
        "matched_compute": {
            "basis": "atom touches during spreading, spent again by the walk",
            "median_atom_touches": statistics.median(r["atom_touches"] for r in rows),
            "ecan_seconds_median": round(
                statistics.median(r["ecan_seconds"] for r in rows), 4
            ),
            "breadth_first_seconds_median": round(
                statistics.median(r["breadth_first_seconds"] for r in rows), 4
            ),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    out = ROOT / args.out
    with local_internal_governed_scope("ecan_stress_campaign"):
        get_file_write_gateway().write_text(
            out, json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
