#!/usr/bin/env python3
"""Load the AtomSpace to a million atoms and time what it costs.

Gap Atlas card 098 says Hyperon is building a dedicated runtime for atom
counts far past anything Aura had measured. The store was measured to 50,000,
which is the default ``max_atoms`` and therefore says nothing about scale — it
says the forgetter works. This runs it past that ceiling and reports where
each operation actually goes, so the card is answered by a curve rather than
by a ceiling nobody crossed.

Reports, at each size: build rate, resident bytes per atom, point-read
latency at p50/p99, a typed pattern match, and the cost of one full ECAN
cycle. Every number is measured here; none is asserted.

    python tools/campaigns/atomspace_scale.py --sizes 10000,100000,1000000
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

from core.knowledge.atomspace import (  # noqa: E402
    EVALUATION,
    LIST,
    AtomSpace,
    Link,
    Node,
    TruthValue,
    Variable,
)

CONCEPT = "ConceptNode"
PREDICATE = "PredicateNode"


def _resident_bytes() -> int:
    """Resident set size in bytes, or 0 where the platform will not say."""
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (ImportError, OSError, ValueError):
        return 0
    # Linux reports kilobytes, macOS bytes. A value under a megabyte for a
    # process this size can only be the kilobyte reading.
    return peak if peak > 1 << 20 else peak * 1024


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(fraction * len(ordered)))
    return ordered[index]


def one_size(size: int, *, reads: int, seed: int) -> dict[str, object]:
    """Build a store of ``size`` atoms and measure it."""
    rng = random.Random(seed)
    # max_atoms above the target: the forgetter must not run, or the store
    # under test is a different store from the one the card asks about.
    space = AtomSpace(max_atoms=size * 4, focus_size=64)

    concepts = [Node(CONCEPT, f"c{i}") for i in range(size // 2)]
    predicate = Node(PREDICATE, "related")

    before = _resident_bytes()
    began = time.perf_counter()
    for node in concepts:
        space.add(node, TruthValue(0.9, 12.0), source="scale")
    # The other half are links, which is what makes this a metagraph rather
    # than a table: each carries its own truth and its own incoming index.
    for i in range(size - len(concepts)):
        a = concepts[rng.randrange(len(concepts))]
        b = concepts[rng.randrange(len(concepts))]
        space.add(
            Link(EVALUATION, (predicate, Link(LIST, (a, b)))),
            TruthValue(0.8, 8.0),
            source="scale",
        )
    build_s = time.perf_counter() - began
    after = _resident_bytes()

    held = len(space)
    sample = [concepts[rng.randrange(len(concepts))] for _ in range(reads)]
    latencies: list[float] = []
    for node in sample:
        t0 = time.perf_counter()
        space.get_tv(node)
        latencies.append((time.perf_counter() - t0) * 1e6)

    # A real pattern with variables in it, not a name lookup: this is the
    # operation whose cost the card is actually about.
    pattern = Link(
        EVALUATION, (predicate, Link(LIST, (Variable("a"), Variable("b"))))
    )
    t0 = time.perf_counter()
    matched = space.match(pattern)
    typed_match_s = time.perf_counter() - t0

    for node in sample[:64]:
        space.stimulate(node)
    t0 = time.perf_counter()
    moved = space.spread_importance()
    space.collect_rent()
    focus = space.attentional_focus()
    ecan_cycle_s = time.perf_counter() - t0

    return {
        "requested": size,
        # Higher than requested: an EVALUATION link brings its LIST child in
        # with it, and the store holds the whole metagraph rather than a row.
        "held": held,
        "build_seconds": round(build_s, 3),
        "atoms_per_second": round(held / build_s) if build_s else 0,
        "resident_bytes_per_atom": round((after - before) / held, 1) if held else 0.0,
        "point_read_us_p50": round(statistics.median(latencies), 3),
        "point_read_us_p99": round(_percentile(latencies, 0.99), 3),
        "typed_match_seconds": round(typed_match_s, 4),
        "typed_match_results": len(matched),
        "ecan_cycle_seconds": round(ecan_cycle_s, 4),
        "ecan_sti_moved": round(moved, 2),
        "ecan_focus": len(focus),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="10000,100000,1000000")
    parser.add_argument("--reads", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=1_618_033)
    parser.add_argument("--out", default="docs/evidence/atomspace_scale.json")
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    rows = []
    for size in sizes:
        row = one_size(size, reads=args.reads, seed=args.seed)
        rows.append(row)
        print(json.dumps(row))

    payload = {
        "schema": "aura.atomspace_scale.v1",
        "card": "098",
        "claim_boundary": (
            "single-process in-memory AtomSpace throughput and read latency at "
            "the stated atom counts on this host; not a distributed or "
            "persistent store comparison"
        ),
        "seed": args.seed,
        "reads_per_size": args.reads,
        "sizes": rows,
    }
    out = ROOT / args.out
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    with local_internal_governed_scope("atomspace_scale_campaign"):
        get_file_write_gateway().write_text(
            out, json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
