#!/usr/bin/env python3
"""Every field the state carries, and whether the core declares it.

A core cannot be declared causally closed by definition. `K` is ten domains of
declared columns, and the declaration is a claim: that everything persistent
which changes what she does next is either read by one of those columns or is
outside the core and findable by the closure test. This walks the state objects
and checks it.

Three answers per field:

    in K         a domain schema names this attribute as a column's source
    periphery    nothing in K reads it, and the closure test can see it
    unmapped     nothing in K reads it and nothing has said why

`unmapped` is the interesting column. It is not automatically a defect — most
of what a state carries is bookkeeping — but each one is a field that could
change the core's future with no line of the battery watching, and the list of
them only shrinks.

    python tools/audit_state_outside_the_core.py            # list them
    python tools/audit_state_outside_the_core.py --check    # fail on new
    python tools/audit_state_outside_the_core.py --by-domain

The walk is over the dataclass fields of the state objects rather than over a
hand-written list of suspects. A list of suspects is a list of the ones already
thought of, and the point is the one that was not.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BASELINE = REPO / "config" / "state_outside_the_core_baseline.json"

#: The state objects the core reads through. Named by the attribute they hang
#: off `AuraState` under, which is also how a schema source names them.
ROOTS: tuple[str, ...] = (
    "identity",
    "affect",
    "motivation",
    "cognition",
    "world",
    "soma",
)

#: Fields that are bookkeeping by construction: when a thing was written, which
#: version it is, what its identifier is. None of these change what she does.
BOOKKEEPING: frozenset[str] = frozenset(
    {
        "state_id",
        "version",
        "created_at",
        "updated_at",
        "last_updated",
        "timestamp",
        "schema_version",
    }
)

#: Fields that are outside K on purpose, with the reason. A field here is a
#: claim that the closure test is what watches it, and closure reads the whole
#: machine minus the core — so if one of these predicts the core's next state,
#: the closure criterion fails and names it.
OUTSIDE_K: dict[str, str] = {
    "cold": "a durable archive; what reaches cognition from it arrives as retrieved memory, which M reads",
    "context_partition": "a visibility rule over fields, not a quantity cognition computes from",
    "partition_mask": "the same rule's switch",
    "health": "circuit-breaker and sidecar state; it gates whether a subsystem runs, and closure reads it as periphery",
    "resilience": "the same",
    # Labels on the record of how this state came to be. They are read back by
    # tooling and by the lineage graph, and nothing computes from them.
    "parent_state_id": "which state this one descended from: lineage bookkeeping, not a quantity",
    "lineage_entity_id": "the same",
    "lineage_signature": "the signature over that lineage record",
    "transition_cause": "why the last transition happened, kept for the record",
    "transition_origin": "and where it came from",
    "formation_timestamp": "when the identity was formed",
    "last_evolution_timestamp": "when it last changed",
    "last_tick": "when motivation last ran",
    "last_thought_at": "when she last thought",
    "last_kernel_cycle_id": "which cycle produced the last decision",
    "name": "her name, which does not vary",
    # Readouts: recomputed from other fields every turn, so nothing written
    # into them survives to change anything. A displacement of a readout is a
    # displacement of the instrument.
    "phi": "recomputed from the state each turn; C carries the estimate that is read",
    "mind_moment": "a projection assembled for display from fields the domains already read",
    "unity_state": "the same, from the binding phase",
    "phenomenal_state": "recomputed each turn from affect and cognition; nothing written into it survives",
    "dominant_emotion": "the argmax of affect.emotions, which A reads in full",
    "last_response": "the text of the last reply, which no domain computes from",
    "last_retrieval_query": "the string the last recall used",
    "kernel_decision_count": "a counter for the health report",
    "kernel_veto_count": "the same",
    "last_veto_reasons": "the reasons behind that counter, kept for the record",
}


def _schema_sources() -> dict[str, set[str]]:
    """Every attribute path a domain column declares, by domain."""
    from core.subject.state import DOMAINS, schema as schema_of

    out: dict[str, set[str]] = {}
    for domain in DOMAINS:
        out[domain] = set()
        for source in schema_of(domain).sources:
            # `world.recent_percepts[*].salience` and `organ:substrate.curiosity`
            # both name a root and a field; the audit only needs the first two
            # segments to say a field is read.
            clean = str(source).split("[", 1)[0].replace("organ:", "")
            parts = clean.split(".")
            if len(parts) >= 2:
                out[domain].add(f"{parts[0]}.{parts[1]}")
            elif parts:
                out[domain].add(parts[0])
    return out


def _fields_of(obj: Any) -> list[str]:
    if dataclasses.is_dataclass(obj):
        return [f.name for f in dataclasses.fields(obj)]
    slots = getattr(type(obj), "__slots__", None)
    if slots:
        return [str(name) for name in slots]
    return sorted(k for k in vars(obj) if not k.startswith("_"))


def findings() -> dict[str, list[dict[str, Any]]]:
    from core.state.aura_state import AuraState

    by_domain = _schema_sources()

    state = AuraState()
    rows: list[dict[str, Any]] = []

    for root in ROOTS:
        child = getattr(state, root, None)
        if child is None:
            continue
        for name in _fields_of(child):
            if name.startswith("_") or name in BOOKKEEPING:
                continue
            path = f"{root}.{name}"
            domains = sorted(d for d, paths in by_domain.items() if path in paths)
            if domains:
                verdict = "in K"
            elif name in OUTSIDE_K:
                verdict = "periphery"
            else:
                verdict = "unmapped"
            rows.append(
                {"path": path, "verdict": verdict, "domains": domains,
                 "why": OUTSIDE_K.get(name, "")}
            )

    # And the top-level fields, which are not inside any of the six objects.
    for name in _fields_of(state):
        if name.startswith("_") or name in BOOKKEEPING or name in ROOTS:
            continue
        path = name
        domains = sorted(d for d, paths in by_domain.items() if path in paths)
        if domains:
            verdict = "in K"
        elif name in OUTSIDE_K:
            verdict = "periphery"
        else:
            verdict = "unmapped"
        rows.append(
            {"path": path, "verdict": verdict, "domains": domains,
             "why": OUTSIDE_K.get(name, "")}
        )

    out: dict[str, list[dict[str, Any]]] = {"in K": [], "periphery": [], "unmapped": []}
    for row in sorted(rows, key=lambda r: r["path"]):
        out[row["verdict"]].append(row)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--by-domain", action="store_true")
    args = parser.parse_args()

    found = findings()
    unmapped = sorted(row["path"] for row in found["unmapped"])

    if args.write_baseline:
        BASELINE.write_text(
            json.dumps({"allowed": unmapped}, indent=2) + "\n", encoding="utf-8"
        )
        print(f"baseline written: {len(unmapped)} unmapped field(s)")
        return 0

    allowed: set[str] = set()
    if BASELINE.is_file():
        allowed = set(json.loads(BASELINE.read_text()).get("allowed", []))

    if args.by_domain:
        counts: dict[str, int] = {}
        for row in found["in K"]:
            for domain in row["domains"]:
                counts[domain] = counts.get(domain, 0) + 1
        for domain, count in sorted(counts.items()):
            print(f"  {domain}: {count} field(s) read")
        print()

    print(f"{len(found['in K'])} field(s) a domain reads")
    for row in found["periphery"]:
        print(f"  outside K  {row['path']}: {row['why']}")
    print(f"{len(found['periphery'])} field(s) outside K on purpose")
    for path in unmapped:
        print(f"  {'  ' if path in allowed else '+ '}unmapped  {path}")
    print(f"{len(unmapped)} field(s) nothing in K reads and nothing has explained")

    if not args.check:
        return 0
    new = sorted(set(unmapped) - allowed)
    gone = sorted(allowed - set(unmapped))
    if gone:
        print(f"\n{len(gone)} baseline entries are resolved; rerun with --write-baseline:")
        for path in gone:
            print(f"  - {path}")
    if new:
        print(f"\n{len(new)} NEW unmapped state field(s):")
        for path in new:
            print(f"  + {path}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
