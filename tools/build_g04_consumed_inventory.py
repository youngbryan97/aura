#!/usr/bin/env python3
"""Everything G03's candidate was fitted on, selected with or tuned against.

G04 asks for transfer to requests none of that touched, so the protocol needs
the consumed population itself, not a description of it. Each consumed
feature bundle's seeded corpus is rebuilt from its own manifest (the same
reconstruction the bundle loader checks membership against), and every
example the manifest lists is kept with its text, program depth and the
operations that took a computed argument. Bundles are named on the command
line; nothing is assumed about which exist.

Usage:
    build_g04_consumed_inventory.py --bundle DIR [--bundle DIR ...] --output FILE
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEQUENCE_OPERATIONS = frozenset({"at", "count_of"})


def inventory(bundles: list[Path]) -> dict:
    from core.learning.semantic_program_feature_materialization import (
        rebuild_semantic_feature_selection,
    )

    rows: dict[str, dict] = {}
    seeds: set[int] = set()
    manifests: list[str] = []
    for directory in bundles:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        listed = {record["example_id"] for record in manifest["records"]}
        config, examples = rebuild_semantic_feature_selection(manifest)
        rebuilt = {example.example_id: example for example in examples}
        if set(rebuilt) != listed:
            raise SystemExit(f"{directory}: rebuilt corpus differs from the manifest's records")
        seeds.add(int(config.seed))
        manifests.append(manifest["manifest_sha256"])
        for example in examples:
            n_inputs = len(example.inputs)
            computed = sum(
                1
                for item in example.instructions
                if item.instruction.op in SEQUENCE_OPERATIONS
                and any(argument >= n_inputs for argument in item.instruction.args)
            )
            rows[example.example_id] = {
                "example_id": example.example_id,
                "split": example.split,
                "construction_id": example.construction_id,
                "source_text": example.source_text,
                "source_sha256": hashlib.sha256(example.source_text.encode("utf-8")).hexdigest(),
                "depth": len(example.instructions),
                "sequence_operations_with_computed_arguments": computed,
            }
    ordered = [rows[key] for key in sorted(rows)]
    return {
        "schema": "aura.g04_consumed_inventory.v1",
        "bundles": [str(path) for path in bundles],
        "manifest_sha256s": manifests,
        "seeds": sorted(seeds),
        "example_count": len(ordered),
        "max_depth": max(row["depth"] for row in ordered),
        "sequence_operations_with_computed_arguments": sum(
            row["sequence_operations_with_computed_arguments"] for row in ordered
        ),
        "examples": ordered,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inventory([path.expanduser() for path in args.bundle])
    args.output.expanduser().write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("example_count", "max_depth", "seeds",
                                                    "sequence_operations_with_computed_arguments")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
