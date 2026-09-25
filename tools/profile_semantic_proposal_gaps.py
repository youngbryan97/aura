#!/usr/bin/env python3
"""Localize source-fold proposal gaps without treating bounded search as exhaustive."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.profile_semantic_crossfit_misses import _digest, _load_verified, classify_row


def classify_proposal_gap(item, row: dict) -> dict:
    """Compare witnessed proposals with the source target, diagnostic-only."""
    classify_row(row)
    target = tuple(instruction.op for instruction in item.ir.instructions)
    charts = [tuple(entry["op"] for entry in chart["operations"])
              for chart in row["bank"]["charts"]]
    reachable = row["diagnosis"]["correct_reachable"]
    if reachable is True:
        stage = "correct_candidate_observed"
    elif any(chart == target for chart in charts):
        stage = "target_operation_chart_observed_no_equivalent_graph"
    else:
        stage = "target_operation_chart_not_observed"
    if reachable is False and not row["bank"]["search_complete"]:
        raise ValueError("incomplete search cannot certify proposal absence")
    if reachable is None and row["bank"]["search_complete"]:
        raise ValueError("completed search cannot have unknown reach")
    return {"source": row["source"], "construction": item.construction_id,
            "stage": stage, "target_operations": list(target),
            "chart_operations": [list(chart) for chart in charts],
            "search_complete": row["bank"]["search_complete"],
            "correct_reachable": reachable}


def profile(items: dict, directories: list[Path]) -> dict:
    seen = set()
    rows = []
    inputs = {}
    source_basis = None
    parent_basis = None
    for directory in directories:
        plan, observations = _load_verified(directory)
        basis = (plan["source_report_sha256"], plan["parent_receipt_sha256"])
        if source_basis is None:
            source_basis, parent_basis = basis
        elif basis != (source_basis, parent_basis):
            raise ValueError("proposal profile combines different source or parent models")
        held = set(plan["held_ids"])
        if held & seen or not held <= set(items):
            raise ValueError("proposal profile sources are duplicated or missing")
        seen.update(held)
        inputs[str(directory)] = hashlib.sha256((directory / "report.json").read_bytes()).hexdigest()
        rows.extend(classify_proposal_gap(items[row["source"]], row)
                    for row in observations)
    stages = Counter(row["stage"] for row in rows)
    constructions = defaultdict(Counter)
    for row in rows:
        constructions[row["construction"]][row["stage"]] += 1
    return {"schema": "aura.semantic_proposal_gap_profile.v1",
            "source_report_sha256": source_basis,
            "parent_receipt_sha256": parent_basis,
            "population": len(rows), "by_stage": dict(sorted(stages.items())),
            "by_construction": {key: dict(sorted(value.items()))
                                for key, value in sorted(constructions.items())},
            "input_report_sha256s": inputs,
            "development_only": True, "serving_authority": False,
            "rows": sorted(rows, key=lambda row: row["source"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
    )
    from tools.refit_semantic_argument_proposals import load_source_examples

    source = json.loads(args.source_report.read_bytes())
    parent = compositional_semantic_program_transducer_from_dict(
        json.loads(args.parent.read_bytes()))
    bundles = [name + "=" + str(args.feature_root / name) for name in
               source["representation_compatibility"]["source_feature_manifest_sha256s"]]
    examples = load_source_examples(parent, source, bundles)
    items = {item.ir.source_text_sha256: item for item in examples if item.split == "train"}
    for directory in args.directories:
        plan, _rows = _load_verified(directory)
        if (plan["source_report_sha256"] != hashlib.sha256(
                args.source_report.read_bytes()).hexdigest()
                or plan["parent_receipt_sha256"] != parent.receipt_sha256):
            raise ValueError("proposal bank differs from source or parent")
    body = profile(items, args.directories)
    payload = json.dumps({**body, "receipt_sha256": _digest(body)}, sort_keys=True).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_bytes() != payload:
        raise ValueError("proposal gap profile already exists with different content")
    args.output.write_bytes(payload)
    print(json.dumps({"population": body["population"], "by_stage": body["by_stage"],
                      "by_construction": body["by_construction"]}, sort_keys=True))


if __name__ == "__main__":
    main()
