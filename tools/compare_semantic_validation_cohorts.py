#!/usr/bin/env python3
"""Compare paired validation receipts without turning an oracle into a router."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def compare(old: dict, new: dict, *, old_name: str) -> dict:
    from core.learning.semantic_validation_checkpoint import _digest

    old_body = {key: value for key, value in old.items() if key != "sha256"}
    if (old.get("schema") != "aura.semantic_validation_checkpoint.v1"
            or old.get("sha256") != _digest(old_body)):
        raise ValueError("old validation checkpoint receipt differs")
    old_rows = old["rows"][old_name]
    if (not isinstance(old_rows, dict)
            or any(not isinstance(value, list) or len(value) != 2
                   or any(type(part) is not bool for part in value)
                   or (value[0] and not value[1]) for value in old_rows.values())):
        raise ValueError("old validation observations differ")
    new_body = {key: value for key, value in new.items() if key != "receipt_sha256"}
    if (new.get("schema") != "aura.semantic_cohort_diagnosis.v2"
            or new.get("receipt_sha256") != _digest(new_body)):
        raise ValueError("new cohort receipt differs")
    paired = [row for row in new["rows"] if row["split"] == "validation"]
    ids = [row["source_text_sha256"] for row in paired]
    if len(ids) != len(set(ids)) or set(ids) != set(old_rows):
        raise ValueError("validation cohorts are not exactly paired")
    counts = {"both": 0, "old_only": 0, "new_only": 0, "neither": 0}
    residual = []
    for row in paired:
        source = row["source_text_sha256"]
        row_body = {key: value for key, value in row.items() if key != "receipt_sha256"}
        if row.get("receipt_sha256") != _digest(row_body):
            raise ValueError("new validation row receipt differs")
        old_correct = old_rows[source][1]
        new_correct = row["observation"]["semantic_status"] == "equivalent"
        key = ("both" if old_correct and new_correct else
               "old_only" if old_correct else "new_only" if new_correct else "neither")
        counts[key] += 1
        if key == "neither":
            residual.append({"source": source, "construction": row["construction_id"],
                             "topology": row["topology_id"]})
    return {"population": len(ids), "counts": counts,
            "old_correct": counts["both"] + counts["old_only"],
            "new_correct": counts["both"] + counts["new_only"],
            "oracle_union": len(ids) - counts["neither"],
            "residual": sorted(residual, key=lambda item: item["source"]),
            "serving_authority": False, "qualification_evidence": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--old-name", default="incumbent")
    args = parser.parse_args()
    print(json.dumps(compare(json.loads(args.old.read_bytes()),
                             json.loads(args.new.read_bytes()),
                             old_name=args.old_name), sort_keys=True))


if __name__ == "__main__":
    main()
