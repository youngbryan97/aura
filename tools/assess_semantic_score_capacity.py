#!/usr/bin/env python3
"""Check capacity of the exact retained graph-factor contrasts from a source fit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--contrast-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict-ranking", action="store_true",
                        help="test any positive ranking margin, rather than the default unit margin")
    args = parser.parse_args()
    from core.governance_context import local_internal_governed_scope
    from core.learning.score_capacity import assess_score_capacity, verify_score_capacity
    from core.learning.semantic_program_campaign import _sha
    from core.learning.semantic_program_compositional_transducer import compositional_semantic_program_transducer_from_dict
    from core.runtime.atomic_writer import atomic_write_bytes_if_absent
    from tools.run_semantic_program_replication import _load_json

    model = compositional_semantic_program_transducer_from_dict(_load_json(args.candidate, max_bytes=64*1024*1024))
    receipt = model.training_receipt["argument_graph_factor_refit"]
    records = []
    with args.contrast_log.open() as stream:
        for line in stream:
            event = json.loads(line)
            if event.get("stage") == "source_graph_contrasts":
                if event["completed"] != len(records) + 1:
                    raise ValueError("contrast progress is not complete and ordered")
                records.append(event["row"])
    if len(records) != receipt["training_examples"] or _sha(records) != receipt["contrast_rows_sha256"]:
        raise ValueError("retained contrasts differ from the source fit")
    rows = [r for r in records if r["status"] == "contrast"]
    differences = [[r["factor_difference"][i] for i in (0, 2, 3)] for r in rows]
    offsets = [r["fixed_difference"] + model.argument_proposal_scale*r["factor_difference"][1]
               for r in rows]
    report = assess_score_capacity(differences, offsets, lower_bounds=[1e-6]*3,
                                   comparison_ids=[r["source_text_sha256"] for r in rows],
                                   margin=0. if args.strict_ranking else 1., strict=args.strict_ranking)
    if report["verified"] and not verify_score_capacity(report):
        raise AssertionError("capacity witness failed independent replay")
    report["source_binding"] = {
        "parent_transducer_receipt_sha256": receipt["parent_transducer_receipt_sha256"],
        "contrast_rows_sha256": receipt["contrast_rows_sha256"],
        "training_examples": receipt["training_examples"],
        "scope": "exact_register_assignment_with_annotated_operations",
        "comparison_requirement": "strictly_better" if args.strict_ranking else "unit_margin",
        "equivalent_programs_may_appear_as_contrasts": True,
        "proof_of_semantic_impossibility": False,
    }
    with local_internal_governed_scope("learning.score_capacity", domain="file_write"):
        if not atomic_write_bytes_if_absent(args.output, (json.dumps(report, sort_keys=True)+"\n").encode(), mode=0o400):
            raise FileExistsError(args.output)
    print(json.dumps({k: v for k, v in report.items() if k not in ("problem", "certificate", "weights")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
