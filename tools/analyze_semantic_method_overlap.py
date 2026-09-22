#!/usr/bin/env python3
"""Compare four frozen semantic methods without learning from exposed labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _digest(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _verified(path: Path) -> dict:
    report = json.loads(path.read_bytes())
    body = {key: value for key, value in report.items() if key != "receipt_sha256"}
    if report.get("receipt_sha256") != _digest(body):
        raise ValueError(f"unverified semantic comparison receipt: {path.name}")
    return report


def _verified_folds(path: Path) -> dict:
    from core.learning.semantic_program_campaign import _sha

    report = json.loads(path.read_bytes())
    body = {key: value for key, value in report.items() if key != "receipt_sha256"}
    if report.get("receipt_sha256") != _sha(body):
        raise ValueError(f"unverified semantic folds receipt: {path.name}")
    return report


def _outcome(bank_row: dict, program_sha: str | None, recorded: bool) -> dict:
    statuses = {entry["program_sha256"]: entry["status"]
                for entry in bank_row["diagnosis"]["comparisons"]}
    candidates = {entry["program_sha256"] for entry in bank_row["bank"]["candidates"]}
    if program_sha is not None and program_sha not in candidates:
        raise ValueError("method selected a program absent from the frozen candidate bank")
    correct = program_sha is not None and statuses.get(program_sha) == "equivalent"
    if correct != recorded:
        raise ValueError("method outcome differs from the frozen semantic comparison")
    return {"program_sha256": program_sha, "correct": correct}


def _normalize(source: str, construction: str, bank_row: dict,
               ranker_row: dict, direct_row: dict, prototype_row: dict,
               *, direct_key: str) -> dict:
    incumbent = bank_row["bank"]["selected_program_sha256"]
    return {"source": source, "construction": construction, "methods": {
        "incumbent": _outcome(bank_row, incumbent, bool(prototype_row["incumbent_correct"])),
        "ranker": _outcome(bank_row, ranker_row["chosen_program_sha256"],
                           bool(direct_row["ranker_correct"])),
        "direct": _outcome(bank_row, direct_row[direct_key],
                           bool(direct_row["direct_correct"])),
        "prototype": _outcome(bank_row, prototype_row["chosen_program_sha256"],
                              bool(prototype_row["prototype_correct"])),
    }}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-comparison", "gap-comparison", "prototype-report",
                 "source-bank", "gap-bank", "ranker-report", "folds", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()

    from core.learning.semantic_method_overlap import analyze_method_overlap
    from tools.evaluate_semantic_candidate_ranker import _read_bank

    source = _verified(args.source_comparison)
    gap = _verified(args.gap_comparison)
    prototype = _verified(args.prototype_report)
    ranker = _verified(args.ranker_report)
    bank = _verified(args.source_bank / "report.json")
    gap_bank = _verified(args.gap_bank)
    folds = _verified_folds(args.folds)
    source_rows = {row["source"]: row for row in source["rows"]}
    ranker_rows = {row["source"]: row for row in ranker["evaluation"]["rows"]}
    prototype_source = {row["source"]: row for row in prototype["heldout_source"]["rows"]}
    gap_rows = {row["source"]: row for row in gap["direct_comparison"]["rows"]}
    gap_ranker = {row["source"]: row for row in gap["evaluation"]["rows"]}
    prototype_gap = {row["source"]: row for row in prototype["exposed_gap"]["rows"]}
    gap_bank_rows = {row["source"]: row for row in gap_bank["rows"]}
    source_constructions = {row[0]: row[1] for row in folds["population"]}
    if (source.get("schema") != "aura.semantic_candidate_methods_source_fold.v1"
            or gap.get("schema") != "aura.semantic_candidate_methods_exposed_gap.v1"
            or prototype.get("schema") != "aura.semantic_operation_evidence_probe.v1"
            or not (source["fold"] == ranker["fold"] == prototype["fold"])
            or gap["direct_comparison"]["source_fold"] != source["fold"]
            or source["direct_checkpoint_sha256"]
            != gap["direct_comparison"]["checkpoint_sha256"]
            or source["source_bank_receipt_sha256"] != bank["receipt_sha256"]
            or prototype["source_bank_receipt_sha256"] != bank["receipt_sha256"]
            or source["ranker_receipt_sha256"] != ranker["receipt_sha256"]
            or prototype["gap_bank_receipt_sha256"] != gap_bank["receipt_sha256"]
            or gap["bank_report_sha256"] != hashlib.sha256(args.gap_bank.read_bytes()).hexdigest()
            or not (set(source_rows) == set(ranker_rows) == set(prototype_source))
            or not (set(gap_rows) == set(gap_ranker) == set(prototype_gap) == set(gap_bank_rows))
            or source["population"] != len(source_rows)
            or gap["direct_comparison"]["population"] != len(gap_rows)):
        raise ValueError("method reports do not share frozen populations and model identities")
    normalized_source = []
    for source_id in sorted(source_rows):
        row = _read_bank(args.source_bank / "rows" / f"{source_id}.json",
                         source=source_id, plan_sha=bank["plan"]["plan_sha256"],
                         model_receipt=bank["plan"]["model_receipt_sha256"],
                         expected_receipt=bank["row_receipts"][source_id])
        normalized_source.append(_normalize(
            source_id, source_constructions[source_id], row,
            ranker_rows[source_id], source_rows[source_id], prototype_source[source_id],
            direct_key="direct_program_sha256"))
    normalized_gap = [_normalize(
        source_id, gap_rows[source_id]["construction"], gap_bank_rows[source_id],
        gap_ranker[source_id], gap_rows[source_id], prototype_gap[source_id],
        direct_key="direct_program_sha256") for source_id in sorted(gap_rows)]
    source_result = analyze_method_overlap(normalized_source)
    gap_result = analyze_method_overlap(normalized_gap)
    body = {"schema": "aura.semantic_method_overlap.v1", "serving_authority": False,
            "development_only": True, "fresh_transfer": False,
            "source_comparison_sha256": hashlib.sha256(args.source_comparison.read_bytes()).hexdigest(),
            "gap_comparison_sha256": hashlib.sha256(args.gap_comparison.read_bytes()).hexdigest(),
            "prototype_report_sha256": hashlib.sha256(args.prototype_report.read_bytes()).hexdigest(),
            "heldout_source": source_result, "exposed_gap": gap_result,
            "rows": {"heldout_source": normalized_source, "exposed_gap": normalized_gap}}
    payload = json.dumps({**body, "receipt_sha256": _digest(body)}, sort_keys=True).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_bytes() != payload:
        raise ValueError("method overlap report already differs")
    args.output.write_bytes(payload)
    print(json.dumps({key: {name: body[key][name] for name in (
        "population", "correct", "unique_successes", "oracle_union", "all_wrong",
        "consensus_covered", "consensus_correct")}
        for key in ("heldout_source", "exposed_gap")}), flush=True)


if __name__ == "__main__":
    main()
