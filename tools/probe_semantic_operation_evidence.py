#!/usr/bin/env python3
"""Test source-trained operation phrase evidence across frozen constructions."""

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


def _evaluate(model, items: dict, rows: dict, sources: set[str],
              *, positioned: bool = False) -> dict:
    from tools.evaluate_semantic_candidate_ranker import _rankable

    results = []
    for source in sorted(sources):
        item, row = items[source], rows[source]
        (programs, labels, keys), _spans, _kinds, anchors = _rankable(item, row)
        selected = model.choose(item.hidden_states, programs, anchors,
                                positioned=positioned)
        results.append({"source": source, "incumbent_correct": bool(
            row["bank"]["selected_program_sha256"] is not None and labels[0]),
            "prototype_correct": labels[selected], "candidate_available": any(labels),
            "chosen_program_sha256": keys[selected]})
    return {"population": len(results),
            "candidate_available": sum(row["candidate_available"] for row in results),
            "incumbent_correct": sum(row["incumbent_correct"] for row in results),
            "prototype_correct": sum(row["prototype_correct"] for row in results),
            "gains": sum(row["prototype_correct"] and not row["incumbent_correct"]
                         for row in results),
            "regressions": sum(row["incumbent_correct"] and not row["prototype_correct"]
                               for row in results),
            "rows": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("transducer", "source-report", "candidate-report", "feature-root",
                 "bank-directory", "folds", "gap-report", "gap-bank-report",
                 "ranker-report", "ranker-weights", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()

    from core.learning.semantic_operation_evidence import OperationEvidence
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
    )
    from tools.evaluate_semantic_candidate_ranker import _read_bank, _verify_sources
    from tools.refit_semantic_argument_proposals import (
        configure_refit_environment,
        load_source_examples,
    )
    from tools.replay_semantic_candidate_ranker_gap import _verify_artifacts

    configure_refit_environment(args.output)
    source_raw, candidate_raw = args.source_report.read_bytes(), args.candidate_report.read_bytes()
    gap_raw = args.gap_report.read_bytes()
    source_report, candidate_report = json.loads(source_raw), json.loads(candidate_raw)
    bank_report = json.loads((args.bank_directory / "report.json").read_bytes())
    folds = json.loads(args.folds.read_bytes())
    gap_report = json.loads(gap_raw)
    gap_bank = json.loads(args.gap_bank_report.read_bytes())
    ranker_report = json.loads(args.ranker_report.read_bytes())
    transducer = compositional_semantic_program_transducer_from_dict(
        json.loads(args.transducer.read_bytes()))
    bundles = [name + "=" + str(args.feature_root / name) for name in
               source_report["representation_compatibility"]["source_feature_manifest_sha256s"]]
    examples = load_source_examples(transducer, source_report, bundles)
    plan, ids = _verify_sources(source_report, candidate_report, transducer,
                                bank_report, folds, examples)
    if (type(args.fold) is not int or not 0 <= args.fold < folds["count"]
            or ranker_report["fold"] != args.fold
            or ranker_report["source_bank_receipt_sha256"] != bank_report["receipt_sha256"]):
        raise ValueError("operation probe source fold differs from frozen bank")
    wanted = _verify_artifacts(ranker_report, args.ranker_weights.read_bytes(),
                               gap_report, gap_bank, gap_raw, source_raw,
                               candidate_raw, transducer)
    source_items = {item.ir.source_text_sha256: item for item in examples
                    if item.split == "train"}
    validation_items = {item.ir.source_text_sha256: item for item in examples
                        if item.split == "validation"}
    held = {source for source in ids if folds["assignments"][source] == args.fold}
    train = set(ids) - held
    if not held or not train or not set(wanted) <= set(validation_items):
        raise ValueError("operation probe lacks its frozen source or validation cohort")
    source_rows = {source: _read_bank(
        args.bank_directory / "rows" / f"{source}.json", source=source,
        plan_sha=plan["plan_sha256"], model_receipt=transducer.receipt_sha256,
        expected_receipt=bank_report["row_receipts"][source]) for source in held}
    gap_rows = {row["source"]: row for row in gap_bank["rows"]}
    if set(gap_rows) != set(wanted):
        raise ValueError("operation probe gap bank changed its population")
    source_model = OperationEvidence.fit(examples, source_ids=train)
    source_result = _evaluate(source_model, source_items, source_rows, held)
    positioned_source = _evaluate(source_model, source_items, source_rows, held,
                                  positioned=True)
    all_source_model = OperationEvidence.fit(examples, source_ids=set(source_items))
    gap_result = _evaluate(all_source_model, validation_items, gap_rows, set(wanted))
    positioned_gap = _evaluate(all_source_model, validation_items, gap_rows, set(wanted),
                               positioned=True)
    body = {"schema": "aura.semantic_operation_evidence_probe.v2",
            "serving_authority": False, "development_only": True,
            "fresh_transfer": False, "fold": args.fold,
            "source_bank_receipt_sha256": bank_report["receipt_sha256"],
            "gap_bank_receipt_sha256": gap_bank["receipt_sha256"],
            "source_report_sha256": hashlib.sha256(source_raw).hexdigest(),
            "source_training_examples": len(train),
            "full_source_training_examples": len(source_items),
            "source_operation_support": source_model.support,
            "all_source_operation_support": all_source_model.support,
            "heldout_source": source_result, "exposed_gap": gap_result,
            "positioned_heldout_source": positioned_source,
            "positioned_exposed_gap": positioned_gap}
    payload = json.dumps({**body, "receipt_sha256": _digest(body)}, sort_keys=True).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_bytes() != payload:
        raise ValueError("operation evidence probe output already differs")
    args.output.write_bytes(payload)
    print(json.dumps({"source": {key: source_result[key] for key in (
        "population", "incumbent_correct", "prototype_correct", "gains", "regressions")},
        "exposed_gap": {key: gap_result[key] for key in (
            "population", "incumbent_correct", "prototype_correct")},
        "positioned_exposed_gap": {key: positioned_gap[key] for key in (
            "population", "incumbent_correct", "prototype_correct")}}), flush=True)


if __name__ == "__main__":
    main()
