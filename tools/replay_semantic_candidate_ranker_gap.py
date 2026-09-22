#!/usr/bin/env python3
"""Diagnose a source-trained ranker on the exposed, frozen G03 gap bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _verify_artifacts(training: dict, weights: bytes, gap: dict, bank: dict,
                      gap_raw: bytes, source_raw: bytes, candidate_raw: bytes,
                      model) -> tuple[str, ...]:
    training_body = {key: value for key, value in training.items() if key != "receipt_sha256"}
    bank_body = {key: value for key, value in bank.items() if key != "receipt_sha256"}
    wanted = tuple(sorted(row["source"] for row in gap["failures"]))
    if (training.get("schema") != "aura.semantic_candidate_ranker_source_fold.v1"
            or training["receipt_sha256"] != _digest(training_body)
            or training["weights_sha256"] != hashlib.sha256(weights).hexdigest()
            or training["model_receipt_sha256"] != model.receipt_sha256
            or training["source_report_sha256"] != hashlib.sha256(source_raw).hexdigest()
            or training["candidate_report_sha256"] != hashlib.sha256(candidate_raw).hexdigest()
            or bank["receipt_sha256"] != _digest(bank_body)
            or bank["schema"] != "aura.semantic_gap_candidate_bank_audit.v1"
            or bank["plan"]["model_receipt_sha256"] != model.receipt_sha256
            or bank["plan"]["gap_report_sha256"] != hashlib.sha256(gap_raw).hexdigest()
            or bank["plan"]["source_report_sha256"] != hashlib.sha256(source_raw).hexdigest()
            or bank["plan"]["candidate_report_sha256"] != hashlib.sha256(
                candidate_raw).hexdigest()
            or tuple(bank["plan"]["sources"]) != wanted
            or bank["complete_population"] is not True
            or bank["observed"] != len(wanted)):
        raise ValueError("ranker diagnostic artifacts do not share a verified source basis")
    return wanted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transducer", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--gap-report", type=Path, required=True)
    parser.add_argument("--bank-report", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from tools.refit_semantic_argument_proposals import (
        configure_refit_environment,
        load_source_examples,
    )

    configure_refit_environment(args.output)
    from safetensors.torch import load_file

    from core.learning.semantic_candidate_ranker import ContextualProgramRanker
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
    )
    from core.learning.semantic_request_context import RequestContextConfig
    from tools.evaluate_semantic_candidate_ranker import _evaluate

    model = compositional_semantic_program_transducer_from_dict(json.loads(args.transducer.read_bytes()))
    source_raw = args.source_report.read_bytes()
    candidate_raw = args.candidate_report.read_bytes()
    source = json.loads(source_raw)
    gap_raw = args.gap_report.read_bytes()
    gap = json.loads(gap_raw)
    bank = json.loads(args.bank_report.read_bytes())
    training = json.loads(args.training_report.read_bytes())
    wanted = _verify_artifacts(training, args.weights.read_bytes(), gap, bank,
                               gap_raw, source_raw, candidate_raw, model)
    bundles = [name + "=" + str(args.feature_root / name) for name in
               source["representation_compatibility"]["source_feature_manifest_sha256s"]]
    examples = load_source_examples(model, source, bundles)
    items = {item.ir.source_text_sha256: item for item in examples if item.split == "validation"}
    rows = {row["source"]: row for row in bank["rows"]}
    if set(rows) != set(wanted) or not set(wanted) <= set(items):
        raise ValueError("diagnostic bank and frozen validation cohort differ")
    config = RequestContextConfig(**training["config"])
    ranker = ContextualProgramRanker(config)
    ranker.load_state_dict(load_file(str(args.weights)), strict=True)
    result = _evaluate(ranker, items, rows, list(wanted))
    body = {"schema": "aura.semantic_candidate_ranker_exposed_gap.v1",
            "development_only": True, "fresh_transfer": False, "serving_authority": False,
            "pilot_training": training["pilot_only"],
            "training_receipt_sha256": training["receipt_sha256"],
            "gap_report_sha256": hashlib.sha256(args.gap_report.read_bytes()).hexdigest(),
            "bank_report_sha256": hashlib.sha256(args.bank_report.read_bytes()).hexdigest(),
            "evaluation": result}
    payload = json.dumps({**body, "receipt_sha256": _digest(body)}, sort_keys=True).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_bytes() != payload:
        raise ValueError("diagnostic report already exists with different content")
    args.output.write_bytes(payload)
    print(json.dumps({"stage": "exposed_gap", "pilot_training": training["pilot_only"],
                      "population": result["population"],
                      "bank_reachable": result["bank_reachable"],
                      "incumbent_correct": result["incumbent_correct"],
                      "ranker_correct": result["ranker_correct"]}), flush=True)


if __name__ == "__main__":
    main()
