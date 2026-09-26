#!/usr/bin/env python3
"""Profile failed source-order validation decodes without fitting or promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def failure_class(predicted, expected) -> str:
    """Separate missing decode, operation choice, order, and binding."""
    if predicted is None:
        return "decode_failure"
    actual_ops = tuple(step.op for step in predicted.instructions)
    expected_ops = tuple(step.op for step in expected.instructions)
    if Counter(actual_ops) != Counter(expected_ops):
        return "operation_choice"
    if actual_ops != expected_ops:
        return "operation_order"
    if predicted != expected:
        return "argument_binding"
    return "none"


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("ascii")).hexdigest()


def verified_validation_rows(evaluation, model, plan, example_count):
    """Return frozen exact rows only after checking the signed cohort identity."""
    body = {key: value for key, value in evaluation.items() if key != "receipt_sha256"}
    if evaluation.get("receipt_sha256") != _sha(body):
        raise ValueError("frozen validation receipt is invalid")
    correct = evaluation["source_input_order"]["candidates"]["frozen"]["program_correct"]
    if (evaluation["candidate_receipt_sha256"] != model.receipt_sha256
            or evaluation["source_order_plan_sha256"] != plan["report_sha256"]
            or evaluation["validation_ids_sha256"] != plan["validation_example_ids_sha256"]
            or len(correct) != example_count
            or any(type(value) is not bool for value in correct)):
        raise ValueError("frozen validation does not match candidate and source-order cohort")
    return correct


def bank_reach_record(bank, target):
    """Grade target-blind alternatives only after the bank is frozen."""
    bank.validate()
    target_sha = target.sha()
    return {
        "bank_receipt_sha256": bank.receipt["receipt_sha256"],
        "candidate_count": len(bank.candidates),
        "exact_target_observed": any(candidate.program.sha() == target_sha
                                     for candidate in bank.candidates),
        "search_complete": bank.receipt["search_complete"],
        "limit_reason": bank.receipt["limit_reason"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--bundle", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", action="append", default=[],
                        help="diagnostic subset of frozen failed validation identities")
    parser.add_argument("--candidate-bank", action="store_true",
                        help="inspect target-blind bounded alternatives on each failed row")
    parser.add_argument("--max-charts", type=int, default=16)
    parser.add_argument("--max-graphs-per-chart", type=int, default=16)
    parser.add_argument("--solve-seconds", type=float, default=10.)
    parser.add_argument("--bank-seconds", type=float, default=20.)
    args = parser.parse_args()
    if (args.max_charts < 1 or args.max_graphs_per_chart < 1
            or not 0 < args.solve_seconds <= 60
            or not 0 < args.bank_seconds <= 300):
        parser.error("candidate-bank allowances must be positive and bounded")
    os.environ.setdefault("AURA_LOG_DIR", str(args.output.parent / "logs"))
    os.environ.setdefault("AURA_STATE_ROOT", str(args.output.parent / "state"))

    from core.learning.semantic_program_compositional_campaign import (
        prepare_compositional_source_training,
    )
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
    )
    from core.learning.semantic_program_feature_materialization import (
        load_standard_semantic_feature_bundle,
    )
    from core.runtime.atomic_writer import atomic_write_bytes_if_absent
    from tools.diagnose_compositional_semantic_transfer import _bundle_arguments

    bundles = {name: load_standard_semantic_feature_bundle(path)
               for name, path in _bundle_arguments(args.bundle).items()}
    examples, plan = prepare_compositional_source_training(bundles, source_order_inputs=True)
    selected = tuple(item for item in examples if item.split == "validation")
    model = compositional_semantic_program_transducer_from_dict(
        json.loads(args.candidate.read_text("ascii")))
    evaluation = json.loads(args.validation.read_text("ascii"))
    correct = verified_validation_rows(evaluation, model, plan, len(selected))
    failed_sources = {item.ir.source_text_sha256 for item, passed in
                      zip(selected, correct, strict=True) if not passed}
    if len(args.source) != len(set(args.source)) or not set(args.source) <= failed_sources:
        raise ValueError("requested source is absent from frozen validation failures")
    rows = []
    for item, passed in zip(selected, correct, strict=True):
        if passed or (args.source and item.ir.source_text_sha256 not in args.source):
            continue
        outcome = model.decode(
            source_token_ids=item.ir.source_token_ids,
            hidden_states=item.hidden_states,
            public_inputs=item.public_inputs,
            source_text_sha256=item.ir.source_text_sha256,
            model_basis_sha256=item.ir.model_basis_receipt_sha256,
        )
        predicted = outcome.ir.to_program() if outcome.ir is not None else None
        expected = item.ir.to_program()
        bank_record = None
        if args.candidate_bank:
            bank = model.decode_candidates(
                source_token_ids=item.ir.source_token_ids,
                hidden_states=item.hidden_states,
                public_inputs=item.public_inputs,
                source_text_sha256=item.ir.source_text_sha256,
                model_basis_sha256=item.ir.model_basis_receipt_sha256,
                max_charts=args.max_charts,
                max_graphs_per_chart=args.max_graphs_per_chart,
                solve_time_limit_s=args.solve_seconds,
                bank_time_limit_s=args.bank_seconds,
            )
            bank_record = bank_reach_record(bank, expected)
        rows.append({
            "source": item.ir.source_text_sha256,
            "construction": item.construction_id,
            "topology": item.topology_id,
            "refusal": outcome.refusal,
            "failure_class": failure_class(predicted, expected),
            "predicted_program": predicted.to_dict() if predicted is not None else None,
            "expected_program": expected.to_dict(),
            "bank": bank_record,
        })
        print(json.dumps({"completed": len(rows), "source": item.ir.source_text_sha256,
                          "failure_class": rows[-1]["failure_class"]}), flush=True)
    counts = Counter(row["failure_class"] for row in rows)
    body = {
        "schema": "aura.semantic_source_order_miss_profile.v1",
        "candidate_receipt_sha256": model.receipt_sha256,
        "validation_receipt_sha256": evaluation["receipt_sha256"],
        "source_order_plan_sha256": plan["report_sha256"],
        "development_only": True,
        "serving_authority": False,
        "failed_validation_rows": len(rows),
        "selected_sources": sorted(args.source),
        "failure_counts": dict(sorted(counts.items())),
        "bank_allowance": ({"max_charts": args.max_charts,
                            "max_graphs_per_chart": args.max_graphs_per_chart,
                            "solve_seconds": args.solve_seconds,
                            "bank_seconds": args.bank_seconds} if args.candidate_bank else None),
        "rows": rows,
    }
    payload = (json.dumps({**body, "receipt_sha256": _sha(body)}, sort_keys=True,
                          separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")
    if not atomic_write_bytes_if_absent(args.output, payload, mode=0o400):
        raise FileExistsError(args.output)
    print(json.dumps({"output": str(args.output), "failure_counts": body["failure_counts"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
